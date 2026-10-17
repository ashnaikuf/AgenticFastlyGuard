"""Fastly Next-Gen WAF Security Agent Pipeline.

Processes incoming HTTP request logs (CSV format) containing security signals,
retrieves embedded WAF rule definitions from per-rule Markdown files
(`rules/RULE-*.md`) indexed in a vector store, and evaluates condition
logic using a LangGraph workflow with an LLM agent.
"""

import json
import logging
import os
import re
from functools import lru_cache, partial
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import NAMESPACE_URL, uuid4, uuid5
from typing_extensions import TypedDict

import pandas as pd
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

# LangChain components
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import MarkdownHeaderTextSplitter

# LangGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# Load environment variables
load_dotenv()
logger = logging.getLogger("waf_agent_pipeline")


# ==============================================================================
# Pydantic Schemas & State Types
# ==============================================================================

# Sentinel rule ID returned when no indexed rule covers the request context.
# Preferred over silently reporting the highest-scoring retrieved rule, which
# produces a confident-looking recommendation with no basis in the rule logic.
NO_RULE_MATCH = "NO_RULE_MATCH"

# Verification outcomes recorded on the final result by the `verify_match` node.
VERIFY_FULL = "verified"            # every condition in the rule was machine-checked
VERIFY_SIGNALS_ONLY = "signals_verified"  # signals checked; non-signal conditions were not
VERIFY_FAILED = "failed"            # the reported match contradicts the rule or the log
VERIFY_NO_MATCH = "no_match"        # NO_RULE_MATCH; nothing to verify


class WAFRuleEvaluation(BaseModel):
    """Structured output schema produced by the evaluator (LLM or deterministic).

    Deliberately excludes the verification fields: the evaluator must not be
    able to certify its own output. `verify_match` re-checks these claims
    independently and emits a `WAFRuleMatchResult`.
    """

    matched_rule_id: str = Field(
        ...,
        description=(
            "The matched rule identifier, e.g., RULE-002. Use 'NO_RULE_MATCH' "
            "when no retrieved rule's conditions cover the request context."
        ),
    )
    action: str = Field(
        ...,
        description="The action associated with the matched rule, e.g., BLOCK",
    )
    detected_signals: List[str] = Field(
        ...,
        description="List of detected attack signals triggering the rule, e.g. ['XSS', 'SQLI']",
    )
    explanation: str = Field(
        ...,
        description="Explanation of why the rule matched the request context",
    )
    yaml_definition: str = Field(
        ...,
        description="Raw YAML condition block from the matched rule",
    )


class WAFRuleMatchResult(WAFRuleEvaluation):
    """Final pipeline output: an evaluation plus its independent verification."""

    verified: bool = Field(
        default=False,
        description="True when the reported match survived independent verification",
    )
    verification_level: str = Field(
        default=VERIFY_FAILED,
        description=(
            "One of 'verified', 'signals_verified', 'failed', 'no_match'. "
            "'signals_verified' means the rule declares conditions beyond "
            "signal_exists that are not yet machine-checked."
        ),
    )
    verification_notes: List[str] = Field(
        default_factory=list,
        description="Human-readable findings from verification; empty when fully verified",
    )
    evaluator: str = Field(
        default="",
        description="Which evaluator produced the match: 'llm' or 'deterministic'",
    )


class WAFState(TypedDict, total=False):
    """State schema tracked across LangGraph nodes.

    NEW (Phase 4): the Qdrant client is deliberately NOT part of the state. The
    graph is compiled with a checkpointer, which serializes state at every node
    boundary, and a live client is not serializable. It is bound to the
    retrieval node at graph construction time instead.
    """

    csv_path: str
    row_index: int
    log_context: Dict[str, Any]
    retrieved_rules: List[Dict[str, Any]]
    final_output: Dict[str, Any]


# ==============================================================================
# Step 1: Ingestion & Vector Indexing
# ==============================================================================

@lru_cache(maxsize=None)
def get_embeddings(provider: str = "auto", dimensions: Optional[int] = None):
    """Instantiate OpenAI or HuggingFace embeddings based on availability.

    Supports automatic selection: if OPENAI_API_KEY is configured,
    OpenAIEmbeddings is used; otherwise HuggingFaceEmbeddings runs locally.
    OpenAI dimensions can be set to match an existing Qdrant collection.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if provider == "openai" or (provider == "auto" and openai_key):
        from langchain_openai import OpenAIEmbeddings
        logger.info(
            "Initializing OpenAIEmbeddings (text-embedding-3-small, dimensions=%s)",
            dimensions or "default",
        )
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=dimensions,
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info("Initializing HuggingFaceEmbeddings (all-MiniLM-L6-v2)")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_qdrant_vector_size(client: QdrantClient, collection_name: str) -> int:
    """Return the configured size of the collection's named dense vector."""
    collection = client.get_collection(collection_name)
    vectors_config = collection.config.params.vectors
    dense_vector_config = (
        vectors_config.get("dense_vector")
        if isinstance(vectors_config, dict)
        else None
    )
    if dense_vector_config is None:
        raise ValueError(
            f"Qdrant collection {collection_name!r} must define a named "
            "'dense_vector' vector."
        )
    return dense_vector_config.size


def extract_yaml_blocks_by_rule(raw_text: str) -> Dict[str, str]:
    """Extract raw YAML condition blocks from the markdown file by rule ID."""
    yaml_blocks = {}
    pattern = re.compile(
        r"## Rule:\s*([A-Za-z0-9_-]+)[^\n]*\n.*?```yaml\s*\n(.*?)\n```",
        re.DOTALL,
    )
    for match in pattern.finditer(raw_text):
        rule_id = match.group(1).strip()
        raw_yaml = match.group(2)
        yaml_blocks[rule_id] = raw_yaml
    return yaml_blocks


def build_waf_rule_vectorstore(
    rules_dir: str = "rules",
    embedding_provider: str = "auto",
) -> QdrantClient:
    """Split the WAF rule Markdown files, extract metadata, and index into Qdrant.

    Reads every ``RULE-*.md`` file in ``rules_dir`` (one rule per file by
    convention; a file may hold several ``## Rule:`` sections).

    Metadata extracted per chunk:
    - rule_id (e.g. RULE-001, RULE-002)
    - action (e.g. BLOCK)
    - signals (list of signal names defined in condition logic)
    - yaml_definition (raw YAML block)
    - source (originating Markdown filename)
    """
    rules_path = Path(rules_dir)
    if not rules_path.is_dir():
        raise FileNotFoundError(f"Rules directory not found: {rules_dir}")

    md_files = sorted(rules_path.glob("RULE-*.md"))
    if not md_files:
        raise FileNotFoundError(f"No RULE-*.md files found in: {rules_dir}")

    # Use MarkdownHeaderTextSplitter to split by '## Rule:'
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("## Rule:", "rule_header")],
        strip_headers=False,
    )

    processed_docs: List[Document] = []
    for md_file in md_files:
        raw_markdown = md_file.read_text(encoding="utf-8")
        raw_yaml_map = extract_yaml_blocks_by_rule(raw_markdown)

        for doc in splitter.split_text(raw_markdown):
            header = doc.metadata.get("rule_header")
            if not header:
                # Skip introductory or non-rule sections
                continue

            # Extract rule_id from header (e.g. "RULE-002 - OWASP Core Attack Prevention")
            rule_id_match = re.search(r"\b(RULE-\d+)\b", header)
            rule_id = rule_id_match.group(1) if rule_id_match else ""

            # Extract Action
            action_match = re.search(r"Action\*\*:\s*`?([A-Z_]+)`?", doc.page_content)
            action = action_match.group(1) if action_match else "BLOCK"

            # Raw YAML block
            raw_yaml = raw_yaml_map.get(rule_id, "")
            if not raw_yaml:
                yaml_match = re.search(r"```yaml\s*\n(.*?)\n```", doc.page_content, re.DOTALL)
                raw_yaml = yaml_match.group(1) if yaml_match else ""

            # Parse signals from YAML if available
            signals: List[str] = []
            if raw_yaml:
                try:
                    parsed_yaml = yaml.safe_load(raw_yaml)
                    if isinstance(parsed_yaml, dict):
                        action = parsed_yaml.get("action", action)
                        rule_id = parsed_yaml.get("rule_id", rule_id)
                        # Traverse conditions
                        conditions = parsed_yaml.get("conditions", {})
                        if isinstance(conditions, dict):
                            rules_list = conditions.get("rules", [])
                            if isinstance(rules_list, list):
                                for r in rules_list:
                                    if isinstance(r, dict) and "signals" in r:
                                        signals.extend(r["signals"])
                except Exception as e:
                    logger.warning("Could not parse YAML for rule %s: %s", rule_id, e)

            # Fallback signal regex extraction if YAML did not provide signals
            if not signals:
                sig_section = re.search(r"signals:\s*\n((?:\s*-\s*[^\n]+\n)+)", doc.page_content)
                if sig_section:
                    signals = [
                        s.strip(' -"\n')
                        for s in sig_section.group(1).splitlines()
                        if s.strip()
                    ]

            metadata = {
                "rule_id": rule_id,
                "action": action,
                "signals": signals,
                "signals_str": " ".join(signals),
                "yaml_definition": raw_yaml,
                "source": md_file.name,
                "content_hash": sha256(doc.page_content.encode("utf-8")).hexdigest(),
            }

            # Build clean search content emphasizing rule intent, signals, and description
            doc.metadata.update(metadata)
            processed_docs.append(doc)

    if not processed_docs:
        raise ValueError(f"No WAF rule sections found in rules directory: {rules_dir}")

    api_key = os.getenv("QDRANT_APIKEY")
    database_url = os.getenv("CLUSTER_ENDPOINT")
    if not api_key:
        raise ValueError("QDRANT_APIKEY is not set in the environment.")
    if not database_url:
        raise ValueError("CLUSTER_ENDPOINT is not set in the environment.")

    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "fastly_wafrules")
    client = QdrantClient(
        url=database_url,
        api_key=api_key,
        timeout=30,
    )
    collection_exists = client.collection_exists(collection_name)
    expected_vector_size = (
        get_qdrant_vector_size(client, collection_name)
        if collection_exists
        else None
    )

    # Deterministic point IDs: one point per rule_id, so re-running overwrites
    # in place instead of creating duplicates.
    point_ids = {
        doc.metadata["rule_id"]: str(
            uuid5(NAMESPACE_URL, f"{collection_name}:{doc.metadata['rule_id']}")
        )
        for doc in processed_docs
    }

    # Compare each rule's content hash against the one already stored in Qdrant
    # so unchanged rules are never re-embedded.
    stored_hashes: Dict[str, Optional[str]] = {}
    if collection_exists:
        stored_points = client.retrieve(
            collection_name=collection_name,
            ids=list(point_ids.values()),
            with_payload=True,
        )
        for point in stored_points:
            payload_metadata = (point.payload or {}).get("metadata") or {}
            stored_hashes[str(point.id)] = payload_metadata.get("content_hash")

    changed_docs = [
        doc
        for doc in processed_docs
        if stored_hashes.get(point_ids[doc.metadata["rule_id"]])
        != doc.metadata["content_hash"]
    ]

    if changed_docs:
        embeddings = get_embeddings(
            provider=embedding_provider,
            dimensions=expected_vector_size,
        )
        vectors = embeddings.embed_documents([doc.page_content for doc in changed_docs])

        if not collection_exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense_vector": models.VectorParams(
                        size=len(vectors[0]),
                        distance=models.Distance.COSINE,
                    )
                },
            )
        elif expected_vector_size != len(vectors[0]):
            raise ValueError(
                f"Qdrant collection {collection_name!r} expects vectors of size "
                f"{expected_vector_size}, but the selected embedding model "
                f"produces vectors of size {len(vectors[0])}."
            )

        points = [
            models.PointStruct(
                id=point_ids[doc.metadata["rule_id"]],
                vector={"dense_vector": vector},
                payload={
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                },
            )
            for doc, vector in zip(changed_docs, vectors)
        ]
        client.upsert(collection_name=collection_name, points=points, wait=True)
        logger.info(
            "Indexed %d of %d WAF rules (%d unchanged, skipped embedding): %s",
            len(changed_docs),
            len(processed_docs),
            len(processed_docs) - len(changed_docs),
            ", ".join(doc.metadata["rule_id"] for doc in changed_docs),
        )
    else:
        logger.info(
            "All %d WAF rules already indexed and unchanged; skipped embedding.",
            len(processed_docs),
        )

    ensure_payload_indexes(client, collection_name)
    delete_orphaned_rule_points(client, collection_name, set(point_ids.values()))
    return client


# Number of rule candidates retrieval returns to the evaluator. The evaluator can
# only match a rule that was retrieved, so this is a recall floor, not a tuning
# knob: at the previous value of 4 against a 6-rule corpus, RULE-002 fell out of
# the window on genuine SQLi rows and the match was silently fabricated instead.
# Keep this at or above the rule count until hybrid retrieval and re-ranking land.
RETRIEVAL_LIMIT = int(os.getenv("WAF_RETRIEVAL_LIMIT", "10"))

# Payload fields the retrieval step filters on. Qdrant rejects a filtered query
# with 400 ("Index required but not found") unless the field is indexed.
FILTERABLE_PAYLOAD_FIELDS = ("metadata.action", "metadata.rule_id")


def ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """Create the keyword payload indexes required by filtered queries.

    Idempotent: re-creating an existing index is a no-op on the Qdrant side,
    and indexes do not survive a collection being dropped and recreated, so
    this runs on every ingest rather than only at collection creation.
    """
    existing = client.get_collection(collection_name).payload_schema or {}
    for field in FILTERABLE_PAYLOAD_FIELDS:
        if field in existing:
            continue
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        logger.info("Created keyword payload index on %r in %r.", field, collection_name)


def delete_orphaned_rule_points(
    client: QdrantClient,
    collection_name: str,
    expected_ids: Set[str],
) -> int:
    """Delete points whose rule no longer exists in the Markdown rules directory.

    Point IDs are derived from ``rule_id``, so a renamed or deleted rule would
    otherwise leave a stale point behind that the agent could still retrieve.
    Returns the number of points deleted.
    """
    orphan_ids: List[str] = []
    next_offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=next_offset,
            with_payload=False,
            with_vectors=False,
        )
        orphan_ids.extend(
            str(point.id) for point in batch if str(point.id) not in expected_ids
        )
        if next_offset is None:
            break

    if orphan_ids:
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=orphan_ids),
            wait=True,
        )
        logger.info(
            "Deleted %d orphaned point(s) from %r with no matching rule file.",
            len(orphan_ids),
            collection_name,
        )

    return len(orphan_ids)


# ==============================================================================
# Step 2: CSV Signal Extraction
# ==============================================================================

# --- Signal taxonomy (data-driven) -------------------------------------------
# Categories live in rules/signal_categories.yaml rather than in Python. The
# previous hardcoded sets had drifted from the log data: SANS and SUSPECTED-BOT
# were absent from every category, so ~45% of requests_maliciousip.csv carried
# no recognized threat and fell through to a blind fallback match.
SIGNAL_CATEGORIES_FILENAME = "signal_categories.yaml"

# Signals that carry no routing information on their own.
CATEGORY_GENERIC = "generic"
# Load-bearing for match precedence: see `_rule_match_precedence`.
CATEGORY_IP_REPUTATION = "ip_reputation"


@lru_cache(maxsize=None)
def load_signal_categories(rules_dir: str = "rules") -> Dict[str, str]:
    """Map every configured signal ID to its category name.

    Returns a flat ``{normalized_signal_id: category}`` lookup. Missing or
    unreadable config degrades to an empty map rather than raising: callers
    treat an uncategorized signal as uncategorized, which `coverage_report`
    surfaces instead of hiding.
    """
    config_path = Path(rules_dir) / SIGNAL_CATEGORIES_FILENAME
    if not config_path.is_file():
        logger.warning(
            "Signal category config not found at %s; signals will be uncategorized.",
            config_path,
        )
        return {}

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.error("Could not parse %s: %s", config_path, exc)
        return {}

    mapping: Dict[str, str] = {}
    for category, members in (config.get("categories") or {}).items():
        for signal in members or []:
            mapping[normalize_signal_id(signal)] = category
    return mapping


@lru_cache(maxsize=None)
def _load_signal_prefixes(rules_dir: str = "rules") -> List[tuple]:
    """Prefix fallbacks as ``(prefix, category)`` pairs, longest prefix first."""
    config_path = Path(rules_dir) / SIGNAL_CATEGORIES_FILENAME
    if not config_path.is_file():
        return []
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    pairs = [
        (normalize_signal_id(prefix), category)
        for category, prefixes in (config.get("prefixes") or {}).items()
        for prefix in prefixes or []
    ]
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


def signal_category(signal: Any, rules_dir: str = "rules") -> str:
    """Category of a signal, or "" when it is not covered by the config."""
    normalized = normalize_signal_id(signal)
    if not normalized:
        return ""
    direct = load_signal_categories(rules_dir).get(normalized)
    if direct:
        return direct
    for prefix, category in _load_signal_prefixes(rules_dir):
        if normalized.startswith(prefix):
            return category
    return ""


def signals_in_category(category: str, rules_dir: str = "rules") -> Set[str]:
    """All configured signal IDs belonging to ``category``."""
    return {
        signal
        for signal, cat in load_signal_categories(rules_dir).items()
        if cat == category
    }


def normalize_signal_id(raw: Any) -> str:
    """Normalize a raw signal ID to the canonical upper/hyphenated spelling.

    Fastly writes IP signals as ``SITE-FLAGGED-IP``/``SIGSCI-IP`` in logs but as
    ``SigSci IP``/``Malicious IP`` in rule YAML; normalizing both sides lets them
    compare equal.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    return re.sub(r"[\s_]+", "-", text).upper()


def is_ip_threat_signal(signal: Any, rules_dir: str = "rules") -> bool:
    """True when a signal denotes an IP reputation / blocklist threat.

    Now resolved from the signal category config rather than a hardcoded set.
    """
    return signal_category(signal, rules_dir) == CATEGORY_IP_REPUTATION


def is_generic_signal(signal: Any, rules_dir: str = "rules") -> bool:
    """True when a signal carries no routing information on its own."""
    return signal_category(signal, rules_dir) == CATEGORY_GENERIC


def _summation_marks_blocklisted_ip(summation: Any) -> bool:
    """True when the `summation` column reports the client IP as blocklisted."""
    if not isinstance(summation, str) or not summation.strip():
        return False
    # Match "blocklisted":"true" / "blocklisted": true, case-insensitively.
    return bool(re.search(r'"blocklisted"\s*:\s*"?true"?', summation, re.IGNORECASE))


def _extract_row_signal_details(
    signals_raw: Any, summation_raw: Any = None
) -> List[Dict[str, str]]:
    """Parse one row's `signals` JSON into detail dicts, de-duplicated by ID.

    NEW: retains each signal's `location`, `value`, and `detector` alongside its
    `id`. Fastly records an IP reputation hit as
    ``{"id": "SANS", "location": "IP", "value": "205.210.31.135",
    "detector": "SansRule"}``; keeping those fields lets verification sanity-check
    the signal's scope and lets explanations name the offending value instead of
    only the signal tag.

    Existing attack signal IDs (XSS, SQLI, ...) are preserved verbatim so that
    current OWASP and path-based rule evaluation is unaffected.
    """
    details: List[Dict[str, str]] = []
    seen: Set[str] = set()

    if pd.notna(signals_raw) and isinstance(signals_raw, str) and signals_raw.strip():
        try:
            parsed = json.loads(signals_raw)
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict) or not item.get("id"):
                        continue
                    signal_id = item["id"]
                    # Normalize IP-related IDs to canonical tags; all other
                    # signal IDs keep their original spelling.
                    if is_ip_threat_signal(signal_id):
                        signal_id = normalize_signal_id(signal_id)
                    if signal_id in seen:
                        continue
                    seen.add(signal_id)
                    details.append({
                        "id": signal_id,
                        "location": str(item.get("location") or ""),
                        "value": str(item.get("value") or ""),
                        "detector": str(item.get("detector") or ""),
                        "detector_scope": str(item.get("detector_scope") or ""),
                    })
        except Exception as err:
            logger.debug("Failed parsing signals JSON: %s", err)

    # Fall back to the `summation` blocklist flag only when the row carries no
    # explicit IP signal. `"blocklisted":"true"` is set on effectively every
    # logged request, so appending it unconditionally would mark all traffic as
    # an IP threat and starve the attack-specific rules.
    if not any(is_ip_threat_signal(d["id"]) for d in details):
        if _summation_marks_blocklisted_ip(summation_raw):
            details.append({
                "id": "BLOCKLISTED-IP",
                "location": "IP",
                "value": "",
                "detector": "summation",
                "detector_scope": "",
            })

    return details


def signal_ids(details: List[Dict[str, str]]) -> List[str]:
    """Flatten signal detail dicts back to the plain ID list used for matching."""
    return [d["id"] for d in details]


def _extract_row_signals(signals_raw: Any, summation_raw: Any = None) -> List[str]:
    """Parse one row's `signals` JSON into a plain list of signal IDs."""
    return signal_ids(_extract_row_signal_details(signals_raw, summation_raw))


def extract_signals_from_csv(csv_path: str) -> List[List[str]]:
    """Read the incoming CSV file using pandas and extract signal IDs for each row.

    The CSV contains a `signals` column with JSON strings like:
    `[{"id":"XSS","location":"QUERYSTRING",...}]`

    IP threat tags (`SITE-FLAGGED-IP`, `SIGSCI-IP`) are normalized to their
    canonical spelling, and the `summation` column is inspected for a
    `"blocklisted":"true"` flag as a fallback IP threat indicator.

    Returns a list of detected signal IDs for each row.
    """
    return [signal_ids(d) for d in extract_signal_details_from_csv(csv_path)]


def extract_signal_details_from_csv(csv_path: str) -> List[List[Dict[str, str]]]:
    """Per-row signal details (`id`, `location`, `value`, `detector`) for a CSV.

    The detail-preserving counterpart of `extract_signals_from_csv`, which keeps
    returning bare ID lists for callers that only match on signal names.
    """
    df = pd.read_csv(csv_path)
    all_details: List[List[Dict[str, str]]] = []

    # NEW: read `summation` alongside `signals` so the blocklist flag is available.
    summations = df["summation"] if "summation" in df.columns else None

    for position, val in enumerate(df.get("signals", [])):
        summation_raw = summations.iloc[position] if summations is not None else None
        all_details.append(_extract_row_signal_details(val, summation_raw))

    return all_details


# ==============================================================================
# Step 3: LangGraph Agent Pipeline
# ==============================================================================

def build_log_context(row: Any, row_index: int = 0) -> Dict[str, Any]:
    """Build the request context the pipeline evaluates, from one CSV row.

    Shared by `extract_log_context` and by callers that already hold the row
    (the Streamlit app reads the CSV once into a DataFrame), so the context is
    assembled in exactly one place and cannot drift between them.
    """
    # Shares `_extract_row_signal_details` with extract_signals_from_csv so IP
    # tag normalization and the `summation` blocklist fallback apply identically.
    signal_details = _extract_row_signal_details(row.get("signals"), row.get("summation"))
    signals = signal_ids(signal_details)

    return {
        "signals": signals,
        # Full signal records (location/value/detector) for verification and for
        # explanations that can name the flagged value.
        "signal_details": signal_details,
        "path": str(row.get("path") or ""),
        "domain": str(row.get("server_name") or row.get("server_hostname") or ""),
        "method": str(row.get("method") or "GET"),
        # NEW (Phase 3): required by the `user_agent_like` / `user_agent_equals`
        # conditions, which were previously never evaluated.
        "user_agent": str(row.get("user_agent") or ""),
        "remote_ip": str(row.get("remote_ip") or ""),
        "uri": str(row.get("uri") or ""),
        "row_index": row_index,
        "ip_threat_signals": [s for s in signals if is_ip_threat_signal(s)],
    }


def extract_log_context(state: WAFState) -> Dict[str, Any]:
    """Node: Reads CSV and extracts signals, path, domain, and HTTP method."""
    if state.get("log_context"):
        return {"log_context": state["log_context"]}

    csv_path = state.get("csv_path", "requests_sqli_xss_cmdexe.csv")
    row_index = state.get("row_index", 0)

    df = pd.read_csv(csv_path)
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index {row_index} out of bounds for CSV of length {len(df)}")

    return {"log_context": build_log_context(df.iloc[row_index], row_index)}


def retrieve_waf_rules(
    state: WAFState, vectorstore: Optional[QdrantClient] = None
) -> Dict[str, Any]:
    """Node: Queries the vector store with extracted signals string and applies metadata filtering.

    `vectorstore` is bound at graph construction time rather than read from
    state, which must stay serializable for checkpointing.
    """
    log_context = state.get("log_context", {})
    signals = log_context.get("signals", [])
    path = log_context.get("path", "")

    if vectorstore is None:
        vectorstore = build_waf_rule_vectorstore()

    # Form query string from detected signals and request path.
    # NEW: IP threat tags are no longer stripped here (SITE-FLAGGED-IP used to be
    # discarded), so an IP-only request can still retrieve RULE-001.
    query_terms = [s for s in signals if not is_generic_signal(s)]
    if not query_terms:
        query_terms = signals

    ip_threat_signals = [s for s in signals if is_ip_threat_signal(s)]
    query_str = f"Signals: {' '.join(query_terms)} Path: {path}"
    # NEW: only bias the query toward IP reputation when the IP hit is the *sole*
    # indicator. Appending it whenever any IP signal was present polluted nearly
    # every query — SITE-FLAGGED-IP rides along on 991/1000 rows of
    # requests_sqli_xss_cmdexe.csv — which pushed RULE-002 out of the result
    # window for genuine SQLi/XSS rows.
    non_ip_terms = [s for s in query_terms if not is_ip_threat_signal(s)]
    if ip_threat_signals and not non_ip_terms:
        # Natural-language IP reputation terms so RULE-001's description
        # ("malicious IP", "reputation", "blocklist") scores well semantically.
        query_str += " Malicious IP reputation blocklist flagged client address"

    # Query vector store with metadata filter for action: BLOCK
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "fastly_wafrules")
    vector_size = get_qdrant_vector_size(vectorstore, collection_name)
    query_vector = get_embeddings(dimensions=vector_size).embed_query(query_str)
    search_result = vectorstore.query_points(
        collection_name=collection_name,
        query=query_vector,
        using="dense_vector",
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.action",
                    match=models.MatchValue(value="BLOCK"),
                )
            ]
        ),
        limit=RETRIEVAL_LIMIT,
        with_payload=True,
    )

    retrieved_rules = []
    for point in search_result.points:
        payload = point.payload or {}
        metadata = payload.get("metadata", {})
        retrieved_rules.append({
            "rule_id": metadata.get("rule_id", ""),
            "action": metadata.get("action", "BLOCK"),
            "signals": metadata.get("signals", []),
            "yaml_definition": metadata.get("yaml_definition", ""),
            "page_content": payload.get("page_content", ""),
        })

    return {"retrieved_rules": retrieved_rules}


# ==============================================================================
# Step 3b: Data-driven condition evaluation
# ==============================================================================

# Condition types the evaluator understands. A rule declaring anything outside
# this set is reported as unevaluable rather than silently treated as matched.
SUPPORTED_CONDITION_TYPES = {
    "signal_exists",
    "path_contains",
    "path_equals",
    "method_equals",
    "user_agent_like",
    "user_agent_equals",
}

# Rule-level `type:` values that cannot be decided from a single request.
STATEFUL_RULE_TYPES = {"rate_limit"}


class ConditionOutcome(BaseModel):
    """Result of evaluating one rule's conditions against one request."""

    matched: bool = False
    matched_signals: List[str] = Field(default_factory=list)
    # Human-readable reasons each satisfied condition passed, for explanations.
    reasons: List[str] = Field(default_factory=list)
    # Condition types present in the rule that this evaluator cannot decide.
    unevaluable: List[str] = Field(default_factory=list)
    # True when the rule's conditions hold but the action depends on state
    # (request rate) that a single log row cannot establish.
    stateful_gate: str = ""


def _as_list(value: Any) -> List[str]:
    """Normalize a YAML `value`/`values` field to a list of strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _wildcard_matches(pattern: str, text: str) -> bool:
    """Case-insensitive `*` wildcard match, as used by `user_agent_like`."""
    regex = "^" + ".*".join(re.escape(part) for part in pattern.split("*")) + "$"
    return bool(re.match(regex, text, re.IGNORECASE))


def _evaluate_condition(
    condition: Dict[str, Any], log_context: Dict[str, Any]
) -> tuple:
    """Evaluate a single condition.

    Returns ``(matched, matched_signals, reason, unevaluable_type)``.
    """
    ctype = condition.get("type", "")
    path = str(log_context.get("path") or "")
    method = str(log_context.get("method") or "")
    user_agent = str(log_context.get("user_agent") or "")
    signals = log_context.get("signals") or []

    if ctype == "signal_exists":
        declared = {normalize_signal_id(s) for s in _as_list(condition.get("signals"))}
        present = {normalize_signal_id(s): s for s in signals}
        hits = sorted(present[n] for n in declared & set(present))
        # `operator` governs ANY vs ALL across the declared signals.
        operator = str(condition.get("operator", "ANY")).upper()
        if operator == "ALL":
            matched = bool(declared) and declared <= set(present)
        else:
            matched = bool(hits)
        reason = (
            f"signal_exists({operator}) satisfied by {', '.join(repr(h) for h in hits)}"
            if matched else ""
        )
        return matched, hits, reason, ""

    if ctype == "path_contains":
        needles = _as_list(condition.get("values") or condition.get("value"))
        hit = next((n for n in needles if n and n in path), "")
        return bool(hit), [], (f"path {path!r} contains {hit!r}" if hit else ""), ""

    if ctype == "path_equals":
        wanted = _as_list(condition.get("value") or condition.get("values"))
        matched = path in wanted
        return matched, [], (f"path equals {path!r}" if matched else ""), ""

    if ctype == "method_equals":
        wanted = [m.upper() for m in _as_list(condition.get("value") or condition.get("values"))]
        matched = method.upper() in wanted
        return matched, [], (f"method is {method!r}" if matched else ""), ""

    if ctype == "user_agent_equals":
        wanted = [u.lower() for u in _as_list(condition.get("values") or condition.get("value"))]
        matched = user_agent.lower() in wanted
        return matched, [], (f"user_agent equals {user_agent!r}" if matched else ""), ""

    if ctype == "user_agent_like":
        patterns = _as_list(condition.get("value") or condition.get("values"))
        hit = next((p for p in patterns if _wildcard_matches(p, user_agent)), "")
        return bool(hit), [], (f"user_agent {user_agent!r} matches {hit!r}" if hit else ""), ""

    # Unknown condition type: report it rather than defaulting either way.
    return False, [], "", ctype or "unknown"


def evaluate_rule_conditions(
    rule: Dict[str, Any], log_context: Dict[str, Any]
) -> ConditionOutcome:
    """Evaluate a rule's YAML conditions against a request context.

    Replaces the per-rule ``if rule_id == "RULE-00X"`` ladder that hardcoded
    domains, paths, and signal lists in Python while the same facts were already
    declared in the rule YAML. Honors the ``conditions.operator`` ALL/ANY
    combinator and the rule-level ``domain`` scope.
    """
    raw_yaml = rule.get("yaml_definition", "")
    if not raw_yaml:
        return ConditionOutcome(unevaluable=["missing_yaml"])

    try:
        parsed = yaml.safe_load(raw_yaml)
    except Exception as exc:
        logger.warning("Could not parse YAML for %s: %s", rule.get("rule_id"), exc)
        return ConditionOutcome(unevaluable=["unparseable_yaml"])
    if not isinstance(parsed, dict):
        return ConditionOutcome(unevaluable=["unparseable_yaml"])

    # Rule-level domain scope (RULE-004 declares `domain:` outside `conditions`).
    rule_domain = str(parsed.get("domain") or "")
    if rule_domain and str(log_context.get("domain") or "") != rule_domain:
        return ConditionOutcome(
            matched=False,
            reasons=[f"rule is scoped to domain {rule_domain!r}"],
        )

    conditions = parsed.get("conditions") or {}
    condition_list = conditions.get("rules") if isinstance(conditions, dict) else None
    if not isinstance(condition_list, list) or not condition_list:
        return ConditionOutcome(unevaluable=["no_conditions"])

    operator = str(conditions.get("operator", "ALL")).upper()

    results: List[bool] = []
    matched_signals: List[str] = []
    reasons: List[str] = []
    unevaluable: List[str] = []

    for condition in condition_list:
        if not isinstance(condition, dict):
            continue
        matched, hits, reason, unknown = _evaluate_condition(condition, log_context)
        if unknown:
            unevaluable.append(unknown)
            continue
        results.append(matched)
        if matched:
            matched_signals.extend(hits)
            if reason:
                reasons.append(reason)

    if not results:
        return ConditionOutcome(unevaluable=unevaluable or ["no_evaluable_conditions"])

    matched = all(results) if operator == "ALL" else any(results)

    # An ALL rule containing an unevaluable condition cannot be confirmed.
    if matched and operator == "ALL" and unevaluable:
        matched = False

    # Rate-limit rules describe a threshold over many requests; a single row can
    # satisfy the conditions but never establish that the limit was exceeded.
    stateful_gate = ""
    if matched and str(parsed.get("type") or "") in STATEFUL_RULE_TYPES:
        stateful_gate = str(parsed.get("type"))

    return ConditionOutcome(
        matched=matched,
        matched_signals=sorted(set(matched_signals)),
        reasons=reasons,
        unevaluable=sorted(set(unevaluable)),
        stateful_gate=stateful_gate,
    )


def _rule_match_precedence(outcome: ConditionOutcome) -> int:
    """Sort key deciding which matching rule to report (lower wins).

    Generalizes the previous hardcoded ``if rule_id == "RULE-001": continue``:
    a rule that matched *only* on IP reputation signals yields to any rule that
    matched on other grounds, so a blocklisted IP carrying a real payload still
    reports the attack-specific rule.
    """
    if outcome.stateful_gate:
        # Conditions hold but the action depends on request rate; prefer a rule
        # that can be confirmed from this request alone.
        return 2
    if outcome.matched_signals and all(
        is_ip_threat_signal(s) for s in outcome.matched_signals
    ):
        return 1
    return 0


def _deterministic_evaluate_rule(
    retrieved_rules: List[Dict[str, Any]],
    log_context: Dict[str, Any],
) -> WAFRuleEvaluation:
    """Deterministic evaluator: match a request against retrieved rule conditions.

    NEW (Phase 3): evaluates each rule's YAML conditions via
    `evaluate_rule_conditions` instead of the previous per-rule
    ``if rule_id == "RULE-00X"`` ladder, which hardcoded sensitive paths, the
    RULE-004 domain, and RULE-001's deferral in Python while the same facts were
    already declared in the rule files. Adding a rule no longer requires a code
    change, and `user_agent_*` conditions are now honored.

    Ensures predictable behavior and tests without requiring live external API keys.
    """
    signals = log_context.get("signals", [])
    path = log_context.get("path", "")

    # Evaluate every retrieved rule, then pick among the matches by precedence
    # rather than by retrieval rank.
    matches: List[tuple] = []
    for rule in retrieved_rules:
        outcome = evaluate_rule_conditions(rule, log_context)
        if outcome.matched:
            matches.append((_rule_match_precedence(outcome), rule, outcome))

    if matches:
        matches.sort(key=lambda m: m[0])
        _, rule, outcome = matches[0]
        rule_id = rule.get("rule_id", "")
        reason_text = "; ".join(outcome.reasons) if outcome.reasons else "conditions satisfied"

        explanation = f"{rule_id} matched: {reason_text}."
        if outcome.stateful_gate:
            explanation += (
                f" Note: {rule_id} is a {outcome.stateful_gate} rule — its conditions "
                "hold for this request, but the configured threshold is evaluated "
                "across many requests and cannot be confirmed from a single log row."
            )
        if outcome.matched_signals and all(
            is_ip_threat_signal(s) for s in outcome.matched_signals
        ):
            # Name the flagged address when the signal detail carries it.
            details = {d.get("id"): d for d in (log_context.get("signal_details") or [])}
            flagged = next(
                (
                    details[s].get("value")
                    for s in outcome.matched_signals
                    if details.get(s) and details[s].get("value")
                ),
                "",
            )
            if flagged:
                explanation = (
                    f"{rule_id} matched: client address {flagged} is flagged by "
                    f"{', '.join(repr(s) for s in outcome.matched_signals)}."
                )

        return WAFRuleEvaluation(
            matched_rule_id=rule_id,
            action=rule.get("action", "BLOCK"),
            detected_signals=outcome.matched_signals,
            explanation=explanation,
            yaml_definition=rule.get("yaml_definition", ""),
        )

    # No rule's conditions covered this request. Previously this returned
    # `retrieved_rules[0]` (or RULE-002) with the explanation "Request evaluated
    # against WAF rules." — a confident recommendation chosen by retrieval
    # ranking rather than by any condition actually matching. Reporting the gap
    # is the useful answer: it identifies traffic needing a new rule.
    candidates = ", ".join(
        f"{r.get('rule_id')}" for r in retrieved_rules if r.get("rule_id")
    ) or "none"
    return WAFRuleEvaluation(
        matched_rule_id=NO_RULE_MATCH,
        action="NONE",
        detected_signals=[],
        explanation=(
            f"No indexed WAF rule covers this request. Signals present: "
            f"{', '.join(signals) if signals else 'none'}. "
            f"Path: '{path}'. Closest retrieved candidates: {candidates}. "
            "Consider authoring a rule for these signals."
        ),
        yaml_definition="",
    )


def evaluate_rule_agent(state: WAFState) -> Dict[str, Any]:
    """Node: LLM agent node (ChatOpenAI / gpt-4o) verifying rule condition logic and producing structured JSON."""
    retrieved_rules = state.get("retrieved_rules", [])
    log_context = state.get("log_context", {})

    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            structured_llm = llm.with_structured_output(WAFRuleEvaluation)

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    (
                        "You are an expert Fastly Next-Gen WAF security agent.\n"
                        "Your job is to inspect the retrieved WAF rule condition logic and verify if the incoming "
                        "HTTP attack context matches the rule's conditions.\n\n"
                        "Rules to follow:\n"
                        "1. If the request context contains OWASP attack signals (e.g., 'XSS', 'SQLI', 'Traversal', "
                        "'CMDEXE', 'Attack Tooling'), it matches RULE-002 ('OWASP Core Attack Prevention').\n"
                        # NEW: RULE-001 IP threat routing, with explicit precedence so
                        # attack payloads still report their attack-specific rule.
                        "1a. IP threat signals ('SITE-FLAGGED-IP', 'SIGSCI-IP', 'MALICIOUS-IP', "
                        "'BLOCKLISTED-IP') indicate a flagged malicious client address and match "
                        "RULE-001 ('Malicious and SigSci IP Block').\n"
                        "1b. PRECEDENCE: attack-specific rules outrank RULE-001. If the request has BOTH an IP "
                        "threat signal AND an attack signal (OWASP payload, sensitive path, or bot conditions), "
                        "return the attack-specific rule (RULE-002/003/004), NOT RULE-001. Return RULE-001 only "
                        "when an IP threat signal is the sole threat present.\n"
                        "2. Output the exact matched_rule_id ('RULE-002' for OWASP attacks, 'RULE-001' for "
                        "IP-reputation-only threats).\n"
                        "3. Output action: 'BLOCK'.\n"
                        "4. Output detected_signals: list of detected attack signals that triggered the rule (e.g. ['XSS', 'SQLI']).\n"
                        "5. Output explanation: concise description explaining the match, e.g. "
                        "\"The request contained OWASP attack signals ('XSS') matching RULE-002 condition criteria.\"\n"
                        "6. Output yaml_definition: the raw YAML block from the matched rule definition.\n"
                        # NEW: an explicit no-match option. Without this the model
                        # is forced to name some rule, which produced matches that
                        # cited signals the rule did not contain.
                        "7. GROUNDING: every signal you list in detected_signals MUST appear verbatim "
                        "in the request's Signals list AND in the matched rule's signals list. Signal "
                        "names are exact identifiers, not descriptions: 'SUSPECTED-BOT' and "
                        "'SUSPECTED-BAD-BOT' are different detectors and must never be treated as "
                        "equivalent.\n"
                        "8. NO MATCH: if no retrieved rule's conditions are satisfied by the request "
                        "context, set matched_rule_id to 'NO_RULE_MATCH', action to 'NONE', "
                        "detected_signals to [], and explain which signals are uncovered. Do NOT pick "
                        "the closest-looking rule."
                    ),
                ),
                (
                    "human",
                    (
                        "Incoming Request Context:\n"
                        "- Method: {method}\n"
                        "- Domain: {domain}\n"
                        "- Path: {path}\n"
                        "- Signals: {signals}\n\n"
                        "Retrieved WAF Rules:\n"
                        "{rules_text}\n\n"
                        "Evaluate the condition logic and return the structured result."
                    ),
                ),
            ])

            rules_text_parts = []
            for r in retrieved_rules:
                rules_text_parts.append(
                    f"--- Rule: {r['rule_id']} (Action: {r['action']}) ---\n"
                    f"{r['page_content']}\n"
                    f"Raw YAML:\n{r['yaml_definition']}\n"
                )
            rules_text = "\n".join(rules_text_parts)

            chain = prompt | structured_llm
            result: WAFRuleEvaluation = chain.invoke({
                "method": log_context.get("method", "GET"),
                "domain": log_context.get("domain", ""),
                "path": log_context.get("path", ""),
                "signals": json.dumps(log_context.get("signals", [])),
                "rules_text": rules_text,
            })

            # Ensure yaml_definition is populated if LLM omitted it
            if not result.yaml_definition:
                for r in retrieved_rules:
                    if r.get("rule_id") == result.matched_rule_id:
                        result.yaml_definition = r.get("yaml_definition", "")
                        break

            # NEW: tag provenance so `verify_match` can fall back to the
            # deterministic evaluator when an LLM match fails verification.
            return {"final_output": {**result.model_dump(), "evaluator": "llm"}}

        except Exception as e:
            logger.warning("ChatOpenAI execution encountered error (%s); using deterministic evaluation.", e)

    # Deterministic fallback evaluation
    result = _deterministic_evaluate_rule(retrieved_rules, log_context)
    return {"final_output": {**result.model_dump(), "evaluator": "deterministic"}}


# ==============================================================================
# Step 4: Independent Verification
# ==============================================================================

def _rule_condition_types(raw_yaml: str) -> List[str]:
    """Condition types declared by a rule's YAML, e.g. ['signal_exists', 'path_contains']."""
    if not raw_yaml:
        return []
    try:
        parsed = yaml.safe_load(raw_yaml)
    except Exception:
        return []
    if not isinstance(parsed, dict):
        return []
    conditions = parsed.get("conditions") or {}
    rules_list = conditions.get("rules") if isinstance(conditions, dict) else None
    if not isinstance(rules_list, list):
        return []
    return [r.get("type", "") for r in rules_list if isinstance(r, dict)]


def _verify_evaluation(
    evaluation: Dict[str, Any],
    retrieved_rules: List[Dict[str, Any]],
    log_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Independently re-check a reported match. Pure assertions, no LLM calls.

    Catches the failure mode where the evaluator reports a fluent, confident
    match citing a signal that the rule does not contain or that the request
    never carried.

    Returns the ``verified`` / ``verification_level`` / ``verification_notes``
    fields to merge onto the evaluation.
    """
    rule_id = evaluation.get("matched_rule_id", "")
    detected = evaluation.get("detected_signals") or []
    notes: List[str] = []

    if rule_id == NO_RULE_MATCH:
        return {
            "verified": True,
            "verification_level": VERIFY_NO_MATCH,
            "verification_notes": ["No rule claimed; nothing to verify."],
        }

    # Check 1: the reported rule must be one that retrieval actually returned.
    matched_rule = next(
        (r for r in retrieved_rules if r.get("rule_id") == rule_id), None
    )
    if matched_rule is None:
        available = ", ".join(r.get("rule_id", "?") for r in retrieved_rules) or "none"
        return {
            "verified": False,
            "verification_level": VERIFY_FAILED,
            "verification_notes": [
                f"Rule {rule_id!r} was not among the retrieved rules ({available}); "
                "it may be hallucinated."
            ],
        }

    rule_signals = {normalize_signal_id(s) for s in (matched_rule.get("signals") or [])}
    log_signals = {normalize_signal_id(s) for s in (log_context.get("signals") or [])}

    # Check 2: every cited signal must belong to the matched rule.
    for sig in detected:
        if normalize_signal_id(sig) not in rule_signals:
            notes.append(
                f"Signal {sig!r} is not declared by {rule_id} "
                f"(rule declares: {sorted(rule_signals) or 'none'})."
            )

    # Check 3: every cited signal must actually be present on the request.
    for sig in detected:
        if normalize_signal_id(sig) not in log_signals:
            notes.append(f"Signal {sig!r} is not present in the request's signals.")

    # Check 4: a signal-based rule reported as matched must cite the signal that
    # triggered it. An empty list is the signature of the old blind fallback.
    if rule_signals and not detected:
        notes.append(
            f"{rule_id} matches on signals but no triggering signal was reported."
        )

    # Check 5 (informational): an IP reputation signal should be IP-scoped.
    details = {d.get("id"): d for d in (log_context.get("signal_details") or [])}
    for sig in detected:
        detail = details.get(sig)
        if detail and is_ip_threat_signal(sig):
            location = detail.get("location", "")
            if location and location.upper() != "IP":
                notes.append(
                    f"IP reputation signal {sig!r} has unexpected location "
                    f"{location!r} (expected 'IP')."
                )

    # Check 6 (NEW, Phase 3): independently re-evaluate the rule's own declared
    # conditions against the request. Previously only signal membership was
    # checked, which passed a RULE-004 match on a request that failed RULE-004's
    # domain and path conditions.
    outcome = evaluate_rule_conditions(matched_rule, log_context)
    if outcome.unevaluable:
        notes.append(
            f"{rule_id} declares conditions this evaluator cannot decide: "
            f"{', '.join(outcome.unevaluable)}."
        )
    elif not outcome.matched:
        detail = "; ".join(outcome.reasons) if outcome.reasons else "no condition was satisfied"
        notes.append(
            f"{rule_id}'s declared conditions are not satisfied by this request "
            f"({detail})."
        )

    if notes:
        return {
            "verified": False,
            "verification_level": VERIFY_FAILED,
            "verification_notes": notes,
        }

    # A rate-limit rule's conditions hold, but its threshold spans many requests
    # and cannot be confirmed from one log row.
    if outcome.stateful_gate:
        return {
            "verified": True,
            "verification_level": VERIFY_SIGNALS_ONLY,
            "verification_notes": [
                f"Conditions verified. {rule_id} is a {outcome.stateful_gate} rule; "
                "its threshold spans multiple requests and is not confirmable from "
                "a single log row."
            ],
        }

    return {
        "verified": True,
        "verification_level": VERIFY_FULL,
        "verification_notes": [],
    }


def rule_signal_vocabulary(rules_dir: str = "rules") -> Dict[str, List[str]]:
    """Map each rule ID to the normalized signals it declares, read from the files.

    Derived from the rule Markdown rather than from a Python constant, so the
    vocabulary cannot drift from the rules the pipeline actually indexes.
    """
    vocabulary: Dict[str, List[str]] = {}
    for md_file in sorted(Path(rules_dir).glob("RULE-*.md")):
        for rule_id, raw_yaml in extract_yaml_blocks_by_rule(
            md_file.read_text(encoding="utf-8")
        ).items():
            try:
                parsed = yaml.safe_load(raw_yaml) or {}
            except Exception:
                continue
            resolved_id = parsed.get("rule_id", rule_id)
            conditions = parsed.get("conditions") or {}
            condition_list = conditions.get("rules") if isinstance(conditions, dict) else []
            declared = [
                normalize_signal_id(s)
                for condition in (condition_list or [])
                if isinstance(condition, dict)
                for s in _as_list(condition.get("signals"))
            ]
            vocabulary[resolved_id] = sorted(set(declared))
    return vocabulary


def coverage_report(csv_path: str, rules_dir: str = "rules") -> Dict[str, Any]:
    """Diagnose which log signals no indexed rule covers.

    Answers "which traffic can this rule set not explain?" — the question that
    matters when the pipeline is used to decide what rule to write next. A signal
    is *covered* when some rule declares it; a row is *covered* when at least one
    of its signals is.
    """
    vocabulary = rule_signal_vocabulary(rules_dir)
    owned = {s for declared in vocabulary.values() for s in declared}

    details_by_row = extract_signal_details_from_csv(csv_path)

    signal_counts: Dict[str, int] = {}
    uncovered_counts: Dict[str, int] = {}
    uncategorized: Dict[str, int] = {}
    rows_uncovered = 0

    for row_details in details_by_row:
        row_signals = {normalize_signal_id(d["id"]) for d in row_details}
        for signal in row_signals:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
            if signal not in owned:
                uncovered_counts[signal] = uncovered_counts.get(signal, 0) + 1
            if not signal_category(signal, rules_dir):
                uncategorized[signal] = uncategorized.get(signal, 0) + 1
        if not (row_signals & owned):
            rows_uncovered += 1

    return {
        "total_rows": len(details_by_row),
        "rows_with_no_covered_signal": rows_uncovered,
        "signal_counts": dict(sorted(signal_counts.items(), key=lambda kv: -kv[1])),
        # Signals seen in the logs that no rule declares: candidates for a new rule.
        "uncovered_signals": dict(sorted(uncovered_counts.items(), key=lambda kv: -kv[1])),
        # Signals missing from signal_categories.yaml: taxonomy gaps.
        "uncategorized_signals": dict(sorted(uncategorized.items(), key=lambda kv: -kv[1])),
        "rule_signal_vocabulary": vocabulary,
    }


def verify_match(state: WAFState) -> Dict[str, Any]:
    """Node: verify the evaluator's claim; substitute deterministic result on failure.

    When an LLM match fails verification the deterministic evaluator is re-run
    and used instead if it verifies, so a hallucinated match degrades to a
    grounded one rather than reaching the user.
    """
    evaluation = dict(state.get("final_output") or {})
    retrieved_rules = state.get("retrieved_rules", [])
    log_context = state.get("log_context", {})

    verification = _verify_evaluation(evaluation, retrieved_rules, log_context)

    if not verification["verified"] and evaluation.get("evaluator") == "llm":
        logger.warning(
            "LLM match %r failed verification (%s); re-evaluating deterministically.",
            evaluation.get("matched_rule_id"),
            "; ".join(verification["verification_notes"]),
        )
        rejected_id = evaluation.get("matched_rule_id")
        rejected_notes = verification["verification_notes"]

        fallback = _deterministic_evaluate_rule(retrieved_rules, log_context).model_dump()
        fallback["evaluator"] = "deterministic (LLM match rejected)"
        fallback_verification = _verify_evaluation(
            fallback, retrieved_rules, log_context
        )
        fallback_verification["verification_notes"] = [
            f"Rejected LLM match {rejected_id!r}: {note}" for note in rejected_notes
        ] + fallback_verification["verification_notes"]
        return {"final_output": {**fallback, **fallback_verification}}

    return {"final_output": {**evaluation, **verification}}


def create_waf_pipeline(
    vectorstore: Optional[QdrantClient] = None,
    checkpointer: Optional[Any] = None,
) -> Any:
    """Build and compile the LangGraph workflow.

    Workflow:
    START -> extract_log_context -> retrieve_waf_rules -> evaluate_rule_agent
          -> verify_match -> END

    NEW (Phase 4): compiles with a checkpointer so each analysis is persisted
    under a ``thread_id``. This is what lets a later conversational layer answer
    follow-up questions about an earlier analysis, and it gives every
    recommendation a replayable audit trail. Pass ``checkpointer=False`` to
    compile without one.
    """
    workflow = StateGraph(WAFState)

    workflow.add_node("extract_log_context", extract_log_context)
    # Bind the client to the node so it stays out of the serialized state.
    workflow.add_node(
        "retrieve_waf_rules",
        partial(retrieve_waf_rules, vectorstore=vectorstore),
    )
    workflow.add_node("evaluate_rule_agent", evaluate_rule_agent)
    workflow.add_node("verify_match", verify_match)

    workflow.add_edge(START, "extract_log_context")
    workflow.add_edge("extract_log_context", "retrieve_waf_rules")
    workflow.add_edge("retrieve_waf_rules", "evaluate_rule_agent")
    workflow.add_edge("evaluate_rule_agent", "verify_match")
    workflow.add_edge("verify_match", END)

    if checkpointer is False:
        return workflow.compile()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def run_waf_analysis(
    csv_path: str = "requests_sqli_xss_cmdexe.csv",
    row_index: int = 0,
    rules_dir: str = "rules",
    vectorstore: Optional[QdrantClient] = None,
    log_context: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
    checkpointer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the pipeline and return the **complete** final state.

    NEW (Phase 4): `process_waf_log` returns only ``final_output``, discarding
    ``log_context`` and ``retrieved_rules`` — precisely the context a follow-up
    question needs ("why this rule and not that one?", "what else did retrieval
    consider?"). This returns everything, plus the ``thread_id`` the run was
    checkpointed under.

    Routing:
    - A rule matches when its declared YAML conditions are satisfied; among
      several matches, one matching only on IP reputation signals yields to any
      rule matching on other grounds.
    - When no rule's conditions are satisfied the result is ``NO_RULE_MATCH``
      rather than the highest-scoring retrieved rule.

    Pass `log_context` to evaluate a synthetic request context directly instead
    of reading a CSV row.
    """
    if vectorstore is None:
        vectorstore = build_waf_rule_vectorstore(rules_dir=rules_dir)

    app = create_waf_pipeline(vectorstore=vectorstore, checkpointer=checkpointer)

    initial_state: WAFState = {
        "csv_path": csv_path,
        "row_index": row_index,
    }

    # An explicit log_context short-circuits CSV reading in extract_log_context,
    # so synthetic request contexts can be evaluated without a CSV.
    if log_context is not None:
        enriched = dict(log_context)
        enriched.setdefault("signals", [])
        enriched.setdefault("signal_details", [])
        enriched["ip_threat_signals"] = [
            s for s in enriched["signals"] if is_ip_threat_signal(s)
        ]
        initial_state["log_context"] = enriched

    # A thread_id is required whenever the graph is compiled with a checkpointer.
    resolved_thread_id = thread_id or str(uuid4())
    config = (
        None
        if checkpointer is False
        else {"configurable": {"thread_id": resolved_thread_id}}
    )

    final_state = app.invoke(initial_state, config=config)

    return {
        "thread_id": resolved_thread_id,
        "final_output": final_state.get("final_output", {}),
        "log_context": final_state.get("log_context", {}),
        "retrieved_rules": final_state.get("retrieved_rules", []),
    }


def process_waf_log(
    csv_path: str = "requests_sqli_xss_cmdexe.csv",
    row_index: int = 0,
    rules_dir: str = "rules",
    vectorstore: Optional[QdrantClient] = None,
    log_context: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the pipeline for one log row and return just the match result.

    Thin wrapper over `run_waf_analysis` preserving the original return shape
    (the `final_output` dict alone) for existing callers. New code that needs
    the retrieved candidates or the request context should call
    `run_waf_analysis` directly.
    """
    return run_waf_analysis(
        csv_path=csv_path,
        row_index=row_index,
        rules_dir=rules_dir,
        vectorstore=vectorstore,
        log_context=log_context,
        thread_id=thread_id,
    )["final_output"]
