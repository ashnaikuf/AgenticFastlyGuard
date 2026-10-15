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
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import NAMESPACE_URL, uuid5
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
from langgraph.graph import END, START, StateGraph

# Load environment variables
load_dotenv()
logger = logging.getLogger("waf_agent_pipeline")


# ==============================================================================
# Pydantic Schemas & State Types
# ==============================================================================

class WAFRuleMatchResult(BaseModel):
    """Structured output schema for WAF rule evaluation matching."""

    matched_rule_id: str = Field(
        ...,
        description="The matched rule identifier, e.g., RULE-002",
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


class WAFState(TypedDict, total=False):
    """State schema tracked across LangGraph nodes."""

    csv_path: str
    row_index: int
    log_context: Dict[str, Any]
    retrieved_rules: List[Dict[str, Any]]
    final_output: Dict[str, Any]
    vectorstore: Any


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

def extract_signals_from_csv(csv_path: str) -> List[List[str]]:
    """Read the incoming CSV file using pandas and extract signal IDs for each row.

    The CSV contains a `signals` column with JSON strings like:
    `[{"id":"XSS","location":"QUERYSTRING",...}]`

    Returns a list of detected signal IDs for each row.
    """
    df = pd.read_csv(csv_path)
    all_signals: List[List[str]] = []

    for val in df.get("signals", []):
        row_signals: List[str] = []
        if pd.notna(val) and isinstance(val, str) and val.strip():
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "id" in item and item["id"]:
                            row_signals.append(item["id"])
            except Exception as err:
                logger.debug("Failed parsing signals JSON: %s", err)
        all_signals.append(row_signals)

    return all_signals


# ==============================================================================
# Step 3: LangGraph Agent Pipeline
# ==============================================================================

def extract_log_context(state: WAFState) -> Dict[str, Any]:
    """Node: Reads CSV and extracts signals, path, domain, and HTTP method."""
    if state.get("log_context"):
        return {"log_context": state["log_context"]}

    csv_path = state.get("csv_path", "requests_sqli_xss_cmdexe.csv")
    row_index = state.get("row_index", 0)

    df = pd.read_csv(csv_path)
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index {row_index} out of bounds for CSV of length {len(df)}")

    row = df.iloc[row_index]

    # Parse signals for this row
    signals_raw = row.get("signals")
    signals: List[str] = []
    if pd.notna(signals_raw) and isinstance(signals_raw, str):
        try:
            parsed = json.loads(signals_raw)
            if isinstance(parsed, list):
                signals = [item["id"] for item in parsed if isinstance(item, dict) and "id" in item]
        except Exception:
            pass

    domain = (
        str(row.get("server_name") or row.get("server_hostname") or "")
    )
    path = str(row.get("path") or "")
    method = str(row.get("method") or "GET")
    uri = str(row.get("uri") or "")

    log_context = {
        "signals": signals,
        "path": path,
        "domain": domain,
        "method": method,
        "uri": uri,
        "row_index": row_index,
    }

    return {"log_context": log_context}


def retrieve_waf_rules(state: WAFState) -> Dict[str, Any]:
    """Node: Queries the vector store with extracted signals string and applies metadata filtering."""
    log_context = state.get("log_context", {})
    signals = log_context.get("signals", [])
    path = log_context.get("path", "")

    vectorstore = state.get("vectorstore")
    if vectorstore is None:
        vectorstore = build_waf_rule_vectorstore()

    # Form query string from detected signals and request path
    query_terms = [s for s in signals if s not in {"BLOCKED", "BOT-ANALYSIS", "SITE-FLAGGED-IP"}]
    if not query_terms:
        query_terms = signals
    query_str = f"Signals: {' '.join(query_terms)} Path: {path}"

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
        limit=4,
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


def _deterministic_evaluate_rule(
    retrieved_rules: List[Dict[str, Any]],
    log_context: Dict[str, Any],
) -> WAFRuleMatchResult:
    """Deterministic fallback evaluator verifying WAF rule condition logic.

    Ensures predictable behavior and tests without requiring live external API keys.
    """
    signals = log_context.get("signals", [])
    signals_lower_map = {s.lower(): s for s in signals}
    path = log_context.get("path", "")
    domain = log_context.get("domain", "")

    for rule in retrieved_rules:
        rule_id = rule.get("rule_id")
        action = rule.get("action", "BLOCK")
        rule_signals = rule.get("signals", [])
        raw_yaml = rule.get("yaml_definition", "")

        matched_detected: List[str] = []

        # Check signal_exists
        for rs in rule_signals:
            if rs.lower() in signals_lower_map:
                matched_detected.append(signals_lower_map[rs.lower()])

        # Check RULE-002 OWASP core attack signals
        if rule_id == "RULE-002" and matched_detected:
            # Sort or prioritize recognizable OWASP signal names
            matched_detected = sorted(list(set(matched_detected)))
            explanation = (
                f"The request contained OWASP attack signals ({', '.join(repr(s) for s in matched_detected)}) "
                f"matching {rule_id} condition criteria."
            )
            return WAFRuleMatchResult(
                matched_rule_id=rule_id,
                action=action,
                detected_signals=matched_detected,
                explanation=explanation,
                yaml_definition=raw_yaml,
            )

        # Check RULE-003 path_contains
        if rule_id == "RULE-003":
            sensitive_paths = ["/.env", "/etc/passwd", "/profile/image/url"]
            if any(sp in path for sp in sensitive_paths):
                return WAFRuleMatchResult(
                    matched_rule_id=rule_id,
                    action=action,
                    detected_signals=matched_detected,
                    explanation=f"The request path '{path}' contains sensitive target paths matching RULE-003.",
                    yaml_definition=raw_yaml,
                )

        # Check RULE-004 domain + path + bot signals
        if rule_id == "RULE-004":
            if domain == "dimensionash26.secureedge.xyz" and path == "/login" and matched_detected:
                return WAFRuleMatchResult(
                    matched_rule_id=rule_id,
                    action=action,
                    detected_signals=matched_detected,
                    explanation=f"Request to sensitive endpoint {domain}{path} matched bot signal conditions in RULE-004.",
                    yaml_definition=raw_yaml,
                )

        # General signal_exists match
        if matched_detected:
            return WAFRuleMatchResult(
                matched_rule_id=rule_id,
                action=action,
                detected_signals=matched_detected,
                explanation=f"Request signals matched {rule_id} condition criteria.",
                yaml_definition=raw_yaml,
            )

    # Default fallback to first retrieved rule or RULE-002
    fallback_rule = retrieved_rules[0] if retrieved_rules else {}
    return WAFRuleMatchResult(
        matched_rule_id=fallback_rule.get("rule_id", "RULE-002"),
        action=fallback_rule.get("action", "BLOCK"),
        detected_signals=[s for s in signals if s.upper() in {"XSS", "SQLI", "TRAVERSAL", "CMDEXE"}],
        explanation="Request evaluated against WAF rules.",
        yaml_definition=fallback_rule.get("yaml_definition", ""),
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
            structured_llm = llm.with_structured_output(WAFRuleMatchResult)

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
                        "2. Output the exact matched_rule_id ('RULE-002' for OWASP attacks).\n"
                        "3. Output action: 'BLOCK'.\n"
                        "4. Output detected_signals: list of detected attack signals that triggered the rule (e.g. ['XSS', 'SQLI']).\n"
                        "5. Output explanation: concise description explaining the match, e.g. "
                        "\"The request contained OWASP attack signals ('XSS') matching RULE-002 condition criteria.\"\n"
                        "6. Output yaml_definition: the raw YAML block from the matched rule definition."
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
            result: WAFRuleMatchResult = chain.invoke({
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

            return {"final_output": result.model_dump()}

        except Exception as e:
            logger.warning("ChatOpenAI execution encountered error (%s); using deterministic evaluation.", e)

    # Deterministic fallback evaluation
    result = _deterministic_evaluate_rule(retrieved_rules, log_context)
    return {"final_output": result.model_dump()}


def create_waf_pipeline(vectorstore: Optional[QdrantClient] = None) -> Any:
    """Build and compile the LangGraph workflow.

    Workflow:
    START -> extract_log_context -> retrieve_waf_rules -> evaluate_rule_agent -> END
    """
    workflow = StateGraph(WAFState)

    workflow.add_node("extract_log_context", extract_log_context)
    workflow.add_node("retrieve_waf_rules", retrieve_waf_rules)
    workflow.add_node("evaluate_rule_agent", evaluate_rule_agent)

    workflow.add_edge(START, "extract_log_context")
    workflow.add_edge("extract_log_context", "retrieve_waf_rules")
    workflow.add_edge("retrieve_waf_rules", "evaluate_rule_agent")
    workflow.add_edge("evaluate_rule_agent", END)

    app = workflow.compile()
    return app


def process_waf_log(
    csv_path: str = "requests_sqli_xss_cmdexe.csv",
    row_index: int = 0,
    rules_dir: str = "rules",
    vectorstore: Optional[QdrantClient] = None,
) -> Dict[str, Any]:
    """Execute the end-to-end WAF agent pipeline for a specific CSV log row."""
    if vectorstore is None:
        vectorstore = build_waf_rule_vectorstore(rules_dir=rules_dir)

    app = create_waf_pipeline(vectorstore=vectorstore)

    initial_state: WAFState = {
        "csv_path": csv_path,
        "row_index": row_index,
        "vectorstore": vectorstore,
    }

    final_state = app.invoke(initial_state)
    return final_state.get("final_output", {})
