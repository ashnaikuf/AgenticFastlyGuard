"""Unit and integration tests for Fastly Next-Gen WAF Agent Pipeline."""

import json
import os
from pathlib import Path
import pandas as pd
import pytest
from langgraph.checkpoint.memory import MemorySaver
from qdrant_client import models

from waf_agent_pipeline import (
    NO_RULE_MATCH,
    VERIFY_FAILED,
    VERIFY_FULL,
    VERIFY_NO_MATCH,
    VERIFY_SIGNALS_ONLY,
    WAFRuleMatchResult,
    _deterministic_evaluate_rule,
    _verify_evaluation,
    build_log_context,
    build_waf_rule_vectorstore,
    coverage_report,
    create_waf_pipeline,
    evaluate_rule_conditions,
    extract_signal_details_from_csv,
    extract_signals_from_csv,
    get_embeddings,
    get_qdrant_vector_size,
    is_ip_threat_signal,
    load_signal_categories,
    normalize_signal_id,
    process_waf_log,
    rule_signal_vocabulary,
    run_waf_analysis,
    signal_category,
)

# A RULE-001 shaped stub whose conditions are signal_exists only.
RULE_001_STUB = {
    "rule_id": "RULE-001",
    "action": "BLOCK",
    "signals": ["Malicious IP", "SigSci IP", "SANS"],
    "yaml_definition": (
        "rule_id: RULE-001\n"
        "action: BLOCK\n"
        "conditions:\n"
        "  operator: ALL\n"
        "  rules:\n"
        "    - type: signal_exists\n"
        "      operator: ANY\n"
        "      signals:\n"
        '        - "SANS"\n'
    ),
    "page_content": "",
}

# A RULE-004 shaped stub: rule-level `domain` scope plus path and signal
# conditions, mirroring rules/RULE-004-client-side-detection.md.
RULE_004_STUB = {
    "rule_id": "RULE-004",
    "action": "BLOCK",
    "signals": ["SUSPECTED-BOT"],
    "yaml_definition": (
        "rule_id: RULE-004\n"
        'domain: "dimensionash26.secureedge.xyz"\n'
        "action: BLOCK\n"
        "conditions:\n"
        "  operator: ALL\n"
        "  rules:\n"
        "    - type: path_equals\n"
        '      value: "/login"\n'
        "    - type: signal_exists\n"
        "      operator: ANY\n"
        "      signals:\n"
        '        - "Suspected Bot"\n'
    ),
    "page_content": "",
}

# A rate-limit rule: its conditions can hold for one request, but the threshold
# spans many, so a single row can never confirm the action fired.
RULE_005_STUB = {
    "rule_id": "RULE-005",
    "action": "BLOCK",
    "signals": ["LOGINDISCOVERY"],
    "yaml_definition": (
        "rule_id: RULE-005\n"
        "type: rate_limit\n"
        "action: BLOCK\n"
        "conditions:\n"
        "  operator: ALL\n"
        "  rules:\n"
        "    - type: path_contains\n"
        "      values:\n"
        '        - "/login"\n'
        "    - type: method_equals\n"
        '      value: "POST"\n'
        "    - type: signal_exists\n"
        "      operator: ANY\n"
        "      signals:\n"
        '        - "LOGINDISCOVERY"\n'
        "rate_limit:\n"
        "  threshold: 10\n"
        "  interval: 1m\n"
    ),
    "page_content": "",
}

# A rule declaring a condition type the evaluator does not implement.
RULE_UNKNOWN_CONDITION_STUB = {
    "rule_id": "RULE-900",
    "action": "BLOCK",
    "signals": ["SANS"],
    "yaml_definition": (
        "rule_id: RULE-900\n"
        "conditions:\n"
        "  operator: ALL\n"
        "  rules:\n"
        "    - type: signal_exists\n"
        '      signals: ["SANS"]\n'
        "    - type: tls_fingerprint_equals\n"
        '      value: "ja4-abc"\n'
    ),
    "page_content": "",
}

ROW_0_CONTEXT = {
    "signals": ["BOT-ANALYSIS", "SANS", "SUSPECTED-BOT"],
    "signal_details": [
        {"id": "SANS", "location": "IP", "value": "205.210.31.135", "detector": "SansRule"},
    ],
    "path": "/",
    "domain": "parcfermestore.secureedge.xyz",
    "method": "GET",
}


@pytest.fixture(scope="session")
def vectorstore():
    """Build and cache the vector store once for testing."""
    return build_waf_rule_vectorstore(
        rules_dir="rules",
        embedding_provider="auto",
    )


def test_extract_signals_from_csv():
    """Verify CSV signal extraction parses JSON lists accurately."""
    csv_file = "requests_sqli_xss_cmdexe.csv"
    assert Path(csv_file).exists(), f"{csv_file} does not exist"

    signals_by_row = extract_signals_from_csv(csv_file)
    assert len(signals_by_row) == 1000

    # Row 0 has XSS
    assert "XSS" in signals_by_row[0]
    assert "BLOCKED" in signals_by_row[0]

    # Row 2 has SQLI
    assert "SQLI" in signals_by_row[2]

    # Row 3 has TRAVERSAL
    assert "TRAVERSAL" in signals_by_row[3]

    # Row 612 has CMDEXE
    assert "CMDEXE" in signals_by_row[612]


def test_vectorstore_indexing_and_metadata(vectorstore):
    """Verify MarkdownHeaderTextSplitter chunking and metadata extraction."""
    results = vectorstore.query_points(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME", "fastly_wafrules"),
        query=get_embeddings(
            dimensions=get_qdrant_vector_size(
                vectorstore,
                os.getenv("QDRANT_COLLECTION_NAME", "fastly_wafrules"),
            )
        ).embed_query(
            "Signals: XSS SQLI Traversal CMDEXE Attack Tooling"
        ),
        using="dense_vector",
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.action",
                    match=models.MatchValue(value="BLOCK"),
                )
            ]
        ),
        limit=2,
        with_payload=True,
    ).points
    assert len(results) > 0

    top_rule = results[0].payload["metadata"]
    assert top_rule["rule_id"] == "RULE-002"
    assert top_rule["action"] == "BLOCK"
    expected_signals = ["XSS", "SQLI", "Traversal", "CMDEXE", "Attack Tooling"]
    for sig in expected_signals:
        assert sig in top_rule["signals"]
    assert "rule_id: RULE-002" in top_rule["yaml_definition"]
    assert "action: BLOCK" in top_rule["yaml_definition"]


def test_pipeline_row_0_xss(vectorstore):
    """Verify full LangGraph pipeline on row 0 matching RULE-002 for XSS."""
    result = process_waf_log(
        csv_path="requests_sqli_xss_cmdexe.csv",
        row_index=0,
        rules_dir="rules",
        vectorstore=vectorstore,
    )

    # Validate Pydantic schema
    validated = WAFRuleMatchResult(**result)

    assert validated.matched_rule_id == "RULE-002"
    assert validated.action == "BLOCK"
    assert "XSS" in validated.detected_signals
    assert "RULE-002" in validated.explanation
    assert "rule_id: RULE-002" in validated.yaml_definition
    assert "signal_exists" in validated.yaml_definition


def test_pipeline_row_2_sqli(vectorstore):
    """Verify full LangGraph pipeline on row 2 matching RULE-002 for SQLI."""
    result = process_waf_log(
        csv_path="requests_sqli_xss_cmdexe.csv",
        row_index=2,
        rules_dir="rules",
        vectorstore=vectorstore,
    )

    validated = WAFRuleMatchResult(**result)
    assert validated.matched_rule_id == "RULE-002"
    assert validated.action == "BLOCK"
    assert "SQLI" in validated.detected_signals
    assert "RULE-002" in validated.explanation


def test_pipeline_row_612_cmdexe(vectorstore):
    """Verify full LangGraph pipeline on row 612 matching RULE-002 for CMDEXE."""
    result = process_waf_log(
        csv_path="requests_sqli_xss_cmdexe.csv",
        row_index=612,
        rules_dir="rules",
        vectorstore=vectorstore,
    )

    validated = WAFRuleMatchResult(**result)
    assert validated.matched_rule_id == "RULE-002"
    assert validated.action == "BLOCK"
    assert "CMDEXE" in validated.detected_signals
    assert "RULE-002" in validated.explanation


def test_pipeline_multiple_owasp_signals(vectorstore):
    """Verify pipeline with synthetic log containing multiple attack signals (XSS and SQLI)."""
    # The Qdrant client is bound to the graph, not passed through state: state is
    # serialized by the checkpointer and a live client is not serializable.
    app = create_waf_pipeline(vectorstore=vectorstore)

    multi_signal_state = {
        "log_context": {
            "signals": ["BLOCKED", "BOT-ANALYSIS", "XSS", "SQLI"],
            "signal_details": [],
            "path": "/rest/products/search",
            "domain": "owaspjuiceshop.secureedge.xyz",
            "method": "POST",
            "user_agent": "",
        },
    }

    final_state = app.invoke(
        multi_signal_state,
        config={"configurable": {"thread_id": "test-multi-signal"}},
    )
    result = final_state.get("final_output", {})

    validated = WAFRuleMatchResult(**result)
    assert validated.matched_rule_id == "RULE-002"
    assert validated.action == "BLOCK"
    assert "XSS" in validated.detected_signals
    assert "SQLI" in validated.detected_signals
    assert "rule_id: RULE-002" in validated.yaml_definition


# ==============================================================================
# Phase 2: signal detail preservation
# ==============================================================================

def test_signal_details_preserve_detector_and_value():
    """SANS must retain location='IP' and the flagged client address."""
    details = extract_signal_details_from_csv("requests_maliciousip.csv")
    assert len(details) == 1000

    row_0 = {d["id"]: d for d in details[0]}
    assert "SANS" in row_0
    assert row_0["SANS"]["location"] == "IP"
    assert row_0["SANS"]["value"] == "205.210.31.135"
    assert row_0["SANS"]["detector"] == "SansRule"


def test_signal_ids_match_legacy_extraction():
    """The detail-preserving path must yield identical IDs to the flat extractor."""
    flat = extract_signals_from_csv("requests_maliciousip.csv")
    details = extract_signal_details_from_csv("requests_maliciousip.csv")
    assert flat == [[d["id"] for d in row] for row in details]


def test_suspected_bot_and_bad_bot_stay_distinct():
    """These are separate Fastly detectors and must never normalize equal."""
    assert normalize_signal_id("SUSPECTED-BOT") != normalize_signal_id("Suspected Bad Bot")


# ==============================================================================
# Phase 2: verification
# ==============================================================================

def test_verify_accepts_grounded_sans_match():
    """Row 0's genuine SANS match against RULE-001 fully verifies."""
    result = _verify_evaluation(
        {"matched_rule_id": "RULE-001", "detected_signals": ["SANS"]},
        [RULE_001_STUB],
        ROW_0_CONTEXT,
    )
    assert result["verified"] is True
    assert result["verification_level"] == VERIFY_FULL
    assert result["verification_notes"] == []


def test_verify_rejects_signal_not_declared_by_rule():
    """The original bug: SUSPECTED-BOT cited against RULE-001, which lacks it."""
    result = _verify_evaluation(
        {"matched_rule_id": "RULE-001", "detected_signals": ["SUSPECTED-BOT"]},
        [RULE_001_STUB],
        ROW_0_CONTEXT,
    )
    assert result["verified"] is False
    assert result["verification_level"] == VERIFY_FAILED
    assert any("not declared by RULE-001" in n for n in result["verification_notes"])


def test_verify_rejects_signal_absent_from_request():
    """A rule-declared signal the request never carried must not verify."""
    result = _verify_evaluation(
        {"matched_rule_id": "RULE-001", "detected_signals": ["SigSci IP"]},
        [RULE_001_STUB],
        {"signals": ["SANS"], "signal_details": [], "path": "/", "domain": "d"},
    )
    assert result["verified"] is False
    assert any("not present in the request" in n for n in result["verification_notes"])


def test_verify_rejects_hallucinated_rule_id():
    """A rule that retrieval never returned must not verify."""
    result = _verify_evaluation(
        {"matched_rule_id": "RULE-099", "detected_signals": ["SANS"]},
        [RULE_001_STUB],
        ROW_0_CONTEXT,
    )
    assert result["verified"] is False
    assert any("hallucinated" in n for n in result["verification_notes"])


def test_verify_rejects_empty_signals_on_signal_rule():
    """The old blind-fallback signature: a signal rule matched citing nothing."""
    result = _verify_evaluation(
        {"matched_rule_id": "RULE-001", "detected_signals": []},
        [RULE_001_STUB],
        ROW_0_CONTEXT,
    )
    assert result["verified"] is False
    assert any("no triggering signal" in n for n in result["verification_notes"])


def test_verify_rejects_match_failing_rule_conditions():
    """Phase 3: the RULE-004 false match that signal-only verification passed.

    Row 0 carries SUSPECTED-BOT, which RULE-004 declares, but it is GET / on a
    different host than RULE-004's domain scope. Signal membership alone could
    not catch this.
    """
    result = _verify_evaluation(
        {"matched_rule_id": "RULE-004", "detected_signals": ["SUSPECTED-BOT"]},
        [RULE_004_STUB],
        ROW_0_CONTEXT,
    )
    assert result["verified"] is False
    assert result["verification_level"] == VERIFY_FAILED
    assert any("not satisfied" in n or "scoped to domain" in n
               for n in result["verification_notes"])


def test_verify_rejects_unimplemented_condition_type():
    """A rule the evaluator cannot fully decide must not be reported as verified."""
    result = _verify_evaluation(
        {"matched_rule_id": "RULE-900", "detected_signals": ["SANS"]},
        [RULE_UNKNOWN_CONDITION_STUB],
        ROW_0_CONTEXT,
    )
    assert result["verified"] is False
    assert any("tls_fingerprint_equals" in n for n in result["verification_notes"])


def test_verify_marks_rate_limit_rule_as_not_fully_confirmable():
    """A rate_limit rule's threshold spans requests, so one row cannot confirm it."""
    context = {
        "signals": ["LOGINDISCOVERY"],
        "signal_details": [],
        "path": "/login",
        "method": "POST",
        "domain": "d",
        "user_agent": "",
    }
    result = _verify_evaluation(
        {"matched_rule_id": "RULE-005", "detected_signals": ["LOGINDISCOVERY"]},
        [RULE_005_STUB],
        context,
    )
    assert result["verified"] is True
    assert result["verification_level"] == VERIFY_SIGNALS_ONLY
    assert any("rate_limit" in n for n in result["verification_notes"])


def test_verify_passes_through_no_rule_match():
    result = _verify_evaluation(
        {"matched_rule_id": NO_RULE_MATCH, "detected_signals": []},
        [RULE_001_STUB],
        ROW_0_CONTEXT,
    )
    assert result["verified"] is True
    assert result["verification_level"] == VERIFY_NO_MATCH


# ==============================================================================
# Phase 2: explicit no-match
# ==============================================================================

def test_no_rule_match_replaces_blind_fallback():
    """Uncovered signals yield NO_RULE_MATCH, not the top retrieved rule."""
    evaluation = _deterministic_evaluate_rule(
        [RULE_001_STUB],
        {"signals": ["TOTALLY-UNKNOWN-SIGNAL"], "signal_details": [], "path": "/x", "domain": "d"},
    )
    assert evaluation.matched_rule_id == NO_RULE_MATCH
    assert evaluation.action == "NONE"
    assert evaluation.detected_signals == []
    assert "TOTALLY-UNKNOWN-SIGNAL" in evaluation.explanation
    assert "RULE-001" in evaluation.explanation  # reported as a near candidate


def test_row_0_malicious_ip_resolves_to_rule_001_on_sans():
    """End-to-end deterministic check of the case that started this work."""
    evaluation = _deterministic_evaluate_rule(
        [RULE_004_STUB, RULE_001_STUB], ROW_0_CONTEXT
    )
    assert evaluation.matched_rule_id == "RULE-001"
    assert evaluation.detected_signals == ["SANS"]


def test_rule_004_still_matches_its_own_endpoint():
    """The condition gate must not over-reject valid matches."""
    evaluation = _deterministic_evaluate_rule(
        [RULE_004_STUB],
        {
            "signals": ["SUSPECTED-BOT"],
            "signal_details": [],
            "path": "/login",
            "method": "POST",
            "domain": "dimensionash26.secureedge.xyz",
            "user_agent": "",
        },
    )
    assert evaluation.matched_rule_id == "RULE-004"


# ==============================================================================
# Phase 2: verification wired into the graph
# ==============================================================================

def test_pipeline_emits_verification_fields(vectorstore):
    """The compiled graph must return verification fields on every result."""
    result = process_waf_log(
        csv_path="requests_sqli_xss_cmdexe.csv",
        row_index=0,
        rules_dir="rules",
        vectorstore=vectorstore,
    )
    validated = WAFRuleMatchResult(**result)
    assert validated.matched_rule_id == "RULE-002"
    assert validated.verified is True
    assert validated.verification_level in (VERIFY_FULL, VERIFY_SIGNALS_ONLY)
    assert validated.evaluator in ("llm", "deterministic")


def test_pipeline_verifies_row_0_malicious_ip(vectorstore):
    """Row 0 of the malicious-IP log must verify as a grounded RULE-001 match."""
    result = process_waf_log(
        csv_path="requests_maliciousip.csv",
        row_index=0,
        rules_dir="rules",
        vectorstore=vectorstore,
    )
    validated = WAFRuleMatchResult(**result)
    assert validated.matched_rule_id == "RULE-001"
    assert validated.verified is True, validated.verification_notes
    assert "SANS" in validated.detected_signals


# ==============================================================================
# Phase 3: data-driven signal taxonomy
# ==============================================================================

def test_signal_categories_load_from_config():
    """The taxonomy comes from rules/signal_categories.yaml, not Python constants."""
    categories = load_signal_categories("rules")
    assert categories, "signal_categories.yaml should be present and parseable"
    assert categories["SANS"] == "ip_reputation"
    assert categories["SIGSCI-IP"] == "ip_reputation"
    assert categories["XSS"] == "attack"
    assert categories["BOT-ANALYSIS"] == "generic"


def test_signal_category_prefix_fallback():
    """CVE identifiers categorize by prefix without being enumerated."""
    assert signal_category("CVE-2025-55182") == "attack"
    assert signal_category("CVE-9999-00000") == "attack"


def test_ip_threat_detection_reads_from_config():
    assert is_ip_threat_signal("SANS") is True
    assert is_ip_threat_signal("SigSci IP") is True   # rule-file spelling
    assert is_ip_threat_signal("XSS") is False
    assert is_ip_threat_signal("SUSPECTED-BOT") is False


def test_every_logged_signal_is_categorized():
    """Both bundled datasets must be fully covered by the taxonomy config."""
    for csv_file in ("requests_maliciousip.csv", "requests_sqli_xss_cmdexe.csv"):
        report = coverage_report(csv_file, rules_dir="rules")
        assert report["uncategorized_signals"] == {}, (
            f"{csv_file} has signals missing from signal_categories.yaml: "
            f"{report['uncategorized_signals']}"
        )


# ==============================================================================
# Phase 3: condition evaluator
# ==============================================================================

def test_path_contains_condition():
    rule = {
        "rule_id": "RULE-003",
        "yaml_definition": (
            "rule_id: RULE-003\n"
            "conditions:\n"
            "  operator: ANY\n"
            "  rules:\n"
            "    - type: path_contains\n"
            "      values:\n"
            '        - "/.env"\n'
        ),
    }
    hit = evaluate_rule_conditions(rule, {"path": "/app/.env", "signals": []})
    assert hit.matched is True
    assert "/.env" in hit.reasons[0]

    miss = evaluate_rule_conditions(rule, {"path": "/index.html", "signals": []})
    assert miss.matched is False


def test_user_agent_wildcard_condition_now_evaluated():
    """user_agent_like was declared by RULE-006 but never previously evaluated."""
    rule = {
        "rule_id": "RULE-006",
        "yaml_definition": (
            "rule_id: RULE-006\n"
            "conditions:\n"
            "  operator: ANY\n"
            "  rules:\n"
            "    - type: user_agent_like\n"
            "      match_type: wildcard\n"
            '      value: "*curl*"\n'
        ),
    }
    hit = evaluate_rule_conditions(rule, {"user_agent": "curl/8.1.2", "signals": []})
    assert hit.matched is True

    miss = evaluate_rule_conditions(
        rule, {"user_agent": "Mozilla/5.0", "signals": []}
    )
    assert miss.matched is False


def test_user_agent_equals_condition():
    rule = {
        "rule_id": "RULE-006",
        "yaml_definition": (
            "rule_id: RULE-006\n"
            "conditions:\n"
            "  operator: ANY\n"
            "  rules:\n"
            "    - type: user_agent_equals\n"
            "      values:\n"
            '        - "nmap"\n'
        ),
    }
    assert evaluate_rule_conditions(rule, {"user_agent": "nmap", "signals": []}).matched
    assert evaluate_rule_conditions(rule, {"user_agent": "NMAP", "signals": []}).matched
    assert not evaluate_rule_conditions(rule, {"user_agent": "wget", "signals": []}).matched


def test_operator_all_requires_every_condition():
    """RULE-004 shape: ALL means domain, path, and signal must all hold."""
    full = {
        "signals": ["SUSPECTED-BOT"], "path": "/login", "method": "POST",
        "domain": "dimensionash26.secureedge.xyz", "user_agent": "",
    }
    assert evaluate_rule_conditions(RULE_004_STUB, full).matched is True

    wrong_path = {**full, "path": "/"}
    assert evaluate_rule_conditions(RULE_004_STUB, wrong_path).matched is False

    wrong_domain = {**full, "domain": "elsewhere.example"}
    assert evaluate_rule_conditions(RULE_004_STUB, wrong_domain).matched is False

    no_signal = {**full, "signals": []}
    assert evaluate_rule_conditions(RULE_004_STUB, no_signal).matched is False


def test_rule_level_domain_scope_is_honored():
    """RULE-004 declares `domain:` outside `conditions`; it must still gate."""
    outcome = evaluate_rule_conditions(RULE_004_STUB, ROW_0_CONTEXT)
    assert outcome.matched is False
    assert any("scoped to domain" in r for r in outcome.reasons)


def test_unknown_condition_type_is_reported_not_assumed():
    outcome = evaluate_rule_conditions(RULE_UNKNOWN_CONDITION_STUB, ROW_0_CONTEXT)
    assert outcome.matched is False
    assert "tls_fingerprint_equals" in outcome.unevaluable


def test_rate_limit_rule_flagged_as_stateful():
    context = {
        "signals": ["LOGINDISCOVERY"], "path": "/login",
        "method": "POST", "domain": "d", "user_agent": "",
    }
    outcome = evaluate_rule_conditions(RULE_005_STUB, context)
    assert outcome.matched is True
    assert outcome.stateful_gate == "rate_limit"


def test_precedence_attack_rule_outranks_ip_reputation_only():
    """A blocklisted IP carrying a payload must report the attack rule."""
    rule_002 = {
        "rule_id": "RULE-002", "action": "BLOCK", "signals": ["XSS"],
        "yaml_definition": (
            "rule_id: RULE-002\n"
            "conditions:\n"
            "  operator: ALL\n"
            "  rules:\n"
            "    - type: signal_exists\n"
            "      operator: ANY\n"
            '      signals: ["XSS"]\n'
        ),
    }
    evaluation = _deterministic_evaluate_rule(
        [RULE_001_STUB, rule_002],
        {
            "signals": ["SANS", "XSS"], "signal_details": [],
            "path": "/", "method": "GET", "domain": "d", "user_agent": "",
        },
    )
    assert evaluation.matched_rule_id == "RULE-002"
    assert evaluation.detected_signals == ["XSS"]


def test_precedence_falls_back_to_ip_rule_when_alone():
    evaluation = _deterministic_evaluate_rule(
        [RULE_001_STUB],
        {
            "signals": ["SANS"],
            "signal_details": [
                {"id": "SANS", "location": "IP", "value": "1.2.3.4", "detector": "SansRule"}
            ],
            "path": "/", "method": "GET", "domain": "d", "user_agent": "",
        },
    )
    assert evaluation.matched_rule_id == "RULE-001"
    # The explanation should name the flagged address, not just the signal tag.
    assert "1.2.3.4" in evaluation.explanation


# ==============================================================================
# Phase 3: coverage reporting
# ==============================================================================

def test_coverage_report_identifies_uncovered_signals():
    """Signals present in logs that no rule declares are rule-authoring candidates."""
    report = coverage_report("requests_sqli_xss_cmdexe.csv", rules_dir="rules")
    assert report["total_rows"] == 1000
    # These appear in the data but are declared by no rule.
    assert "CHALLENGE-TOKEN-INVALID" in report["uncovered_signals"]
    assert "DOUBLEENCODING" in report["uncovered_signals"]
    # Signals that ARE declared must not be listed as uncovered.
    assert "XSS" not in report["uncovered_signals"]
    assert "SQLI" not in report["uncovered_signals"]


def test_rule_signal_vocabulary_matches_rule_files():
    vocabulary = rule_signal_vocabulary("rules")
    assert "SANS" in vocabulary["RULE-001"]
    assert "XSS" in vocabulary["RULE-002"]
    assert "NOUA" in vocabulary["RULE-006"]
    # RULE-003 matches on paths only and declares no signals.
    assert vocabulary["RULE-003"] == []


# ==============================================================================
# Phase 4: full state return, checkpointing, shared context builder
# ==============================================================================

def test_run_waf_analysis_returns_full_state(vectorstore):
    """Follow-up questions need log_context and retrieved_rules, not just the match."""
    analysis = run_waf_analysis(
        csv_path="requests_maliciousip.csv",
        row_index=0,
        rules_dir="rules",
        vectorstore=vectorstore,
    )

    assert set(analysis) == {"thread_id", "final_output", "log_context", "retrieved_rules"}
    assert analysis["final_output"]["matched_rule_id"] == "RULE-001"

    # The request context survives for follow-up questions.
    context = analysis["log_context"]
    assert "SANS" in context["signals"]
    assert context["remote_ip"] == "205.210.31.135"
    assert context["user_agent"]  # Phase 3 added this; conditions depend on it

    # The rules retrieval considered, for "why not rule X?" questions.
    assert len(analysis["retrieved_rules"]) > 1
    assert any(r["rule_id"] == "RULE-001" for r in analysis["retrieved_rules"])


def test_process_waf_log_preserves_legacy_return_shape(vectorstore):
    """The original wrapper must keep returning the bare final_output dict."""
    result = process_waf_log(
        csv_path="requests_maliciousip.csv",
        row_index=0,
        rules_dir="rules",
        vectorstore=vectorstore,
    )
    assert "matched_rule_id" in result
    assert "thread_id" not in result
    assert "retrieved_rules" not in result


def test_thread_id_is_generated_and_echoed(vectorstore):
    analysis = run_waf_analysis(
        csv_path="requests_maliciousip.csv", row_index=0,
        rules_dir="rules", vectorstore=vectorstore,
    )
    assert analysis["thread_id"]

    explicit = run_waf_analysis(
        csv_path="requests_maliciousip.csv", row_index=0,
        rules_dir="rules", vectorstore=vectorstore, thread_id="my-session-1",
    )
    assert explicit["thread_id"] == "my-session-1"


def test_checkpointer_persists_state_under_thread_id(vectorstore):
    """The audit trail: a completed run is retrievable by its thread_id."""
    checkpointer = MemorySaver()
    app = create_waf_pipeline(vectorstore=vectorstore, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "audit-trail-1"}}

    app.invoke({"csv_path": "requests_maliciousip.csv", "row_index": 0}, config=config)

    restored = app.get_state(config)
    assert restored.values["final_output"]["matched_rule_id"] == "RULE-001"
    assert "SANS" in restored.values["log_context"]["signals"]


def test_state_stays_serializable(vectorstore):
    """Regression: a live QdrantClient or file buffer in state breaks checkpointing."""
    app = create_waf_pipeline(vectorstore=vectorstore)
    state = app.get_state({"configurable": {"thread_id": "serializable-check"}})
    assert state is not None  # compiles and is queryable with a checkpointer

    analysis = run_waf_analysis(
        csv_path="requests_maliciousip.csv", row_index=0,
        rules_dir="rules", vectorstore=vectorstore,
    )
    # Nothing unserializable leaked into the returned state.
    json.dumps(analysis)


def test_build_log_context_shared_by_app_and_pipeline():
    """The app builds context from its DataFrame; it must equal the node's output."""
    df = pd.read_csv("requests_maliciousip.csv", nrows=1)
    direct = build_log_context(df.iloc[0], 0)

    from waf_agent_pipeline import extract_log_context
    via_node = extract_log_context(
        {"csv_path": "requests_maliciousip.csv", "row_index": 0}
    )["log_context"]

    assert direct == via_node


def test_pipeline_runs_without_checkpointer(vectorstore):
    """checkpointer=False must still compile and run, for non-conversational use."""
    app = create_waf_pipeline(vectorstore=vectorstore, checkpointer=False)
    final_state = app.invoke({"csv_path": "requests_maliciousip.csv", "row_index": 0})
    assert final_state["final_output"]["matched_rule_id"] == "RULE-001"
