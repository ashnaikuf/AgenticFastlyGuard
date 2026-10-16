"""Unit and integration tests for Fastly Next-Gen WAF Agent Pipeline."""

import json
import os
from pathlib import Path
import pytest
from qdrant_client import models

from waf_agent_pipeline import (
    WAFRuleMatchResult,
    build_waf_rule_vectorstore,
    create_waf_pipeline,
    extract_signals_from_csv,
    get_embeddings,
    get_qdrant_vector_size,
    process_waf_log,
)


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
    app = create_waf_pipeline(vectorstore=vectorstore)

    multi_signal_state = {
        "log_context": {
            "signals": ["BLOCKED", "BOT-ANALYSIS", "XSS", "SQLI"],
            "path": "/rest/products/search",
            "domain": "owaspjuiceshop.secureedge.xyz",
            "method": "POST",
        },
        "vectorstore": vectorstore,
    }

    final_state = app.invoke(multi_signal_state)
    result = final_state.get("final_output", {})

    validated = WAFRuleMatchResult(**result)
    assert validated.matched_rule_id == "RULE-002"
    assert validated.action == "BLOCK"
    assert "XSS" in validated.detected_signals
    assert "SQLI" in validated.detected_signals
    assert "rule_id: RULE-002" in validated.yaml_definition
