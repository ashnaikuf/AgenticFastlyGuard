"""Tests for the conversational layer (waf_chat_agent).

The tool tests here are deliberately LLM-free: they invoke each tool directly
with structured arguments and assert exact figures against the bundled CSVs.
The tools are where correctness lives — if a tool returns a wrong count, no
amount of prompt engineering above it will produce a right answer.
"""

import pandas as pd
import pytest

from waf_agent_pipeline import build_waf_rule_vectorstore, run_waf_analysis
from waf_chat_agent import (
    AGGREGATABLE_COLUMNS,
    MAX_SAMPLE_ROWS,
    OUT_OF_SCOPE_REPLY,
    ROUTE_OUT_OF_SCOPE,
    QuestionRoute,
    WafChatContext,
    answer,
    build_chat_tools,
    tools_used,
)

CSV = "requests_maliciousip.csv"


@pytest.fixture(scope="module")
def vectorstore():
    return build_waf_rule_vectorstore(rules_dir="rules")


@pytest.fixture(scope="module")
def log_df():
    return pd.read_csv(CSV)


@pytest.fixture(scope="module")
def analysis(vectorstore):
    return run_waf_analysis(
        csv_path=CSV, row_index=0, rules_dir="rules", vectorstore=vectorstore
    )


@pytest.fixture
def tools(log_df, vectorstore, analysis):
    context = WafChatContext(
        df=log_df, vectorstore=vectorstore, analysis=analysis, csv_name=CSV
    )
    return {t.name: t for t in build_chat_tools(context)}


@pytest.fixture
def tools_without_analysis(log_df, vectorstore):
    context = WafChatContext(df=log_df, vectorstore=vectorstore, csv_name=CSV)
    return {t.name: t for t in build_chat_tools(context)}


# ==============================================================================
# Tool registration
# ==============================================================================

def test_expected_tools_are_registered(tools):
    assert set(tools) == {
        "search_waf_rules",
        "get_rule_definition",
        "query_log_counts",
        "aggregate_log_column",
        "sample_log_rows",
        "explain_current_recommendation",
        "check_signal_coverage",
    }


def test_every_tool_has_a_description(tools):
    """ReAct routing is driven entirely by these descriptions."""
    for name, t in tools.items():
        assert t.description and len(t.description) > 30, name


# ==============================================================================
# Log analytics: exact figures, no LLM
# ==============================================================================

def test_query_log_counts_signal_filter(tools):
    out = tools["query_log_counts"].invoke({"signal": "SANS"})
    assert "467 of 1000" in out


def test_query_log_counts_normalizes_signal_spelling(tools):
    """'SigSci IP' in rule files vs 'SIGSCI-IP' in logs must count the same."""
    hyphenated = tools["query_log_counts"].invoke({"signal": "SIGSCI-IP"})
    spaced = tools["query_log_counts"].invoke({"signal": "SigSci IP"})
    assert "546 of 1000" in hyphenated
    assert "546 of 1000" in spaced


def test_query_log_counts_distinguishes_similar_signals(tools):
    """SUSPECTED-BOT and SUSPECTED-BAD-BOT are different detectors."""
    bot = tools["query_log_counts"].invoke({"signal": "SUSPECTED-BOT"})
    bad_bot = tools["query_log_counts"].invoke({"signal": "SUSPECTED-BAD-BOT"})
    assert "748 of 1000" in bot
    assert "252 of 1000" in bad_bot


def test_query_log_counts_with_no_filters_reports_total(tools):
    assert "1000 rows in total" in tools["query_log_counts"].invoke({})


def test_query_log_counts_combines_filters_with_and(tools):
    out = tools["query_log_counts"].invoke({"signal": "SANS", "method": "GET"})
    assert "of 1000 rows match" in out
    assert "signal='SANS'" in out and "method='GET'" in out


def test_query_log_counts_unmatched_filters_return_zero(tools):
    out = tools["query_log_counts"].invoke({"remote_ip": "203.0.113.255"})
    assert "0 of 1000" in out


def test_aggregate_signals_frequency(tools):
    out = tools["aggregate_log_column"].invoke({"column": "signals", "top_n": 3})
    assert "BOT-ANALYSIS: 1000 rows" in out
    assert "SUSPECTED-BOT: 748 rows" in out


def test_aggregate_rejects_freetext_columns(tools):
    """Grouping on summation/request_headers would produce meaningless buckets."""
    for column in ("summation", "request_headers", "signals_raw"):
        out = tools["aggregate_log_column"].invoke({"column": column})
        assert "not available for aggregation" in out


def test_aggregate_allowlist_columns_all_work(tools, log_df):
    for column in AGGREGATABLE_COLUMNS:
        if column not in log_df.columns:
            continue
        out = tools["aggregate_log_column"].invoke({"column": column, "top_n": 2})
        assert "Top" in out and column in out


def test_aggregate_caps_top_n(tools):
    """A chat turn must not be able to dump the whole column into context."""
    out = tools["aggregate_log_column"].invoke({"column": "remote_ip", "top_n": 999})
    value_lines = [line for line in out.splitlines() if line.startswith("  ")]
    assert len(value_lines) == 25


def test_sample_log_rows_returns_bounded_examples(tools):
    out = tools["sample_log_rows"].invoke({"signal": "SANS", "limit": 2})
    assert "467 rows match; showing 2" in out
    assert out.count("- row ") == 2


def test_sample_log_rows_caps_limit(tools):
    out = tools["sample_log_rows"].invoke({"signal": "SANS", "limit": 500})
    assert out.count("- row ") <= MAX_SAMPLE_ROWS


def test_sample_log_rows_handles_no_matches(tools):
    out = tools["sample_log_rows"].invoke({"path_contains": "/nonexistent-xyz"})
    assert "No rows match" in out


# ==============================================================================
# Rule knowledge
# ==============================================================================

def test_get_rule_definition_returns_yaml_and_signals(tools):
    out = tools["get_rule_definition"].invoke({"rule_id": "RULE-001"})
    assert "RULE-001" in out
    assert "SANS" in out
    assert "rule_id: RULE-001" in out


def test_get_rule_definition_is_case_insensitive(tools):
    assert "RULE-002" in tools["get_rule_definition"].invoke({"rule_id": "rule-002"})


def test_get_rule_definition_reports_unknown_rule(tools):
    out = tools["get_rule_definition"].invoke({"rule_id": "RULE-999"})
    assert "No indexed rule" in out


def test_get_rule_definition_grounds_the_original_hallucination(tools):
    """RULE-001 must not appear to declare SUSPECTED-BOT.

    The bug that started this work was a confident claim that RULE-001 matched
    on SUSPECTED-BOT. The tool the agent consults before making such a claim
    must show otherwise.
    """
    out = tools["get_rule_definition"].invoke({"rule_id": "RULE-001"})
    declared = out.split("Declared signals:")[1].split("\n")[0]
    assert "SUSPECTED-BOT" not in declared


def test_search_waf_rules_returns_candidates(tools):
    out = tools["search_waf_rules"].invoke({"query": "malicious IP reputation block"})
    assert "RULE-" in out


# ==============================================================================
# Prior analysis
# ==============================================================================

def test_explain_current_recommendation_reports_verified_match(tools):
    out = tools["explain_current_recommendation"].invoke({})
    assert "RULE-001" in out
    assert "SANS" in out
    assert "Verified: True" in out
    # Follow-ups like "what else was considered?" need the candidate list.
    assert "Other rules retrieval considered" in out


def test_explain_current_recommendation_without_analysis(tools_without_analysis):
    out = tools_without_analysis["explain_current_recommendation"].invoke({})
    assert "No analysis has been run yet" in out


def test_check_signal_coverage_lists_uncovered_signals(tools):
    out = tools["check_signal_coverage"].invoke({})
    assert "DATACENTER" in out
    assert "PRIVATEFILE" in out
    # A covered signal must not be reported as uncovered.
    assert "SANS:" not in out


# ==============================================================================
# Session state isolation
# ==============================================================================

def test_signal_details_are_cached_per_context(log_df, vectorstore):
    """Chat turns must not re-parse the signals JSON for every question."""
    context = WafChatContext(df=log_df, vectorstore=vectorstore, csv_name=CSV)
    first = context.signal_details
    assert first is context.signal_details  # same object, not recomputed


def test_tools_degrade_without_vectorstore(log_df):
    context = WafChatContext(df=log_df, vectorstore=None, csv_name=CSV)
    tools = {t.name: t for t in build_chat_tools(context)}
    assert "not available" in tools["get_rule_definition"].invoke({"rule_id": "RULE-001"})
    assert "not available" in tools["search_waf_rules"].invoke({"query": "bots"})
    # Log analytics must still work: they do not depend on Qdrant.
    assert "467 of 1000" in tools["query_log_counts"].invoke({"signal": "SANS"})


# ==============================================================================
# Phase 6: routing and orchestration (LLM-free)
# ==============================================================================

class _ExplodingAgent:
    """Stands in for the chat agent; fails if it is invoked at all."""

    def invoke(self, *args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("agent must not be invoked for out-of-scope questions")


def test_out_of_scope_route_skips_the_agent_entirely():
    """The whole point of the router: no ReAct loop, no tool calls, no cost."""
    reply = answer(
        _ExplodingAgent(),
        "What is the weather in Chicago?",
        route=ROUTE_OUT_OF_SCOPE,
    )
    assert reply["answer"] == OUT_OF_SCOPE_REPLY
    assert reply["tools_used"] == []
    assert reply["route"] == ROUTE_OUT_OF_SCOPE


def test_classification_is_skipped_when_no_classifier_model():
    """classifier_model="" must not trigger a classification call."""
    class _Stub:
        def invoke(self, payload, config=None):
            return {"messages": [type("M", (), {"content": "ok", "tool_calls": []})()]}

    reply = answer(_Stub(), "How many rows?", classifier_model="")
    assert reply["route"] == "agent"
    assert reply["answer"] == "ok"


def test_tools_used_extracted_from_message_history():
    """The UI reports what actually ran, read from the returned messages."""
    class _Msg:
        def __init__(self, calls):
            self.tool_calls = calls

    messages = [
        _Msg([{"name": "query_log_counts"}]),
        _Msg([]),
        _Msg([{"name": "get_rule_definition"}, {"name": "query_log_counts"}]),
    ]
    assert tools_used(messages) == ["query_log_counts", "get_rule_definition"]


def test_tools_used_handles_messages_without_tool_calls():
    assert tools_used([object(), object()]) == []


def test_question_route_schema_rejects_unknown_category():
    with pytest.raises(Exception):
        QuestionRoute(category="something_else")
