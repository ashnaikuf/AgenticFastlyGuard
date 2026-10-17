"""Conversational layer over the Fastly Next-Gen WAF agent pipeline.

Answers follow-up questions about a completed analysis, the indexed WAF rules,
and the loaded log file, using a ReAct agent with three classes of tool:

- **Rule knowledge** (`search_waf_rules`, `get_rule_definition`) — semantic and
  exact lookups against the Qdrant-indexed rule Markdown.
- **Log analytics** (`query_log_counts`, `aggregate_log_column`,
  `sample_log_rows`) — aggregation over the parsed log DataFrame.
- **Prior result** (`explain_current_recommendation`, `check_signal_coverage`) —
  reads the state the pipeline already produced for this analysis.

Design note on log analytics: these tools take *structured arguments*, not
generated SQL or pandas expressions. Embedding log rows in a vector store cannot
answer "how many", and generating code to run against the DataFrame would mean
executing model output. Structured operations keep answers deterministic,
testable, and free of code execution, at the cost of only answering the shapes
of question enumerated below.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from waf_agent_pipeline import (
    NO_RULE_MATCH,
    coverage_report,
    extract_signal_details_from_dataframe,
    get_embeddings,
    get_qdrant_vector_size,
    normalize_signal_id,
    signal_ids,
)

load_dotenv()
logger = logging.getLogger("waf_chat_agent")

# Columns the aggregation tool is allowed to group by. Restricting this keeps
# the model from grouping on free-text blobs (request_headers, summation) that
# would produce thousands of meaningless one-row buckets.
AGGREGATABLE_COLUMNS = (
    "remote_ip",
    "path",
    "method",
    "user_agent",
    "country",
    "response_code",
    "server_name",
    "server_hostname",
    "ja3",
    "ja4",
    "tls_protocol",
)

# Cap on rows returned by sampling tools, so a chat turn cannot dump the file
# into the model's context.
MAX_SAMPLE_ROWS = 10

SYSTEM_PROMPT = """You are a Fastly Next-Gen WAF analyst assistant. You answer \
questions about one loaded HTTP request log and a set of indexed WAF rules.

GROUNDING RULES — these are not stylistic preferences:
1. Never state a count, frequency, or row figure you did not get from a tool \
result. If you need a number, call a tool.
2. Never claim a rule matched a signal unless that signal appears in that \
rule's own signals list. Use `get_rule_definition` to check before asserting.
3. Signal names are exact identifiers, not descriptions. 'SUSPECTED-BOT' and \
'SUSPECTED-BAD-BOT' are different Fastly detectors and must never be treated \
as equivalent or used interchangeably.
4. If no rule covers a signal, say so plainly and suggest it as a candidate for \
a new rule. Do not substitute the closest-looking rule.
5. If a question is not about this log file or these WAF rules, say that it is \
outside what you can answer. Do not speculate.

TOOL SELECTION:
- Questions about what a rule does or why it matched -> `get_rule_definition`, \
`search_waf_rules`.
- Questions about the current analysis result -> `explain_current_recommendation`.
- Questions with 'how many', 'which', 'most', 'top', 'distribution' -> \
`query_log_counts`, `aggregate_log_column`.
- Questions asking for concrete examples -> `sample_log_rows`.
- Questions about gaps in rule coverage -> `check_signal_coverage`.

Be concise. Report figures exactly as the tools return them."""


@dataclass
class WafChatContext:
    """State a chat session answers questions against.

    Holds the loaded log, the completed analysis, and the Qdrant client, so the
    tools close over real state instead of reading module globals.
    """

    df: pd.DataFrame
    vectorstore: Optional[QdrantClient] = None
    analysis: Optional[Dict[str, Any]] = None
    rules_dir: str = "rules"
    csv_name: str = ""
    # Lazily computed per-row signal details, cached for the session: the chat
    # turn should not re-parse the signals JSON for every question.
    _signal_details: Optional[List[List[Dict[str, str]]]] = field(
        default=None, repr=False
    )

    @property
    def signal_details(self) -> List[List[Dict[str, str]]]:
        if self._signal_details is None:
            self._signal_details = extract_signal_details_from_dataframe(self.df)
        return self._signal_details

    @property
    def row_signals(self) -> List[List[str]]:
        return [signal_ids(d) for d in self.signal_details]


def build_chat_tools(context: WafChatContext) -> List[Any]:
    """Create the tool set bound to one chat session's state."""

    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "fastly_wafrules")

    # ---------------------------------------------------------------- rules ---

    @tool
    def search_waf_rules(query: str) -> str:
        """Semantically search the indexed Fastly WAF rule definitions.

        Use for open-ended questions about which rule covers a kind of threat,
        e.g. "which rule handles bot traffic" or "rules about sensitive paths".
        For a specific known rule ID, prefer get_rule_definition.
        """
        if context.vectorstore is None:
            return "The WAF rule vector store is not available in this session."
        try:
            vector_size = get_qdrant_vector_size(context.vectorstore, collection_name)
            query_vector = get_embeddings(dimensions=vector_size).embed_query(query)
            points = context.vectorstore.query_points(
                collection_name=collection_name,
                query=query_vector,
                using="dense_vector",
                limit=4,
                with_payload=True,
            ).points
        except Exception as exc:
            logger.warning("Rule search failed: %s", exc)
            return f"Rule search failed: {exc}"

        if not points:
            return "No rules matched that query."

        lines = []
        for point in points:
            metadata = (point.payload or {}).get("metadata", {})
            lines.append(
                f"- {metadata.get('rule_id', '?')} (action={metadata.get('action', '?')}, "
                f"signals={metadata.get('signals') or 'none'}, "
                f"source={metadata.get('source', '?')})"
            )
        return "Matching rules:\n" + "\n".join(lines)

    @tool
    def get_rule_definition(rule_id: str) -> str:
        """Return one rule's full text and raw YAML condition block by exact ID.

        Use this before making any claim about what a rule matches on. rule_id
        is an identifier like 'RULE-001'.
        """
        if context.vectorstore is None:
            return "The WAF rule vector store is not available in this session."
        wanted = rule_id.strip().upper()
        try:
            points = context.vectorstore.query_points(
                collection_name=collection_name,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.rule_id",
                            match=models.MatchValue(value=wanted),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
            ).points
        except Exception as exc:
            logger.warning("Rule lookup failed: %s", exc)
            return f"Rule lookup failed: {exc}"

        if not points:
            return f"No indexed rule with ID {wanted!r}."

        payload = points[0].payload or {}
        metadata = payload.get("metadata", {})
        return (
            f"{wanted} (action={metadata.get('action')}, "
            f"source={metadata.get('source')})\n"
            f"Declared signals: {metadata.get('signals') or 'none'}\n\n"
            f"Raw YAML:\n{metadata.get('yaml_definition') or '(none)'}\n\n"
            f"Description:\n{payload.get('page_content', '')[:1200]}"
        )

    # ------------------------------------------------------------ analytics ---

    def _matching_indices(
        signal: str = "",
        path_contains: str = "",
        method: str = "",
        remote_ip: str = "",
    ) -> List[int]:
        """Row indices satisfying all supplied filters (empty filters ignored)."""
        df = context.df
        signals_per_row = context.row_signals
        wanted_signal = normalize_signal_id(signal) if signal else ""

        matches: List[int] = []
        for i in range(len(df)):
            if wanted_signal:
                if wanted_signal not in {
                    normalize_signal_id(s) for s in signals_per_row[i]
                }:
                    continue
            row = df.iloc[i]
            if path_contains and path_contains not in str(row.get("path") or ""):
                continue
            if method and str(row.get("method") or "").upper() != method.upper():
                continue
            if remote_ip and str(row.get("remote_ip") or "") != remote_ip:
                continue
            matches.append(i)
        return matches

    @tool
    def query_log_counts(
        signal: str = "",
        path_contains: str = "",
        method: str = "",
        remote_ip: str = "",
    ) -> str:
        """Count log rows matching any combination of filters.

        Use for "how many requests..." questions. All arguments are optional and
        combine with AND; omit the ones you do not need.

        Args:
            signal: exact signal ID, e.g. 'SQLI' or 'SANS'.
            path_contains: substring to find in the request path, e.g. '/login'.
            method: HTTP method, e.g. 'POST'.
            remote_ip: exact client IP address.
        """
        if not any([signal, path_contains, method, remote_ip]):
            return f"The loaded log has {len(context.df)} rows in total."

        matches = _matching_indices(signal, path_contains, method, remote_ip)
        criteria = ", ".join(
            f"{name}={value!r}"
            for name, value in (
                ("signal", signal), ("path_contains", path_contains),
                ("method", method), ("remote_ip", remote_ip),
            )
            if value
        )
        return (
            f"{len(matches)} of {len(context.df)} rows match {criteria}. "
            f"Example row indices: {matches[:5] or 'none'}"
        )

    @tool
    def aggregate_log_column(column: str, top_n: int = 10) -> str:
        """Show the most frequent values of a log column, or of request signals.

        Use for "which IPs hit us most", "what are the top paths", or
        "what signals appear in this file".

        Args:
            column: one of the log columns (remote_ip, path, method, user_agent,
                country, response_code, server_name, ja3, ja4, tls_protocol) or
                the special value 'signals' for signal frequency.
            top_n: how many values to return (capped at 25).
        """
        top_n = max(1, min(int(top_n), 25))

        if column.lower() in ("signals", "signal"):
            counts: Dict[str, int] = {}
            for row in context.row_signals:
                for s in set(row):
                    counts[s] = counts.get(s, 0) + 1
            ordered = sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
            body = "\n".join(f"  {name}: {n} rows" for name, n in ordered)
            return f"Signal frequency across {len(context.df)} rows:\n{body}"

        if column not in AGGREGATABLE_COLUMNS:
            return (
                f"Column {column!r} is not available for aggregation. "
                f"Choose one of: {', '.join(AGGREGATABLE_COLUMNS)}, or 'signals'."
            )
        if column not in context.df.columns:
            return f"The loaded log has no {column!r} column."

        counts = context.df[column].astype(str).value_counts().head(top_n)
        body = "\n".join(f"  {value}: {n} rows" for value, n in counts.items())
        return f"Top {column} values across {len(context.df)} rows:\n{body}"

    @tool
    def sample_log_rows(
        signal: str = "",
        path_contains: str = "",
        limit: int = 3,
    ) -> str:
        """Return a few concrete example rows, for "show me" questions.

        Args:
            signal: exact signal ID to filter on, e.g. 'CODEINJECTION'.
            path_contains: substring to find in the request path.
            limit: how many rows to return (capped at 10).
        """
        limit = max(1, min(int(limit), MAX_SAMPLE_ROWS))
        matches = _matching_indices(signal=signal, path_contains=path_contains)
        if not matches:
            return "No rows match those filters."

        lines = []
        for i in matches[:limit]:
            row = context.df.iloc[i]
            lines.append(
                f"- row {i}: {row.get('method')} {row.get('path')} "
                f"from {row.get('remote_ip')} "
                f"(status {row.get('response_code')}) "
                f"signals={context.row_signals[i]}"
            )
        return (
            f"{len(matches)} rows match; showing {min(limit, len(matches))}:\n"
            + "\n".join(lines)
        )

    # ------------------------------------------------------- prior analysis ---

    @tool
    def explain_current_recommendation() -> str:
        """Explain the rule recommendation already produced for the selected row.

        Use for "why this rule", "what did you match", "show me the YAML you
        matched", or "what else was considered". Reports the verification result
        and the other rules retrieval returned.
        """
        if not context.analysis:
            return (
                "No analysis has been run yet in this session. Run an analysis "
                "on a log row first."
            )

        output = context.analysis.get("final_output", {})
        log_context = context.analysis.get("log_context", {})
        retrieved = context.analysis.get("retrieved_rules", [])

        if output.get("matched_rule_id") == NO_RULE_MATCH:
            return (
                "The selected request matched NO rule.\n"
                f"Signals present: {log_context.get('signals')}\n"
                f"Reason: {output.get('explanation')}\n"
                f"Rules considered: {[r.get('rule_id') for r in retrieved]}"
            )

        return (
            f"Matched rule: {output.get('matched_rule_id')} "
            f"(action={output.get('action')})\n"
            f"Triggering signals: {output.get('detected_signals')}\n"
            f"Explanation: {output.get('explanation')}\n"
            f"Verified: {output.get('verified')} "
            f"(level={output.get('verification_level')})\n"
            f"Verification notes: {output.get('verification_notes') or 'none'}\n"
            f"Evaluated by: {output.get('evaluator')}\n"
            f"Request: {log_context.get('method')} {log_context.get('path')} "
            f"on {log_context.get('domain')} from {log_context.get('remote_ip')}\n"
            f"All signals on the request: {log_context.get('signals')}\n"
            f"Other rules retrieval considered: "
            f"{[r.get('rule_id') for r in retrieved]}\n\n"
            f"Matched rule YAML:\n{output.get('yaml_definition') or '(none)'}"
        )

    @tool
    def check_signal_coverage() -> str:
        """Report which signals in the loaded log no indexed rule covers.

        Use for "what traffic can't we explain", "which signals have no rule",
        or "what rule should I write next".
        """
        if not context.csv_name:
            return (
                "Coverage analysis needs the log file path, which is not "
                "available in this session."
            )
        try:
            report = coverage_report(context.csv_name, rules_dir=context.rules_dir)
        except Exception as exc:
            return f"Coverage analysis failed: {exc}"

        uncovered = report["uncovered_signals"]
        if not uncovered:
            return "Every signal in this log is declared by at least one rule."

        body = "\n".join(f"  {name}: {n} rows" for name, n in list(uncovered.items())[:15])
        return (
            f"{report['rows_with_no_covered_signal']} of {report['total_rows']} rows "
            "have no signal that any rule declares.\n"
            f"Signals present in the log that no rule covers:\n{body}"
        )

    return [
        search_waf_rules,
        get_rule_definition,
        query_log_counts,
        aggregate_log_column,
        sample_log_rows,
        explain_current_recommendation,
        check_signal_coverage,
    ]


def build_chat_agent(
    context: WafChatContext,
    model: str = "gpt-4o",
    checkpointer: Optional[Any] = None,
) -> Any:
    """Create the ReAct chat agent for a session.

    Uses the same model as the analysis pipeline so the project needs one
    provider and one key. Compiled with a checkpointer so conversation history
    persists per ``thread_id``.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0)
    tools = build_chat_tools(context)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return create_agent(
        llm,
        tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def ask(agent: Any, question: str, thread_id: str = "waf-chat") -> str:
    """Send one question to the agent and return its final text answer."""
    return answer(agent, question, thread_id=thread_id)["answer"]


# ==============================================================================
# Routing and orchestration
# ==============================================================================

ROUTE_RULES_KB = "rules_kb"
ROUTE_LOG_DATA = "log_data"
ROUTE_PRIOR_RESULT = "prior_result"
ROUTE_OUT_OF_SCOPE = "out_of_scope"

OUT_OF_SCOPE_REPLY = (
    "That's outside what I can answer. I can help with questions about the "
    "loaded WAF log file (counts, top values, example requests) or about the "
    "indexed WAF rules and the current rule recommendation."
)


class QuestionRoute(BaseModel):
    """Classification of a user question, used to skip pointless tool loops."""

    category: Literal["rules_kb", "log_data", "prior_result", "out_of_scope"] = Field(
        description=(
            "rules_kb: about WAF rule definitions or which rule covers something. "
            "log_data: about the loaded log file's contents, counts, or examples. "
            "prior_result: about the rule recommendation already produced. "
            "out_of_scope: anything else, including general knowledge, current "
            "events, weather, or unrelated coding questions."
        )
    )


def classify_question(question: str, model: str = "gpt-4o") -> str:
    """Classify a question before invoking the agent.

    The only load-bearing case is ``out_of_scope``: it lets an unrelated
    question be answered without running the ReAct loop or any tool call. The
    other three categories are informational — the agent routes between tools
    perfectly well on its own via tool descriptions — so a misclassification
    among them costs nothing.

    Falls back to ``log_data`` (i.e. let the agent decide) if classification
    fails, so a classifier outage never blocks a legitimate question.
    """
    from langchain_openai import ChatOpenAI

    try:
        llm = ChatOpenAI(model=model, temperature=0)
        route = llm.with_structured_output(QuestionRoute).invoke(
            "Classify this question about a Fastly WAF log analysis session.\n\n"
            f"Question: {question}"
        )
        return route.category
    except Exception as exc:
        logger.warning("Question classification failed (%s); routing to agent.", exc)
        return ROUTE_LOG_DATA


def tools_used(messages: List[Any]) -> List[str]:
    """Names of tools invoked while producing an answer, for UI transparency.

    Read from the returned message history rather than inferred, so the UI
    reports what actually ran.
    """
    names: List[str] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name and name not in names:
                names.append(name)
    return names


def answer(
    agent: Any,
    question: str,
    thread_id: str = "waf-chat",
    route: Optional[str] = None,
    classifier_model: str = "",
) -> Dict[str, Any]:
    """Answer one question, short-circuiting out-of-scope requests.

    Returns ``{"answer", "route", "tools_used"}``. Pass ``route`` to skip
    classification, or ``classifier_model=""`` to skip it entirely and let the
    agent handle scope itself via its system prompt.
    """
    if route is None and classifier_model:
        route = classify_question(question, model=classifier_model)

    if route == ROUTE_OUT_OF_SCOPE:
        # Answered without invoking the agent: no tool calls, no ReAct loop.
        return {"answer": OUT_OF_SCOPE_REPLY, "route": route, "tools_used": []}

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = result["messages"]
    return {
        "answer": messages[-1].content,
        "route": route or "agent",
        "tools_used": tools_used(messages),
    }
