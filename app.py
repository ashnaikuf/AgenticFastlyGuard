"""Fastly CyberNode — WAF Rule Recommendation Engine (Streamlit Web Application).

Interactive web interface for Fastly Next-Gen WAF Agentic RAG pipeline.
Ingests WAF HTTP request log CSVs in-memory, extracts attack signals and request
telemetry, queries indexed WAF rules via Qdrant, and runs an AI agent to recommend
and verify condition logic.
"""

import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from waf_agent_pipeline import (
    NO_RULE_MATCH,
    build_log_context,
    build_waf_rule_vectorstore,
    extract_signal_details_from_csv,
    run_waf_analysis,
    signal_ids,
)
from waf_chat_agent import WafChatContext, answer, build_chat_agent

# Model used only to classify a question's scope before invoking the chat agent.
# Set to "" to skip classification entirely and let the agent handle scope via
# its system prompt, saving one LLM call per turn.
CHAT_CLASSIFIER_MODEL = os.getenv("WAF_CHAT_CLASSIFIER_MODEL", "gpt-4o-mini")

# Load environment configuration (.env)
load_dotenv()

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("waf_streamlit_app")

# Page Configuration
st.set_page_config(
    page_title="Fastly CyberNode — WAF Rule Recommendation Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Initializing vector store and indexing WAF rules in Qdrant...")
def load_vectorstore(rules_dir: str = "rules"):
    """Instantiate and cache the Qdrant vector store connection."""
    try:
        return build_waf_rule_vectorstore(rules_dir=rules_dir)
    except Exception as exc:
        logger.error("Failed to initialize vector store: %s", exc)
        st.error(
            f"❌ Unable to initialize Qdrant vector store: {exc}\n\n"
            "Please verify `QDRANT_APIKEY` and `CLUSTER_ENDPOINT` in your environment or `.env` file."
        )
        return None


def main():
    # Header & Description
    st.title("🛡️ Fastly CyberNode — WAF Rule Recommendation Engine")
    st.markdown(
        """
        Analyze incoming HTTP request logs containing security signals, retrieve embedded 
        Fastly Next-Gen WAF rules via **Agentic RAG**, and evaluate condition logic using 
        a **LangGraph Agent Workflow** to produce structured rule recommendations.
        """
    )
    st.divider()

    # Sidebar: System Status & Sample Data
    with st.sidebar:
        st.header("⚙️ Configuration & Status")
        rules_dir = "rules"
        rules_path = Path(rules_dir)
        if rules_path.is_dir():
            rule_files = sorted([f.name for f in rules_path.glob("RULE-*.md")])
            st.success(f"Loaded {len(rule_files)} Rule Definitions")
            with st.expander("Available WAF Rules"):
                for rf in rule_files:
                    st.markdown(f"- `{rf}`")
        else:
            st.error(f"Rules directory `{rules_dir}` not found.")

        # Vector store connection
        vectorstore = load_vectorstore(rules_dir=rules_dir)
        if vectorstore:
            st.info("Vector Store: Connected to Qdrant")

        st.subheader("📁 Sample Datasets")
        sample_options = ["(None - Upload File)", "requests_maliciousip.csv", "requests_sqli_xss_cmdexe.csv"]
        selected_sample = st.selectbox(
            "Quick-load Bundled Sample Log:",
            sample_options,
            index=0,
            help="Choose a pre-bundled log file to test the pipeline immediately.",
        )

        if st.button("🔄 Clear Vector Cache & Re-index"):
            st.cache_resource.clear()
            st.rerun()

    # Section 1: File Uploader
    st.subheader("1. Ingest Fastly WAF Log CSV")

    uploaded_file = st.file_uploader(
        "Upload Fastly WAF Log CSV file",
        type=["csv"],
        help="Upload an HTTP request log file in CSV format containing Fastly WAF telemetry and signals.",
    )

    # Resolve active file buffer (in-memory)
    active_buffer: Optional[io.BytesIO] = None
    source_filename = ""

    if uploaded_file is not None:
        source_filename = uploaded_file.name
        # Keep everything in memory as BytesIO
        active_buffer = io.BytesIO(uploaded_file.getvalue())
    elif selected_sample != "(None - Upload File)":
        sample_path = Path(selected_sample)
        if sample_path.exists():
            source_filename = sample_path.name
            active_buffer = io.BytesIO(sample_path.read_bytes())
            st.info(f"Using bundled sample dataset: `{source_filename}`")
        else:
            st.warning(f"Sample file `{selected_sample}` not found on disk.")

    if active_buffer is None:
        st.info("👆 Please upload a Fastly WAF log CSV file or select a sample dataset from the sidebar to continue.")
        return

    # In-memory DataFrame read
    active_buffer.seek(0)
    try:
        df = pd.read_csv(active_buffer)
    except Exception as err:
        st.error(f"Failed to parse CSV file: {err}")
        return

    total_rows = len(df)
    if total_rows == 0:
        st.warning("The uploaded CSV file is empty. Please upload a file with log rows.")
        return

    # Extract all signals in memory for overview
    active_buffer.seek(0)
    try:
        # Detail-preserving extraction: keeps each signal's detector/location/value
        # for the provenance view, with IDs derived for the existing summaries.
        all_signal_details = extract_signal_details_from_csv(active_buffer)
        all_signals = [signal_ids(d) for d in all_signal_details]
    except Exception as err:
        logger.warning("Could not extract signals overview: %s", err)
        all_signal_details = [[] for _ in range(total_rows)]
        all_signals = [[] for _ in range(total_rows)]

    # Metrics overview
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Log Rows", total_rows)
    with c2:
        rows_with_signals = sum(1 for sigs in all_signals if sigs)
        st.metric("Rows with Detected Signals", rows_with_signals)
    with c3:
        unique_signals = sorted(list({s for row_sigs in all_signals for s in row_sigs}))
        st.metric("Distinct Signal Types", len(unique_signals))
    with c4:
        st.metric("Active File", source_filename)

    # Preview Dataset
    with st.expander("🔍 Preview Loaded Log Data (First 5 Rows)", expanded=False):
        st.dataframe(df.head(5), width="stretch")

    st.divider()

    # Section 2: Controls & Row Selection
    st.subheader("2. Select Log Entry for Evaluation")

    col_ctrl, col_info = st.columns([1, 1])

    with col_ctrl:
        selector_mode = st.radio(
            "Selection Method:",
            ["Number Input", "Slider"],
            horizontal=True,
        )

        if selector_mode == "Slider" and total_rows > 1:
            row_index = st.slider(
                "Select Log Row Index:",
                min_value=0,
                max_value=total_rows - 1,
                value=0,
                step=1,
                help="Select the 0-based index of the log row to evaluate.",
            )
        else:
            row_index = st.number_input(
                "Select Log Row Index:",
                min_value=0,
                max_value=max(0, total_rows - 1),
                value=0,
                step=1,
                help="Select the 0-based index of the log row to evaluate (default: row 0).",
            )

    selected_row = df.iloc[int(row_index)]
    row_sigs = all_signals[int(row_index)] if int(row_index) < len(all_signals) else []

    with col_info:
        st.markdown(f"**Context for Row `{row_index}`:**")
        st.markdown(
            f"- **Method / Path:** `{selected_row.get('method', 'GET')}` `{selected_row.get('path', '/')}`\n"
            f"- **Host:** `{selected_row.get('server_name') or selected_row.get('server_hostname') or 'N/A'}`\n"
            f"- **Client IP:** `{selected_row.get('remote_ip', 'N/A')}`\n"
            f"- **Extracted Signals:** `{row_sigs if row_sigs else '[]'}`"
        )

    with st.expander(f"Inspect Complete Telemetry for Row #{row_index}", expanded=False):
        st.json(selected_row.dropna().to_dict())

    st.divider()

    # Section 3: Execution Button & Output Display
    st.subheader("3. Pipeline Execution & Rule Recommendation")

    run_analysis = st.button("🚀 Analyze Log & Recommend Rule", type="primary", width="stretch")

    if run_analysis:
        if vectorstore is None:
            st.error("Vector store is not initialized. Please check your Qdrant configuration.")
            return

        with st.spinner(f"Running LangGraph Agent Pipeline on Log Row #{row_index}..."):
            try:
                # Build the request context from the row already in memory. The
                # pipeline is checkpointed, so state must stay serializable — an
                # in-memory CSV buffer cannot be placed in it — and this also
                # avoids re-parsing the whole CSV on every analysis.
                log_context = build_log_context(selected_row, int(row_index))

                # A stable per-analysis thread_id keys this run's checkpoint, so
                # follow-up questions can be answered against the same state.
                thread_id = str(uuid4())
                analysis = run_waf_analysis(
                    row_index=int(row_index),
                    rules_dir=rules_dir,
                    vectorstore=vectorstore,
                    log_context=log_context,
                    thread_id=thread_id,
                )

                # Retain the full analysis for follow-up questions and reruns.
                # The DataFrame is kept too: the chat tools aggregate over it on
                # every turn, and each chat message triggers a Streamlit rerun,
                # so re-parsing the CSV per question would be wasteful.
                st.session_state["analysis"] = analysis
                st.session_state["thread_id"] = analysis["thread_id"]
                st.session_state["log_df"] = df
                st.session_state["log_name"] = source_filename
                st.session_state["analysis_row"] = int(row_index)
                # A new analysis starts a new conversation.
                st.session_state["chat_history"] = []
                st.session_state.pop("chat_agent", None)

            except Exception as e:
                logger.exception("Error executing WAF agent pipeline: %s", e)
                st.error(f"Error during WAF rule analysis: {e}")

    # Render the current analysis, if any. NEW: reads from session state rather
    # than only rendering inside the button block, so the results survive the
    # rerun that every chat message triggers.
    stored = st.session_state.get("analysis")
    if stored:
        if st.session_state.get("analysis_row") != int(row_index):
            st.info(
                f"Showing the analysis of row "
                f"**{st.session_state.get('analysis_row')}**. Click "
                f"**Analyze** to evaluate the currently selected row {row_index}."
            )
        _render_analysis(stored, all_signal_details)
        st.divider()
        _render_chat(rules_dir=rules_dir, vectorstore=vectorstore)


def _render_analysis(
    analysis: Dict[str, Any], all_signal_details: List[List[Dict[str, str]]]
) -> None:
    """Render a completed analysis: verification, summary, provenance, YAML."""
    result = analysis.get("final_output") or {}
    if not result:
        st.warning("Pipeline completed, but no rule match was returned.")
        return

    row_index = (analysis.get("log_context") or {}).get("row_index", 0)

    matched_rule_id = result.get("matched_rule_id", "N/A")
    action = result.get("action", "BLOCK")
    detected_signals = result.get("detected_signals", [])
    explanation = result.get("explanation", "")
    yaml_definition = result.get("yaml_definition", "")
    verified = result.get("verified", False)
    verification_level = result.get("verification_level", "")
    verification_notes = result.get("verification_notes", [])
    evaluator = result.get("evaluator", "")

    # No-match is a first-class outcome, not a failed analysis: it identifies
    # traffic that needs a new rule authored.
    if matched_rule_id == NO_RULE_MATCH:
        st.warning("⚠️ **NO RULE MATCH** — no indexed rule covers this request.")
        st.info(explanation)
        st.markdown(
            "**Next step:** author a rule covering the uncovered signals above, "
            "add it to `rules/`, then re-index from the sidebar."
        )
        return

    # Success Notification
    st.success(f"Analysis Complete! Matched Rule: **{matched_rule_id}** (Action: `{action}`)")

    # Verification banner
    if verified and verification_level == "verified":
        st.success(
            "✅ **Verified** — every reported signal is declared by the matched "
            "rule and present on the request, and all rule conditions were checked."
        )
    elif verified and verification_level == "signals_verified":
        st.info(
            "☑️ **Signals verified** — reported signals are grounded in both the "
            "rule and the request. " + " ".join(verification_notes)
        )
    else:
        st.error(
            "❌ **Unverified match** — the reported match contradicts the rule "
            "definition or the request log:"
        )
        for note in verification_notes:
            st.markdown(f"- {note}")

    # Structured Summary Cards
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric("Matched Rule ID", matched_rule_id)
    with res_col2:
        st.metric("Action", action)
    with res_col3:
        st.metric("Triggering Signals", ", ".join(detected_signals) if detected_signals else "None")

    # Explanation Block
    st.markdown("### Explanation")
    st.info(explanation)
    if evaluator:
        st.caption(f"Evaluated by: {evaluator}")

    # Signal provenance: which detector fired, where, and on what value
    signal_details = (
        all_signal_details[int(row_index)]
        if int(row_index) < len(all_signal_details)
        else []
    )
    if signal_details:
        with st.expander("🔎 Signal Provenance (detector, location, value)"):
            st.dataframe(pd.DataFrame(signal_details), width="stretch")

    # Raw YAML Condition Block
    st.markdown("### Raw YAML Condition Block")
    if yaml_definition:
        st.code(yaml_definition, language="yaml")
    else:
        st.warning("No YAML condition definition returned.")


def _render_chat(rules_dir: str, vectorstore: Optional[Any]) -> None:
    """Render the follow-up chat interface for the current analysis.

    Depends on session state populated by the analysis run: the agent's tools
    read the loaded DataFrame and the completed analysis, so there is nothing
    useful to ask before an analysis exists.
    """
    st.subheader("4. Ask Follow-up Questions")

    analysis = st.session_state.get("analysis")
    log_df = st.session_state.get("log_df")
    if analysis is None or log_df is None:
        st.info("Run an analysis above to start asking questions about it.")
        return

    st.caption(
        "Ask about this log file (counts, top values, examples), the indexed "
        "WAF rules, or why this rule was recommended."
    )

    # Build the agent once per analysis and keep it in session state, so the
    # conversation's checkpointed history survives Streamlit reruns.
    if "chat_agent" not in st.session_state:
        try:
            context = WafChatContext(
                df=log_df,
                vectorstore=vectorstore,
                analysis=analysis,
                rules_dir=rules_dir,
                csv_name=st.session_state.get("log_name", ""),
            )
            st.session_state["chat_agent"] = build_chat_agent(context)
        except Exception as exc:
            logger.exception("Could not initialize chat agent: %s", exc)
            st.error(f"Chat is unavailable: {exc}")
            return

    history = st.session_state.setdefault("chat_history", [])

    # Replay the conversation so far.
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("tools"):
                st.caption(f"via {', '.join(turn['tools'])}")

    question = st.chat_input("e.g. How many rows hit a SANS-flagged IP?")
    if not question:
        return

    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = answer(
                    st.session_state["chat_agent"],
                    question,
                    thread_id=st.session_state.get("thread_id", "waf-chat"),
                    classifier_model=CHAT_CLASSIFIER_MODEL,
                )
            except Exception as exc:
                logger.exception("Chat turn failed: %s", exc)
                st.error(f"Could not answer that: {exc}")
                history.pop()
                return

        st.markdown(reply["answer"])
        if reply["tools_used"]:
            st.caption(f"via {', '.join(reply['tools_used'])}")

    history.append({
        "role": "assistant",
        "content": reply["answer"],
        "tools": reply["tools_used"],
    })


if __name__ == "__main__":
    main()
