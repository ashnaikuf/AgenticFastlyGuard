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

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from waf_agent_pipeline import (
    build_waf_rule_vectorstore,
    extract_signals_from_csv,
    process_waf_log,
)

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
        all_signals = extract_signals_from_csv(active_buffer)
    except Exception as err:
        logger.warning("Could not extract signals overview: %s", err)
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
                # Reset buffer position before passing into process_waf_log
                active_buffer.seek(0)

                # Process the log row directly using the in-memory buffer
                result = process_waf_log(
                    csv_path=active_buffer,
                    row_index=int(row_index),
                    rules_dir=rules_dir,
                    vectorstore=vectorstore,
                )

                if not result:
                    st.warning("Pipeline completed, but no rule match was returned.")
                    return

                matched_rule_id = result.get("matched_rule_id", "N/A")
                action = result.get("action", "BLOCK")
                detected_signals = result.get("detected_signals", [])
                explanation = result.get("explanation", "")
                yaml_definition = result.get("yaml_definition", "")

                # Success Notification
                st.success(f"Analysis Complete! Matched Rule: **{matched_rule_id}** (Action: `{action}`)")

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

                # Raw YAML Condition Block
                st.markdown("### Raw YAML Condition Block")
                if yaml_definition:
                    st.code(yaml_definition, language="yaml")
                else:
                    st.warning("No YAML condition definition returned.")

                # Final Agent Structured JSON Output (matching main.py CLI output)
                st.markdown("### Final Agent Structured JSON Output")
                st.json(result)

                # CLI Console Format
                st.markdown("#### Exact CLI Format (Matching `main.py`)")
                cli_output = (
                    "=" * 80 + "\n"
                    "FINAL AGENT STRUCTURED JSON OUTPUT:\n"
                    "=" * 80 + "\n"
                    f"{json.dumps(result, indent=2)}\n"
                    "=" * 80
                )
                st.code(cli_output, language="text")

            except Exception as e:
                logger.exception("Error executing WAF agent pipeline: %s", e)
                st.error(f"Error during WAF rule analysis: {e}")


if __name__ == "__main__":
    main()
