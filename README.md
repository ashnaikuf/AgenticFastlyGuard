# From Edge Telemetry to Automated Defense: Mining WAF Logs with Python
### *AgenticFastlyGuard — Fastly CyberNode Next-Gen WAF Recommendation Engine*

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Fastly Next-Gen WAF](https://img.shields.io/badge/Fastly-Next--Gen%20WAF-red.svg)](https://www.fastly.com/products/next-gen-waf)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Vector%20Store-Qdrant-crimson.svg)](https://qdrant.tech/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

---

## 📺 Presentation & Live Demo Video

> **Watch the full tech talk and live demonstration on YouTube:**  
> 🔗 **[YouTube Video: From Edge Telemetry to Automated Defense: Mining WAF Logs with Python](https://www.youtube.com/watch?v=TODO)**   
>
> *Experience the live walkthrough: scanning gigabytes of edge telemetry, uncovering a stealthy distributed botnet through behavioral fingerprints, and deploying an active defense rule in under two minutes.*

---

## Abstract & Overview

Web Application Firewalls (WAFs) log millions of web requests daily, but turning raw telemetry into actionable mitigation is traditionally slow and manual. Security engineers spend hours parsing giant log dumps to construct custom blocking rules by hand—often long after an attack has caused impact.

To solve this, **AgenticFastlyGuard** is an open-source Python tool that extracts raw security logs from **Fastly’s Next-Gen WAF (NGWAF)** and uses machine learning and **Agentic RAG (Retrieval-Augmented Generation)** to cluster suspicious behavioral patterns (request timing, header ordering, and TLS/HTTP fingerprints like JA3/JA4). Once identified, the tool automatically synthesizes, verifies, and deploys targeted WAF blocking rules in real time.

---

## What You Will Learn in This Video & Project

1. **The Manual Bottleneck**: Why traditional manual log analysis fails against modern automated threats, credential stuffing, and distributed botnets.
2. **Feature Extraction & Clustering**: How unsupervised ML and telemetry parsers group malicious traffic based on behavioral signatures (JA3/JA4 TLS fingerprints, request timing, header distributions) rather than static IPs or easily spoofed User-Agents.
3. **Automated Rule Generation & Verification**: Automatically synthesizing and deploying declarative mitigation rules into Fastly NGWAF without human lag, backed by an independent machine verification node that eliminates LLM hallucinations.
4. **Live Demonstration**: Scanning gigabytes of log telemetry, uncovering a stealthy botnet, and deploying an active defense rule in under two minutes.

---

## System Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                  FASTLY EDGE TELEMETRY                                 │
│  - HTTP Request Logs (CSV / Streams)                                                   │
│  - Attack Signals (XSS, SQLi, CMDEXE, Malicious IP, SANS, Suspected Bots)              │
│  - Fingerprints (JA3/JA4 hashes, TLS protocols, Header ordering, GeoIP)                │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        AGENTIC RAG PIPELINE (LangGraph Workflow)                       │
│                                                                                        │
│   ┌─────────────────────────┐         ┌────────────────────────────────────────────┐   │
│   │ 1. extract_log_context  │         │          WAF Rule Base (Markdown + YAML)   │   │
│   │ Normalizes telemetry,   │         │          - rules/RULE-*.md                 │   │
│   │ extracts attack signals │         │          - rules/signal_categories.yaml    │   │
│   └────────────┬────────────┘         └─────────────────────┬──────────────────────┘   │
│                │                                            │                          │
│                ▼                                            ▼                          │
│   ┌─────────────────────────┐                  ┌───────────────────────────┐           │
│   │ 2. retrieve_waf_rules   │◄─────────────────┤   Qdrant Vector Store     │           │
│   │ Semantic vector search  │                  │   Dense semantic indexing │           │
│   │ (OpenAI / HuggingFace)  │                  └───────────────────────────┘           │
│   └────────────┬────────────┘                                                          │
│                │                                                                       │
│                ▼                                                                       │
│   ┌─────────────────────────┐                                                          │
│   │ 3. evaluate_rule_agent  │  Evaluates declarative YAML conditions;                  │
│   │ Rule Candidate Matcher  │  Precedence: Application Attack > IP Reputation;         │
│   │                         │  Returns NO_RULE_MATCH on coverage gaps.                 │
│   └────────────┬────────────┘                                                          │
│                │                                                                       │
│                ▼                                                                       │
│   ┌─────────────────────────┐                                                          │
│   │ 4. verify_match         │  INDEPENDENT VERIFICATION GUARDRAIL:                     │
│   │ Machine-checked logic   │  Evaluator cannot self-certify. Verifies condition logic │
│   │                         │  against telemetry -> Verified / Signals Verified.       │
│   └────────────┬────────────┘                                                          │
└────────────────┼───────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           INTERACTIVE CONSUMPTION & DEFENSE                            │
│                                                                                        │
│   ┌─────────────────────────────────────────┐   ┌──────────────────────────────────┐   │
│   │ Fastly CyberNode Web UI (Streamlit)     │   │ Grounded ReAct Chat Analyst      │   │
│   │ - Interactive log upload & inspection   │   │ - Safe structured analytic tools │   │
│   │ - Instant rule recommendations          │   │ - Query distributions & counts   │   │
│   │ - Telemetry metrics & visual insights   │   │ - Zero arbitrary code execution  │   │
│   └─────────────────────────────────────────┘   └──────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Dual-Format WAF Rule Engineering (`rules/RULE-*.md`)
- **Semantic Dense Embeddings**: Human-readable prose, metadata, and intent are embedded into **Qdrant** for high-recall semantic vector matching.
- **Deterministic Condition Execution**: Exact Fastly condition logic is declared in fenced YAML blocks, allowing machine parsing, strict precedence routing, and automated deployment.

### 2. Zero-Hallucination Independent Verification (`verify_match`)
- Large Language Models are prone to overconfidence when recommending security rules.
- `AgenticFastlyGuard` enforces an **independent verification node** that isolates the evaluator from certification.
- Condition logic (`signal_exists`, `path_contains`, `user_agent_like`, `user_agent_equals`) is validated against actual log attributes.
- Outcomes are tagged with definitive levels: `verified`, `signals_verified`, `failed`, or `no_match`.

### 3. Precedence & Gap Detection Hierarchy
- **Attack Precedence**: Application-layer attack signals (e.g., SQLi, XSS, CMDEXE) outrank pure IP reputation signals, ensuring multi-vector attacks are mitigated by root-cause attack rules rather than generic IP blocks.
- **Explicit Gap Detection**: Uncovered signals explicitly emit `NO_RULE_MATCH`, alerting security teams to missing rule definitions rather than hallucinating the nearest unrelated rule.

### 4. Fastly CyberNode Streamlit Web Application (`app.py`)
- In-memory CSV log ingestion with instant telemetry dashboards.
- Quick-load bundled sample logs for immediate demonstrations.
- Interactive log row inspection displaying request headers, JA3/JA4 TLS fingerprints, server names, and detected attack signals.
- Single-click automated rule recommendation and verification pipeline.

### 5. Grounded ReAct Chat Assistant (`waf_chat_agent.py`)
- Provides conversational investigation over loaded logs and the WAF rule base.
- **Security-First Tooling**: Employs strictly structured tools (`query_log_counts`, `aggregate_log_column`, `sample_log_rows`, `search_waf_rules`, `get_rule_definition`) with column allowlists.
- **Zero Arbitrary Code Execution**: Never runs LLM-generated Python or SQL against customer logs.

### 6. Advanced RAG & Machine Learning Research Lab (`research/`)
The repository contains 10+ standalone research scripts exploring advanced retrieval and agentic designs:
- `09_Dense_Sparse.py`: Hybrid RAG / Ensemble retrieval (Dense FAISS + Sparse BM25).
- `10a_ReRanking.py` & `10b_ReRanking.py`: Two-stage retrieval with LLM and Cross-Encoder re-ranking.
- `24_AgenticRAG.py`: Stateful LangGraph RAG workflows.
- `25a_ReAct.py` & `25b_ReAct.py`: Reasoning + Acting loops.
- `25c_AgenticRAG_Grader.py`: Corrective RAG (CRAG) with document relevance grading.
- `27_SelfReflection.py`: Self-reflection and iterative query refinement.
- `28_QueryPlanDecompose.py`: Multi-step query planning and decomposition.
- `31_Adaptive_RoutingRAG.py`: Adaptive intent-based dynamic routing.

---

## Project Structure

```text
AgenticFastlyGuard/
├── app.py                      # Fastly CyberNode Streamlit Web Application
├── waf_agent_pipeline.py       # Core LangGraph Agent Pipeline & Verification Engine
├── waf_chat_agent.py           # Grounded ReAct Chat Agent with Structured Analytics Tools
├── run_streamlit.sh            # One-click execution script for the Streamlit UI
├── pyproject.toml              # Project dependencies and packaging configuration
├── quiz.md                     # Assessment quiz covering RAG and WAF security architecture
├── csv/                        # Real-world Fastly NGWAF attack telemetry samples
│   ├── requests_maliciousip.csv       # Telemetry flagged with IP reputation & SANS signals
│   └── requests_sqli_xss_cmdexe.csv   # Telemetry flagged with OWASP attack vectors
├── rules/                      # Fastly Next-Gen WAF Rule Knowledge Base
│   ├── README.md               # Rule engineering and formatting standards
│   ├── signal_categories.yaml  # Normalized signal taxonomy (ip_reputation, attack, bot, auth)
│   ├── RULE-001-malicious-ip-block.md
│   ├── RULE-002-owasp-core.md
│   ├── RULE-003-sensitive-file-access.md
│   ├── RULE-004-client-side-detection.md
│   ├── RULE-005-login-attempt-rate-limit.md
│   └── RULE-006-BadUserAgents.md
├── research/                   # Advanced RAG and Agentic Research Scripts
│   ├── main.py                 # CLI execution runner
│   ├── 09_Dense_Sparse.py
│   ├── 10a_ReRanking.py
│   ├── 10b_ReRanking.py
│   ├── 24_AgenticRAG.py
│   ├── 25a_ReAct.py / 25b_ReAct.py
│   ├── 25c_AgenticRAG_Grader.py
│   ├── 27_SelfReflection.py
│   ├── 28_QueryPlanDecompose.py
│   └── 31_Adaptive_RoutingRAG.py
└── tests/                      # Automated test suite (78 tests)
    ├── test_app.py             # Streamlit application UI tests
    ├── test_chat_agent.py      # ReAct agent & structured analytics tool tests
    └── test_waf_pipeline.py    # Pipeline, vectorstore, condition logic & verifier tests
```

---

## Installation & Setup

### Prerequisites
- **Python**: Version `3.12` or higher
- **Virtual Environment**: `uv` (recommended) or standard `venv`
- **Vector Database**: Qdrant Cloud or local Qdrant instance

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/AgenticFastlyGuard.git
cd AgenticFastlyGuard
```

### 2. Create Virtual Environment & Install Dependencies
Using `uv`:
```bash
uv sync
```
Or using standard `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Configure Environment Variables
Create a `.env` file in the project root:
```env
# OpenAI API Key (used for LLM evaluation, embeddings, and chat agent)
OPENAI_API_KEY="your-openai-api-key"

# Qdrant Vector Database Configuration
QDRANT_APIKEY="your-qdrant-api-key"
CLUSTER_ENDPOINT="https://your-qdrant-cluster.qdrant.tech:6333"
QDRANT_COLLECTION_NAME="fastly_wafrules"

# Pipeline Settings
WAF_RETRIEVAL_LIMIT=10
WAF_CHAT_CLASSIFIER_MODEL="gpt-4o-mini"

# Optional Keys for Research Scripts
GROQ_API_KEY=""
TAVILY_API_KEY=""
```

*Note: If `OPENAI_API_KEY` is omitted, the pipeline automatically falls back to local HuggingFace embeddings (`all-MiniLM-L6-v2`) and deterministic rule evaluation.*

---

## Usage Guide

### 1. Launch the Streamlit Web Application
Run the helper script or invoke Streamlit directly:
```bash
./run_streamlit.sh
# or
streamlit run app.py
```
Open your browser at `http://localhost:8501`:
1. Select a bundled dataset from the sidebar (e.g., `requests_sqli_xss_cmdexe.csv`) or upload your own CSV log export.
2. Select an HTTP request row index to inspect.
3. Click **"🚀 Analyze Log & Recommend Rule"** to trigger the LangGraph pipeline.
4. Review the matched Rule ID, recommended action (`BLOCK`), detected signals, machine verification status, and generated YAML policy.
5. Use the built-in **AI Security Analyst Chat** to ask natural-language questions about your log data and rule coverage.

### 2. Run the Command-Line Pipeline
To run a sample analysis on edge telemetry directly from the terminal:
```bash
python research/main.py
```

### 3. Run the Automated Test Suite
Verify pipeline integrity, verifier logic, and chat tools across 78 unit and integration tests:
```bash
PYTHONPATH=. pytest
```

---

## Fastly Next-Gen WAF Rule Schema

Rules are defined under `rules/RULE-*.md`. Each rule combines natural language documentation with an authoritative YAML condition block:

```markdown
## Rule: RULE-002 - OWASP Core Attack Prevention

### Metadata & Overview
- **Rule ID**: `RULE-002`
- **Rule Name**: OWASP Core Attack Prevention
- **Target Domain**: Global / All Domains
- **Action**: `BLOCK`
- **Threat Category**: Application Attacks / Injection / Exploitation Tools
- **Description**: Detects and blocks web application attack vectors including Cross-Site Scripting (XSS), SQL Injection (SQLi), Path Traversal, and Command Execution (CMDEXE).

### Condition Logic
```yaml
rule_id: RULE-002
action: BLOCK
conditions:
  operator: ALL
  rules:
    - type: signal_exists
      operator: ANY
      signals:
        - "XSS"
        - "SQLI"
        - "Traversal"
        - "CMDEXE"
        - "Attack Tooling"
```

## License

This project is licensed under the Apache 2.0 License.
