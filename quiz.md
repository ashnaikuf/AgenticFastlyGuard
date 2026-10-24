# Fastly CyberNode & Agentic RAG — Assessment Quiz

This quiz tests understanding of the core concepts, design patterns, and implementations in this repository, spanning:
- **Retrieval & Embeddings**: Dense vs. Sparse retrieval, BM25, FAISS, Cross-Encoder / LLM Re-Ranking.
- **Vector Database**: Qdrant indexing, Cosine similarity, payload filtering, content hashing, and orphan management.
- **Agent Architectures**: Agentic RAG, ReAct loop, Corrective RAG (CRAG) Graders, Self-Reflection, Query Planning & Decomposition, and Adaptive Intent-Based Routing.
- **WAF Security Engine**: Rule evaluation, precedence hierarchy, independent verification nodes, and `NO_RULE_MATCH` gap detection.
- **Application & Tool Safety**: Structured agent tool arguments, anti-prompt-injection design, and Streamlit session caching.

---

## Multiple-Choice Questions

### Section 1: Retrieval Architectures & Vector Stores

#### Question 1
In `research/09_Dense_Sparse.py`, what is the primary architectural motivation for combining a Dense Retriever (`FAISS` + `HuggingFaceEmbeddings`) with a Sparse Retriever (`BM25Retriever`) using LangChain's `EnsembleRetriever`?
- **A)** Dense retrieval excels at semantic synonyms and conceptual intent (e.g., mapping "marine organism system" to coral reefs) but can dilute exact technical acronyms (e.g., "qubits"), which BM25 captures via exact keyword overlap.
- **B)** Dense retrieval only works on structured relational database tables, whereas BM25 is required for unstructured text documents.
- **C)** Sparse retrieval requires a dedicated GPU cluster, whereas dense embeddings run without mathematical vector calculations.
- **D)** Dense retrieval is used exclusively to cache API responses, while BM25 is used to serialize documents into JSON.

---

#### Question 2
In the two-stage retrieval implementations (`research/10a_ReRanking.py` and `research/10b_ReRanking.py`), why is broad initial retrieval (e.g., $k=8$) performed before applying an LLM or Cross-Encoder re-ranker?
- **A)** Vector stores like FAISS throw an index underflow error if queried with $k < 8$.
- **B)** Initial vector search is fast and optimizes for high recall across candidate chunks, allowing a slower, higher-precision model to re-score and re-order the candidates for maximum relevance.
- **C)** The initial vector retrieval stage removes duplicate stop words, which LLMs cannot process directly.
- **D)** Fast vector search produces generative responses, while the re-ranker converts text back into vector embeddings.

---

#### Question 3
What specific engineering safeguard was introduced in `research/10b_ReRanking.py` to prevent formatting and parsing failures in the re-ranking stage compared to `10a_ReRanking.py`?
- **A)** Setting the re-ranking LLM's `temperature=0.0`, enforcing a strict bracketed array prompt (`[index, index, index]`), and stripping brackets before comma-splitting digits.
- **B)** Setting `temperature=1.0` and using regular expressions to capture markdown bullet lists.
- **C)** Calling an external search engine to verify document relevance scores before parsing.
- **D)** Suppressing all Python runtime exceptions by wrapping the parser in an infinite retry loop.

---

#### Question 4
According to `rules/README.md`, why are WAF rules stored as Markdown files with embedded YAML blocks (`.md`) rather than pure YAML (`.yaml`) files?
- **A)** Qdrant cannot ingest files with an extension other than `.md`.
- **B)** Natural language headers and descriptions produce higher-quality semantic dense vectors during search, while embedded YAML blocks preserve deterministic structured condition logic for rule evaluation.
- **C)** Plain YAML files cannot be loaded by Python's `open()` function in modern versions of Python.
- **D)** Markdown files compress vector dimensionality by 50% automatically during ingestion.

---

#### Question 5
In `waf_agent_pipeline.py` (`build_waf_rule_vectorstore`), how does the indexing engine prevent redundant embedding computations and point duplications across application restarts?
- **A)** It computes a SHA-256 hash of each rule's content, compares it to the stored payload metadata in Qdrant, and generates deterministic point IDs using `uuid5`.
- **B)** It wipes and re-creates the entire Qdrant collection on every startup.
- **C)** It stores all vector points in local browser session storage.
- **D)** It skips indexing whenever the operating system uptime exceeds 10 minutes.

---

#### Question 6
Why does `ensure_payload_indexes` in `waf_agent_pipeline.py` explicitly register keyword payload schemas on fields like `metadata.action` and `metadata.rule_id` in Qdrant?
- **A)** Qdrant rejects filtered queries with an HTTP 400 error (`"Index required but not found"`) if filtered payload fields are not explicitly indexed.
- **B)** Payload indexes are required to compute Cosine similarity distance between dense vectors.
- **C)** Qdrant deletes collections that lack payload indexes after 60 seconds of inactivity.
- **D)** Keyword indexing reduces the dimensional size of 1536-dimensional vectors to 512 dimensions.

---

### Section 2: Agentic RAG, Routing, & StateGraph Workflows

#### Question 7
How does the Agentic RAG pattern in `research/24_AgenticRAG.py` fundamentally differ from a traditional naive RAG pipeline?
- **A)** Traditional RAG requires fine-tuning the foundational model, whereas Agentic RAG operates without an LLM.
- **B)** Traditional RAG is a rigid, one-shot "retrieve-then-generate" sequence, whereas Agentic RAG executes over a stateful graph where nodes can reason, plan, use tools, evaluate, and dynamically route or loop.
- **C)** Agentic RAG is restricted to Edge VCL and cannot run in Python runtimes.
- **D)** Traditional RAG uses LangGraph StateGraph, while Agentic RAG is restricted to linear shell scripts.

---

#### Question 8
In `waf_agent_pipeline.py`, why is the live `QdrantClient` bound to the retrieval node via `functools.partial` at graph creation rather than stored inside `WAFState`?
- **A)** `WAFState` only permits primitive string data types.
- **B)** LangGraph checkpointers (such as `MemorySaver`) serialize state at every node boundary, and live network client connections cannot be serialized.
- **C)** QdrantClient terminates the underlying socket if referenced inside a Python dictionary.
- **D)** Live network clients violate Python's typing rules when using `TypedDict`.

---

#### Question 9
In `research/31_Adaptive_RoutingRAG.py`, how does the `router_node` determine whether a query should route to the internal vector store or external web search?
- **A)** By checking if the query string starts with `"http://"` or `"https://"`.
- **B)** By prompting the LLM with structured output (`RouteQuery` containing `Literal["vectorstore", "web_search"]`) to classify the user's intent based on knowledge domain.
- **C)** By running both retrievers in parallel and measuring which returns data faster.
- **D)** By inspecting the network ping latency to Tavily.

---

#### Question 10
In `research/28_QueryPlanDecompose.py`, how does query decomposition handle complex multi-part questions?
- **A)** It compresses the multi-part question into a single 3-word keyword string.
- **B)** It prompts the LLM planner to break the complex query into 2–3 sub-questions, retrieves context sequentially for each sub-question, and merges all documents before generating a consolidated answer.
- **C)** It generates an SQL `UNION` statement and executes it against FAISS.
- **D)** It routes each sub-question to a different proprietary LLM provider.

---

### Section 3: ReAct Agents, Graders, & Self-Correction

#### Question 11
What is the core execution loop executed by a ReAct agent, as demonstrated in `research/25a_ReAct.py` and `waf_chat_agent.py`?
- **A)** Tokenize -> Embed -> Quantize -> Output
- **B)** Think (Reason) -> Act (Call Tool) -> Observe (Inspect Tool Output) -> Reflect / Final Answer
- **C)** Load CSV -> Train Model -> Predict -> Export
- **D)** Compile VCL -> Push to Edge -> Cache Purge -> Reload

---

#### Question 12
In `research/25b_ReAct.py`, how does the ReAct agent determine which tool to select when faced with multiple registered data sources (e.g., internal docs vs. academic research papers)?
- **A)** It executes all registered tools sequentially in alphabetical order.
- **B)** It matches regex patterns against hardcoded domain keywords in the prompt.
- **C)** The LLM reasons over the natural-language description provided for each registered tool to select the one whose scope matches the current sub-goal.
- **D)** It executes all tools concurrently and takes a majority vote on the results.

---

#### Question 13
In Corrective RAG (`research/25c_AgenticRAG_Grader.py`), what decision does the `grade_documents` node make?
- **A)** It calculates the dollar cost of OpenAI API calls before generating text.
- **B)** It evaluates whether the retrieved documents are relevant to the user query ("yes" vs "no") to route to generation or trigger query rewriting / fallback.
- **C)** It deletes chunks that have a cosine similarity below 0.1 from the vector store.
- **D)** It grades the grammar and readability of the user's prompt.

---

#### Question 14
In `research/27_SelfReflection.py`, what is the critical architectural distinction between a **Document Grader** and an **Answer Reflector**?
- **A)** Document Graders check for API rate limits, while Answer Reflectors check for network timeouts.
- **B)** Document Graders catch retrieval failures (irrelevant context), while Answer Reflectors catch generation failures (hallucinations, omissions, or ungrounded synthesis).
- **C)** Document Graders run only on the client side, while Answer Reflectors run on edge CDN nodes.
- **D)** Document Graders validate Markdown syntax, while Answer Reflectors validate YAML indentation.

---

#### Question 15
Why do the graph node functions in `research/27_SelfReflection.py` return `state.model_copy(update={...})` rather than mutating attributes on the incoming `state` object directly?
- **A)** Pydantic models automatically lock their memory with OS-level read-only permissions.
- **B)** Returning a new copy preserves immutability, ensuring that LangGraph reliably tracks state transitions and history across node boundaries.
- **C)** Direct attribute mutation triggers an unrecoverable segmentation fault in Python's garbage collector.
- **D)** Mutating attributes directly causes FAISS to clear its vector cache.

---

### Section 4: Fastly WAF Pipeline & Independent Verification

#### Question 16
In `waf_agent_pipeline.py`, what is the rule precedence rule implemented when an incoming request contains BOTH an OWASP attack signal (e.g., `SQLI`, `XSS`) AND an IP reputation signal (e.g., `SITE-FLAGGED-IP`)?
- **A)** The pipeline randomly selects between RULE-001 and RULE-002.
- **B)** Attack-specific rules (RULE-002, RULE-003, RULE-004) take precedence over IP reputation rules (RULE-001); RULE-001 is only selected when an IP threat signal is the sole threat.
- **C)** IP reputation rules always supersede attack rules because IP checks are faster at the edge.
- **D)** The pipeline merges both rules into a newly generated rule called `RULE-COMBINED`.

---

#### Question 17
When an incoming request log contains signals or attributes that no indexed WAF rule covers, how does `waf_agent_pipeline.py` respond?
- **A)** It silently returns the rule with the highest vector similarity score to guarantee an action is always recommended.
- **B)** It returns `NO_RULE_MATCH` with action `NONE`, reporting the uncovered signals to identify gaps where new WAF rules should be authored.
- **C)** It raises an unhandled `IndexError` that halts the server.
- **D)** It synthesizes a new YAML rule using GPT-4o and commits it directly to the `rules/` directory.

---

#### Question 18
Why does the evaluator architecture in `waf_agent_pipeline.py` separate `WAFRuleEvaluation` from `WAFRuleMatchResult`?
- **A)** `WAFRuleEvaluation` is for HuggingFace models, while `WAFRuleMatchResult` is for OpenAI models.
- **B)** Separation of duties: the evaluator (LLM or deterministic) must not certify its own output; an independent node (`verify_match`) verifies the match against facts and attaches verification metadata.
- **C)** Pydantic does not allow optional fields on a single model.
- **D)** `WAFRuleEvaluation` stores vector embeddings, while `WAFRuleMatchResult` stores raw strings.

---

#### Question 19
How does the independent verification node (`verify_match`) in `waf_agent_pipeline.py` handle a rate-limit rule like `RULE-005` on a single log row?
- **A)** It marks the verification as `VERIFY_FAILED` because rate limiting is strictly prohibited in Next-Gen WAFs.
- **B)** It verifies condition criteria (path, method, signal) but marks the verification level as `signals_verified`, noting that rate thresholds span multiple requests and cannot be verified from a single row.
- **C)** It assumes the threshold was exceeded if the client IP address contains an odd number.
- **D)** It blocks the entire subnet automatically without verifying signals.

---

#### Question 20
What automated safeguard occurs in `verify_match` (`waf_agent_pipeline.py`) if an LLM evaluator produces a hallucinated match (e.g., citing a signal not declared by the rule or not present on the request)?
- **A)** The pipeline immediately crashes and returns an HTTP 500 error.
- **B)** The hallucinated match is delivered to the user with an unverified warning flag.
- **C)** The match fails verification, the pipeline rejects the LLM output, and it automatically falls back to re-evaluating the request using the deterministic rule evaluator.
- **D)** The pipeline prompts the user to manually classify the attack via a terminal dialog.

---

### Section 5: Secure Agent Tooling & Streamlit Integration

#### Question 21
Why do the analytics tools in `waf_chat_agent.py` (e.g., `query_log_counts`, `aggregate_log_column`) take structured parameters instead of having the LLM generate and execute dynamic SQL or Pandas Python code?
- **A)** Pandas and Python are unable to perform grouping operations on in-memory DataFrames.
- **B)** Executing model-generated code or queries against DataFrames introduces prompt injection and arbitrary code execution vulnerabilities; structured parameters keep operations deterministic, testable, and secure.
- **C)** Dynamic code generation is only supported when running on Linux containers.
- **D)** The LLM would be unable to read CSV headers without a commercial plugin.

---

#### Question 22
In `waf_chat_agent.py`, what strict grounding mandate is enforced regarding Fastly attack signal identifiers (e.g., `SUSPECTED-BOT` vs. `SUSPECTED-BAD-BOT`)?
- **A)** The agent is instructed to treat similar signal names as interchangeable synonyms to keep answers readable.
- **B)** Signal names are exact identifiers and must never be used interchangeably; the agent must never claim a rule matched a signal unless that signal appears in that rule's own signals list.
- **C)** All signals must be converted to generic labels such as "Suspicious Activity".
- **D)** If a signal contains the word "BOT", it must automatically trigger an immediate permanent IP ban.

---

#### Question 23
In `app.py`, why are `@st.cache_resource` and `st.session_state` used to manage the vector store connection and analysis results?
- **A)** Streamlit reruns the entire Python script on every user interaction; `@st.cache_resource` prevents re-indexing Qdrant on every click, while `st.session_state` preserves the active dataset, analysis state, and chat history.
- **B)** Streamlit cannot access local disk files without using `@st.cache_resource`.
- **C)** `st.session_state` is required to encrypt all displayed HTML output.
- **D)** Caching is required by Streamlit to enable GPU acceleration on tabular data.

---

#### Question 24
Why are analysis runs in `waf_agent_pipeline.py` and chat turns in `waf_chat_agent.py` executed with a persistent `thread_id` and LangGraph's `MemorySaver`?
- **A)** To compress the local vector store to save RAM.
- **B)** To persist state across turns, enabling the conversational agent to recall and explain earlier analyses (such as what rules were retrieved and why a rule matched) within the same thread.
- **C)** To force all LLM inferences to execute synchronously on a single CPU thread.
- **D)** To prevent more than one user from opening the Streamlit application at the same time.

---

## Answer Key & Explanations

| Question | Correct Answer | Primary Code Reference | Technical Explanation |
| :---: | :---: | :--- | :--- |
| **1** | **A** | `research/09_Dense_Sparse.py` | Hybrid RAG balances semantic synonym retrieval (Dense) with exact token/keyword hits (BM25 Sparse). |
| **2** | **B** | `research/10a_ReRanking.py`, `10b_ReRanking.py` | Two-stage RAG: fast broad search optimizes recall; precision re-ranking optimizes ranking order. |
| **3** | **A** | `research/10b_ReRanking.py` | `temperature=0.0`, strict bracketed array templates, and character stripping eliminate parsing failures. |
| **4** | **B** | `rules/README.md` | Markdown text provides natural semantic vectors; embedded YAML provides deterministic logic. |
| **5** | **A** | `waf_agent_pipeline.py` | SHA-256 hash checking against payload metadata and deterministic `uuid5` IDs prevent duplicate embeddings. |
| **6** | **A** | `waf_agent_pipeline.py` | Qdrant requires explicit keyword payload indexes on fields used in query filters (`FieldCondition`). |
| **7** | **B** | `research/24_AgenticRAG.py` | Agentic RAG replaces rigid one-shot pipelines with dynamic, multi-turn state graphs with tool calling. |
| **8** | **B** | `waf_agent_pipeline.py` | StateGraph checkpointers serialize state at node boundaries; live network clients are not serializable. |
| **9** | **B** | `research/31_Adaptive_RoutingRAG.py` | Structured LLM output (`with_structured_output`) dynamically classifies intent to route execution branches. |
| **10** | **B** | `research/28_QueryPlanDecompose.py` | Decomposing complex queries into sub-questions allows targeted retrieval across multiple topics before synthesis. |
| **11** | **B** | `research/25a_ReAct.py`, `waf_chat_agent.py` | ReAct executes a continuous cycle of Reasoning (Think), Action (Tool Call), and Observation (Result). |
| **12** | **C** | `research/25b_ReAct.py` | ReAct agents use tool descriptions to semantically match user sub-goals to specific data capabilities. |
| **13** | **B** | `research/25c_AgenticRAG_Grader.py` | CRAG document grading evaluates document relevance before generation to trigger query rewrites or fallbacks. |
| **14** | **B** | `research/27_SelfReflection.py` | Document Graders catch retrieval errors; Answer Reflectors catch generation errors/hallucinations. |
| **15** | **B** | `research/27_SelfReflection.py` | `model_copy(update={...})` guarantees immutability, enabling LangGraph to maintain reliable state history. |
| **16** | **B** | `waf_agent_pipeline.py` | Attack-specific rules (RULE-002/003/004) outrank IP reputation rules (RULE-001) when an attack payload is present. |
| **17** | **B** | `waf_agent_pipeline.py` | `NO_RULE_MATCH` avoids false confidence from nearest vector neighbours and flags gaps for rule authoring. |
| **18** | **B** | `waf_agent_pipeline.py` | Evaluators must not certify themselves; `verify_match` independently validates claims against ground facts. |
| **19** | **B** | `waf_agent_pipeline.py` | Rate-limit rules require state across many requests; single rows can only verify signals and conditions. |
| **20** | **C** | `waf_agent_pipeline.py` | If LLM evaluation fails verification, the pipeline automatically recovers via deterministic evaluation. |
| **21** | **B** | `waf_chat_agent.py` | Structured tool parameters prevent prompt injection, untrusted code execution, and non-deterministic logic. |
| **22** | **B** | `waf_chat_agent.py` | Fastly attack signals are exact detector identifiers; treating them as synonyms produces ungrounded claims. |
| **23** | **A** | `app.py` | `@st.cache_resource` preserves expensive connections; `st.session_state` preserves state across UI reruns. |
| **24** | **B** | `waf_agent_pipeline.py`, `waf_chat_agent.py` | Thread IDs and checkpointers maintain conversational memory and enable auditability over prior analyses. |
