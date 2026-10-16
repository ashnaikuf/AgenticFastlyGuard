# Fastly NG WAF Rule Base for Vector Search & Agentic RAG

This directory holds one Markdown file per WAF rule. Files are ingested by
`build_waf_rule_vectorstore()` in `waf_agent_pipeline.py`, which globs
`RULE-*.md` — so this README is documentation only and is never indexed.

## Adding a rule

1. Create `rules/RULE-0NN-short-slug.md`.
2. Start the file with a Level 2 header: `## Rule: RULE-0NN - Human Readable Name`.
   The ingestion splitter keys off this exact prefix.
3. Include a `### Metadata & Overview` bullet list and a `### Condition Logic`
   fenced `yaml` block. The YAML block is the authoritative definition — the
   pipeline parses `rule_id`, `action`, and nested `signals` from it, falling
   back to the prose only if the YAML is missing or unparseable.
4. Re-run the indexer to rebuild the Qdrant collection.

## Architectural Decision: File Format Strategy for RAG Ingestion

**Selected Format:** Markdown with Embedded Structuring (`.md`)

### Format Comparison Analysis
| Criterion | Markdown (.md) | Plain YAML (.yaml) |
| :--- | :--- | :--- |
| **Dense Vector Embeddings** | **Superior**: Natural language headers and descriptions convert cleanly into high-quality semantic vector space. | **Suboptimal**: Key-value pairs often create noisy semantic vectors with low cosine similarity against natural query phrasing. |
| **Agent / LLM Parsing** | **High**: LLMs naturally parse both Markdown syntax and embedded structural logic (e.g., YAML/JSON blocks). | **High**: Standardized, deterministic data structure. |
| **Chunking Utility** | **Native**: Chunking frameworks (e.g., LangChain's `MarkdownHeaderTextSplitter`) cleanly separate discrete rules without splitting conditions across chunks. | **Manual**: Standard recursive chunkers often break nested YAML objects mid-rule unless custom loaders are built. |

### Ingestion Guidelines
1. **Chunk Boundary:** Split on Level 2 Headers (`## Rule: ...`). Each rule must exist as an independent chunk.
2. **Metadata Strategy:** Store `rule_id`, `action`, and `threat_category` as document metadata attributes alongside the vector embedding.
3. **Retrieval Workflow:**
   - Step 1: Perform dense semantic search on the *Overview & Intent* section using incoming CSV attack descriptions.
   - Step 2: Extract the exact rule definition from the retrieved vector document's *Condition Logic* block.
