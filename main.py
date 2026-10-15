"""Main execution entrypoint for Agentic Fastly Next-Gen WAF Rule Matching.

Demonstrates:
1. Ingestion & vector indexing of rules/RULE-*.md into Qdrant.
2. Signal extraction from requests_sqli_xss_cmdexe.csv using pandas.
3. LangGraph agent pipeline execution (extract_log_context -> retrieve_waf_rules -> evaluate_rule_agent).
4. Formatted structured JSON output with matched_rule_id, action, detected_signals, explanation, and yaml_definition.
"""

import json
import logging
from waf_agent_pipeline import (
    build_waf_rule_vectorstore,
    extract_signals_from_csv,
    process_waf_log,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main")


def main():
    CSV_FILE = "requests_sqli_xss_cmdexe.csv"
    RULES_DIR = "rules"

    logger.info("Step 1: Ingesting and vector indexing WAF rules from %s/...", RULES_DIR)
    vectorstore = build_waf_rule_vectorstore(rules_dir=RULES_DIR)

    logger.info("Step 2: Testing CSV signal extraction on %s...", CSV_FILE)
    extracted_signals = extract_signals_from_csv(CSV_FILE)
    logger.info("Parsed %d rows. Sample signals from row 0: %s", len(extracted_signals), extracted_signals[0])

    logger.info("Step 3: Running LangGraph Agent Pipeline on attack log row 0 (containing OWASP attack signal)...")
    result = process_waf_log(
        csv_path=CSV_FILE,
        row_index=0,
        rules_dir=RULES_DIR,
        vectorstore=vectorstore,
    )

    print("\n" + "=" * 80)
    print("FINAL AGENT STRUCTURED JSON OUTPUT:")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
