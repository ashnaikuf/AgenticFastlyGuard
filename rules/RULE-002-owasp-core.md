## Rule: RULE-002 - OWASP Core Attack Prevention

### Metadata & Overview
- **Rule ID**: `RULE-002`
- **Rule Name**: OWASP Core Attack Prevention
- **Target Domain**: Global / All Domains
- **Action**: `BLOCK`
- **Threat Category**: Application Attacks / Injection / Exploitation Tools
- **Description**: Detects and blocks web application attack vectors including Cross-Site Scripting (XSS), SQL Injection (SQLi), Path Traversal, Command Execution (CMDEXE), and automated attack scanners or vulnerability exploitation tools.

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
