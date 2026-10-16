## Rule: RULE-004 - Advanced Client-Side Detection on Sensitive Endpoint

### Metadata & Overview
- **Rule ID**: `RULE-004`
- **Rule Name**: Advanced Client-Side Detection on Sensitive Endpoint
- **Target Domain**: `dimensionash26.secureedge.xyz`
- **Action**: `BLOCK`
- **Threat Category**: Credential Stuffing / Automated Bots / Client Integrity Validation
- **Description**: Applied specifically to sensitive login endpoints on domain `dimensionash26.secureedge.xyz` to enforce client-side cookie validation and block suspected automated, headless, or illegitimate bot traffic attempting login access.

### Condition Logic
```yaml
rule_id: RULE-004
domain: "dimensionash26.secureedge.xyz"
action: BLOCK
conditions:
  operator: ALL
  rules:
    - type: path_equals
      value: "/login"
    - type: signal_exists
      operator: ANY
      signals:
        - "Suspected Bot"
        - "Suspected Bad Bot Headless"
        - "Client-Side Cookie Valid"
```
