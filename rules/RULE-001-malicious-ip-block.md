## Rule: RULE-001 - Malicious and SigSci IP Block

### Metadata & Overview
- **Rule ID**: `RULE-001`
- **Rule Name**: Malicious and SigSci IP Block
- **Target Domain**: Global / All Domains
- **Action**: `BLOCK`
- **Threat Category**: Malicious Traffic / Reputation / Bot Mitigation
- **Description**: Evaluates inbound requests and automatically blocks traffic if any known high-risk threat signal or malicious IP signal is present. Used when dealing with requests originating from flagged botnets or malicious actors.

### Condition Logic
```yaml
rule_id: RULE-001
action: BLOCK
conditions:
  operator: ALL
  rules:
    - type: signal_exists
      operator: ANY
      signals:
        - "Malicious IP"
        - "SigSci IP"
        - "Suspected Bad Bot"
```
