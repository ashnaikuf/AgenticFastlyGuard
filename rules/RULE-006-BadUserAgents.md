## Rule: RULE-006 - Bad User Agents Block

### Metadata & Overview
- **Rule ID**: `RULE-006`
- **Rule Name**: Bad User Agents
- **Target Domain**: Global / All Domains
- **Action**: `BLOCK`
- **Threat Category**: Automated Tools / Reconnaissance / Bot Mitigation
- **Description**: Evaluates inbound request headers and automatically blocks traffic if the User-Agent header matches known automated scanners, command-line utilities, or attack tooling such as curl, python, nmap, or wpscan.

### Condition Logic
```yaml
rule_id: RULE-006
action: BLOCK
conditions:
  operator: ANY
  rules:
    - type: signal_exists
      operator: ANY
      signals:
        - "ATTACK-TOOLING"
        - "NOUA"
        - "SUSPECTED-BAD-BOT"
    - type: user_agent_like
      match_type: wildcard
      value: "*curl*"
    - type: user_agent_equals
      values:
        - "python"
        - "nmap"
        - "wpscan"
```        