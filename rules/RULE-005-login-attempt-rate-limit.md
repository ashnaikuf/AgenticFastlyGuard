## Rule: RULE-005 - Login Attempt Rate Limit

### Metadata & Overview
- **Rule ID**: `RULE-005`
- **Rule Name**: Login Attempt 10 Request
- **Rule Type**: Rate Limit
- **Target Domain**: Global / All Domains
- **Action**: `BLOCK`
- **Threat Category**: Credential Stuffing / Brute Force / Login Abuse
- **Description**: Rate limits repeated login attempts against `/login` endpoints. Counts POST requests carrying the Login Discovery signal and, once a client exceeds 10 counting signals within 1 minute, adds the `LoginAttempt10Request` signal and blocks all requests from that client for 5 minutes. Used to defend credential stuffing and password brute-force campaigns that stay under per-request attack detection thresholds.

### Condition Logic
```yaml
rule_id: RULE-005
type: rate_limit
action: BLOCK
conditions:
  operator: ALL
  rules:
    - type: path_contains
      values:
        - "/login"
    - type: method_equals
      value: "POST"
    - type: signal_exists
      operator: ANY
      signals:
        - "Login Discovery"
        - "LOGINDISCOVERY"
rate_limit:
  signal_added: "LoginAttempt10Request"
  threshold: 10
  interval: 1m
  action_duration: 5m
  action: BLOCK_ALL_REQUESTS
```
