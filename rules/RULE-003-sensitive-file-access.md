## Rule: RULE-003 - Sensitive Information File Access Defense

### Metadata & Overview
- **Rule ID**: `RULE-003`
- **Rule Name**: Sensitive Information File Access Defense
- **Target Domain**: Global / All Domains
- **Action**: `BLOCK`
- **Threat Category**: Reconnaissance / Information Disclosure / Path Traversal
- **Description**: Inspects incoming request URI paths for direct access attempts targeting sensitive configuration files, system credentials, or unauthorized user media paths (e.g., .env files, /etc/passwd, user profile image uploads).

### Condition Logic
```yaml
rule_id: RULE-003
action: BLOCK
conditions:
  operator: ANY
  rules:
    - type: path_contains
      values:
        - "/.env"
        - "/etc/passwd"
        - "/profile/image/url"
```
