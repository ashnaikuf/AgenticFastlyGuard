# AgenticFastlyGuard
Abstract & Overview
Web Application Firewalls (WAFs) log millions of web requests daily, but turning raw telemetry into actionable mitigation is traditionally slow and manual. Security engineers spend hours parsing giant log dumps to construct custom blocking rules by hand—often long after an attack has caused impact.

To solve this, I developed an open-source Python tool that extracts raw security logs from Fastly’s Next-Gen WAF (NGWAF) and uses machine learning to cluster suspicious behavioral patterns (request timing, header ordering, and TLS/HTTP fingerprints). Once identified, the tool automatically synthesizes and deploys targeted WAF blocking rules in real time.

What You Will Learn in This Video
The Manual Bottleneck: Why traditional manual log analysis fails against modern automated threats and distributed botnets.

Feature Extraction & Clustering: How unsupervised ML groups malicious traffic based on behavioral signatures rather than static IPs or simple User-Agents.

Automated Rule Generation: Automatically pushing mitigation rules into Fastly NGWAF without human lag.

Live Demonstration: Scanning gigabytes of log telemetry, uncovering a stealthy botnet, and deploying an active defense rule in under two minutes.

Resources & Source Code
GitHub Repository: https://github.com/ashnaikuf/AgenticFastlyGuard 

Event: BSidesPDX 2026 | Portland State University (Smith Memorial Student Union, Portland, OR)
