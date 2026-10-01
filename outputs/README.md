# Official-kit outputs

Generated 2026-10-01T13:44:06 by `bench/official.py` (62.9 s) · LLM routing: `none`, extraction `rules`, enrichment `rules`.

One file per `data/input.txt` complaint, each the response to `POST /v1/troubleshoot` with that row's `siis_response` payload, in exactly the `data/sample_output.json` shape (`?view=contract`).

| Metric | Value |
|---|---|
| Complaints / distinct SIIS articles | 20 / 11 |
| Schema-valid (`data/schema.py`) | 100.0% |
| All 11 rule checks pass | 100.0% |
| Plans / fallbacks | 20 / 0 |
| Steps grounded in the payload (fuzzy >= 80) | 100.0% of 308 |
| Actions (auto / manual / critical) | 78 (auto 6, manual 56, critical 16) |
| Actionable deeplinks (dummy_positive) / validation deeplinks | 9 (0) / 0 |
| Cold latency P50 / P95 | 336.9 / 1511.4 ms |
| Repeat (payload-scoped cache) hits, P95 | 20/20, 27.5 ms |
| Other complaint, same article: cache hit / miss | 4 / 5 |
| Goal score mean (min-max) | 0.61 (0.5-0.75) |

| Row | SIIS article | Goal | Actions | Steps | Deeplinks | Score |
|---|---|---|---|---|---|---|
| [row_1](row_1.json) | Email server not responding on smartphone or tablet | Email Server Connection Troubleshooting | 8 | 30 | 3 | 0.58 |
| [row_2](row_2.json) | Blank or black display on a smartphone or tablet | Blank Display Troubleshooting | 4 | 16 | 0 | 0.58 |
| [row_3](row_3.json) | Some things to check first | Blank Screen Troubleshooting | 3 | 7 | 0 | 0.55 |
| [row_4](row_4.json) | Blank or black display on a smartphone or tablet | Blank Display Troubleshooting | 4 | 16 | 0 | 0.74 |
| [row_5](row_5.json) | Transfer Secure folder with Data Transfer | Secure Folder Transfer Troubleshooting | 3 | 14 | 0 | 0.72 |
| [row_7](row_7.json) | Use Multi window and App pairs on your smartphone or tablet | Multi Window Troubleshooting | 6 | 40 | 1 | 0.58 |
| [row_8](row_8.json) | Screen mirroring to your TechCorp TV | Screen Mirroring Troubleshooting | 1 | 5 | 0 | 0.56 |
| [row_9](row_9.json) | Access your smartphone's data if the screen does not respond | Data Access Troubleshooting | 2 | 14 | 0 | 0.52 |
| [row_10](row_10.json) | Screen flickers when using the Camera on a smartphone | Screen Flicker Troubleshooting | 2 | 6 | 0 | 0.5 |
| [row_11](row_11.json) | Some things to check first | Blank Screen Troubleshooting | 2 | 3 | 0 | 0.53 |
| [row_12](row_12.json) | Use Multi window and App pairs on your smartphone or tablet | Multi Window Troubleshooting | 3 | 13 | 0 | 0.6 |
| [row_13](row_13.json) | Blank or black display on a smartphone or tablet | Blank Display Troubleshooting | 4 | 16 | 0 | 0.7 |
| [row_14](row_14.json) | Cracked or bleeding screen on smartphone or tablet | Cracked Screen Troubleshooting | 2 | 3 | 0 | 0.73 |
| [row_15](row_15.json) | Blank or black display on a smartphone or tablet | Blank Display Troubleshooting | 4 | 16 | 0 | 0.72 |
| [row_16](row_16.json) | Blank or black display on a smartphone or tablet | Blank Display Troubleshooting | 4 | 16 | 0 | 0.59 |
| [row_17](row_17.json) | Some things to check first | Blank Screen Troubleshooting | 3 | 7 | 0 | 0.53 |
| [row_19](row_19.json) | Cracked or bleeding screen on smartphone or tablet | Cracked Screen Troubleshooting | 2 | 3 | 0 | 0.57 |
| [row_20](row_20.json) | Screen does not rotate on smartphone or tablet | Screen Rotation Troubleshooting | 6 | 25 | 1 | 0.6 |
| [row_21](row_21.json) | Touchscreen issues on a smartphone or tablet | Touchscreen Issues Troubleshooting | 11 | 42 | 4 | 0.55 |
| [row_22](row_22.json) | Blank or black display on a smartphone or tablet | Blank Display Troubleshooting | 4 | 16 | 0 | 0.75 |

## Reference sample

`data/sample_output.json` passes every check (0 violations). Its query has no SIIS payload in the kit, so [sample_query.json](sample_query.json) is our free-text answer (retrieval over the official articles): Follow these steps to perform this Cracked Screen Troubleshooting; step accuracy vs the reference 0.0/3, deeplink relevance 1.0/2. sample_output.json ships without its SIIS payload, so this plan is retrieved from the 11 official articles; its back-up step comes from an article that is not in the kit.

The deeplink catalog (`data/deeplinks.json`) is synthetic except the Back up data (TechCorp Cloud) entry copied from `data/sample_output.json`: the official kit ships no catalog.
