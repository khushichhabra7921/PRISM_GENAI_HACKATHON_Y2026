# Data

## Official starter kit (this folder)

| File | Contents |
|---|---|
| `input.txt` | 20 user complaints, one per line (some are numbered lists of quoted parts) |
| `siis_responses.json` | the raw SIIS response for each complaint, in the same order: `{"id", "original_query", "siis_response": {"title", "content"}}`. `siis_response` is the payload the API accepts in `POST /v1/troubleshoot`. 20 rows, 11 distinct articles |
| `sample_output.json` | the reference response shape: `{"query", "response": {"contexts": [...]}}` |
| `schema.py` | the Pydantic response contract (byte-identical to the official file); `app/validate.py` imports it |

Only `app/data_loader.py` knows these raw formats; `app/siis_text.py` turns a `{title, content}` payload into a
clean, sectioned article (product-category prefix, markdown headings, step-per-line lists, garbled lines).

## Deeplink catalog (`deeplinks.json`): synthetic

The official kit ships **no deeplink catalog**. The only real deeplink pair it reveals is the
*Back up data (TechCorp Cloud)* entry in `sample_output.json` (`voiceassist://masked/act/b3ed3ed663` with its
validation deeplink `voiceassist://masked/val/266037d0c5`); it is copied into the catalog unchanged.
The other 153 entries are synthetic Settings screens (written from general knowledge of Android Settings menus),
in the official URI scheme `voiceassist://masked/act/<10 hex>`; toggles and sliders carry a
`validation` deeplink (`voiceassist://masked/val/<10 hex>`, key, result type). `originalType` is `onURL`, as in
the official entry. Regenerate with `python scripts/make_synthetic_kit.py`.

If an official catalog becomes available, drop it in as `deeplinks.json` (field aliases in
`app/data_loader.py`) and re-run `python bench/official.py`.

## Synthetic regression kit (`synthetic/`)

The official kit has no labelled queries, held-out paraphrases or gold plans, so the seeded synthetic kit that the
prototype was built on is kept for regression tests and the cache / latency / ablation benchmarks
(`bench/run.py`). See `synthetic/README.md`.
