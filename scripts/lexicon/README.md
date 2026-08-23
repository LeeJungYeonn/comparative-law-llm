# Legal-lexicon pipeline status

The active development entry point is `build_nested_discovery_v1.py`. It creates the
metadata-only 20/10/10 nested split and processes opinion text from discovery 20 only.

`build_dev_candidates.py` is retained solely for historical reproducibility and is marked
`DEPRECATED_METHOD_OVERLAPPING_DEV_VALIDATION`. Its all-40-development output must not be
used as evidence by the nested v1 workflow.

The nested v1 run intentionally stops after discovery candidate generation. Tuning text
must not be accessed until the discovery review table has been completed by a human.
