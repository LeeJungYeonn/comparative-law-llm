# Revised corpus v2.2 implementation report

Run date: 2026-08-10 (Asia/Seoul)

## Scope of the revision

This revision starts from commit `7a6e804bea471247bbecc5f110deadd03508668b`.
The legacy Korean/California pilot remains unchanged. Stable hashing, atomic
outputs, controlling-opinion selection, deterministic leakage checks, and
request checkpoint concepts are retained; source and eligibility assumptions
that caused the prior shortfall are replaced.

## Letsur configuration

The v2 runtime now matches the successful legacy configuration:

- gateway: `https://gw.letsur.ai/v1/chat/completions`;
- bearer credential name: `LETSUR_API_KEY`;
- `.env` loading through `python-dotenv` without environment override;
- default previously used model identifier: `gpt-5.6-luna`;
- JSON Schema, JSON-object, then JSON-only compatibility modes.

The credential value is never emitted or persisted. A one-request connectivity
smoke passed using JSON Schema. Raw response, request ID, returned model ID,
usage, hashes, and gateway identifier were retained without the credential.

## Korean structured source

The strict collector now uses `legalize-kr/precedent-kr`, a Git Markdown corpus
derived from the Korean National Law Information Center API. The pinned sparse
snapshot is `d4a9982a272518e83312184c584f6a3542c9ce23`. Eligibility requires the
structured `사건종류=민사`, `법원명=대법원`, `법원등급=대법원`, an in-window
`선고일자`, and usable `판례내용`. LBox is not used to infer court or date.

For 2000-01-01 through 2025-12-31 the full structured path contained 11,807
records. Broad retrieval retained 6,030 candidates; 5,834 had adequate text,
4,145 satisfied all four mandatory fact dimensions, and 1,955 passed every
deterministic strict-source gate. Strict primary-domain counts were:

| Primary domain | Count |
|---|---:|
| general negligence / personal injury | 607 |
| medical / professional liability | 428 |
| product liability | 12 |
| other civil liability | 908 |

Thus the Korean source no longer creates a 100-case source shortage in the
primary window. The window has not been extended.

## Revised rules

Fact sufficiency now requires parties/relationships, concrete conduct or
omission, harm, and a minimally reconstructable causal/event sequence.
Context/location, detailed chronology, and defense facts are optional; 5/7 is
preferred. Lower-court lookup is attempted only to rescue a missing mandatory
dimension, using exact identifiers only.

The primary taxonomy is general, medical/professional, product, and other civil
liability. Vicarious liability, respondeat superior, negligent supervision,
premises liability, wrongful death, comparative fault, intentional misconduct,
and punitive/multiple damages are nonexclusive secondary tags.

Final U.S. state totals may vary from 10 through 30. A lower-bound/upper-bound
circulation enforces five-state representation and one identical KR/U.S.
primary-domain allocation. The allocation is derived after collection rather
than hard-coded.

## Validation status

The focused v2 suite currently contains 23 passing tests, including structured
Korean Markdown parsing, mandatory-four fact sufficiency, secondary employer
tags, full court-type checks, source grounding, bilingual invariants, flexible
state bounds, and final 100+100 invariants on a sufficient synthetic pool.

The all-32-shard U.S. run scanned 544,841 in-window `court_type=S` metadata
rows. The five preserved states produced 592 broad opinion candidates, 575
adequate controlling opinions, 376 core-fact-sufficient records, and 164 strict
records (163 after family deduplication). Strict domains were general 65,
medical 79, product 14, and other 5. The preserved states remained feasible.

Before bulk calls, `pre_llm_feasibility.json` confirmed KR 960 and US 163
strict deduplicated candidates and a feasible 100+100 bounded flow. The 3+3
Letsur smoke then passed extraction, grounding, bilingual alignment, leakage,
and mandatory-four QC for all six cases.

The frozen final allocation is identical in both countries: general 55,
medical 28, product 12, and other 5. U.S. state totals are Pennsylvania 27,
Michigan 24, Louisiana 19, Nevada 16, and West Virginia 14. Bulk extraction,
translation, deterministic QC, and warning-routed secondary Letsur semantic QC
all reached 200/200 final eligibility. Twenty-five records retain a transparent
currency/unit normalization review flag, but all passed semantic QC and the
final hard checks. `collection_summary.json` has status `complete`; every
court, date, family, bilingual, QC, state, domain, and 20/80 split invariant is
true. The 2000-2025 window was not extended.
