# Tuning review checklist

Use only the 10 cases in `tuning_case_manifest.csv`.

For each matcher hit:
- TP / FP
- category correct?
- concept/concept-family correct?
- overly broad literal/regex?
- overlap problem?

For false negatives:
- inspect all zero-hit documents;
- sample no-hit sentences/paragraphs in every tuning case;
- record genuine legal concepts in doctrine/remedy/procedure/party_arg that the matcher missed.

Allowed after tuning:
- add/remove/demote surface forms;
- modify regex;
- correct category/concept mapping;
- modify overlap handling if justified.

Every change must be entered in `tuning_revision_log.csv`.

Forbidden:
- validation evidence;
- confirmatory evidence.

When tuning is finished:
1. create a pre-validation candidate;
2. record its SHA256 before validation;
3. run validation 10 once;
4. never modify based on validation results;
5. freeze the unchanged candidate as v1.0.
