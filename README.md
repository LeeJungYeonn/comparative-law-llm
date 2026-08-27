# comparative-law-llm

Korean/U.S. legal-language corpus의 frozen v4.0.2를 이용해 legal lexicon과
confirmatory measurement를 개발·검증하는 프로젝트입니다.

## Current code

- `scripts/lexicon/`: lexicon 구축, 검증, confirmatory 및 sensitivity 분석
- `scripts/build_posthoc_measurement_validation_v1.py`: post-hoc measurement 검증
- `legal_lexicon/legal_lexicon_v1_0/`: 현재 frozen lexicon
- `tests/`: 현재 코드의 테스트

현재 분석 입력은 `outputs_v2/v4.0.2/`의 frozen corpus와 manifest입니다.

```powershell
& .venv\Scripts\python.exe -m pytest -q
& .venv\Scripts\python.exe scripts\lexicon\run_confirmatory_reference_v1.py
& .venv\Scripts\python.exe scripts\lexicon\run_occurrence_sensitivity_v1.py
```

## Legacy

더 이상 현행 분석에서 호출하지 않는 코드는 `legacy/`에 보관합니다.

- `legacy/corpus_v2_v4/`: v2~v4 코퍼스 구축·수정·동결 파이프라인
- `legacy/legal_lexicon_history/`: 참조가 끊긴 초기 lexicon 버전
- 그 밖의 `legacy/` 항목: 이전 pilot, Stage 1/2 및 실험 코드와 산출물

Frozen corpus 산출물은 현행 분석의 입력이므로 `outputs_v2/`에 유지합니다.
