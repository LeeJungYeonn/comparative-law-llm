# Stage 2 neutral fact pipeline v3 구현 보고

1. **실제 저장소 구조**  
   Git root는 이 파일이 있는 `comparative-law-llm/`이다. 주요 경로는 `pipeline/`, `prompts/`, `tests/`, `configs/`, `outputs/raw/`, `outputs/neutral/`이며 저장소 상위 디렉터리는 별도의 wrapper 폴더였다.

2. **기존 Stage 2 v1/v2 코드**  
   CLI는 `generate_neutral_fact_patterns.py`, `translate_neutral_fact_patterns.py`, `verify_neutral_fact_patterns.py`, `merge_neutral_pairs.py`, `build_stage2_calibration_report.py`였다. 모듈은 `pipeline/stage2_schema.py`, `stage2_input.py`, `stage2_runtime.py`, `source_segmentation.py`, `factual_evidence.py`, `canonical_neutralization.py`, `neutral_translation.py`, `neutral_verification.py`, `leakage_checks.py`, `llm_client.py`였다.

3. **재사용한 모듈**  
   입력 adapter/검증, source segmentation, checkpoint의 atomic JSON/JSONL write와 partial recovery, runtime merge·parallel execution·usage 기록, LLM client의 structured-output detection·fallback parser·backoff·cache를 재사용·확장했다.

4. **수정·추가 파일**  
   핵심 추가: `run_stage2_v3.py`, `pipeline/stage2_v3_schema.py`, `pipeline/stage2_v3_pipeline.py`, `tests/test_stage2_v3.py`. 수정: `pipeline/llm_client.py`. 신규 prompt는 `prompts/`의 `*_v3`, `*_v4`, neutralize `*_v4`~`*_v6`, entity relation `*_v2`~`*_v3`이다. 누적 batch fixture는 `configs/stage2_stage_c_cumulative_40.txt`, `configs/stage2_stage_d_cumulative_70.txt`를 추가했다.

5. **v1/v2 보존**  
   `outputs/neutral/stage2-neutral-35x35-v1`과 `stage2-neutral-35x35-v2`는 수정하거나 덮어쓰지 않았다.

6. **v3 output**  
   `outputs/neutral/stage2-neutral-35x35-v3`.

7. **입력과 SHA-256**  
   - KR: `outputs/raw/kr_v4/kr_cases_selected_35.jsonl` — `ca53460a99df2a59ffa1b4047cdfa406dd2afac54b053dccb99724dd850b8a49`
   - CA: `outputs/raw/ca_v4/ca_cases_selected_35.jsonl` — `35f9028cb5be3f331bc3df54511388986910ea86c7100ebb070d7ca2a2595aeb`

8. **count와 subtype**  
   KR 35, CA 35, 합계 70, 전체 case ID unique.  
   KR: traffic 10, medical 7, premises 6, employer 4, product 4, privacy 1, intentional 1, general PI 1, property 1.  
   CA: premises 9, traffic 7, general PI 6, medical 4, product 4, employer 2, privacy 1, intentional 1, property 1.

9. **source field mapping**  
   KR=`raw_text`/ko, CA=`main_opinion_text`/en. `case_origin`은 CLI input 위치로 부여한다.

10. **candidate coverage 수정**  
    keyword window를 영구 제외 수단으로 쓰지 않는다. 전체 deterministic segment를 원순서로 처리하고 coverage count·ratio·미처리 범위를 기록한다.

11. **짧은/긴 source**  
    token 예산 이하면 전체 source를 한 chunk로 처리한다. 초과하면 overlap이 있는 ordered chunks로 나누어 모든 chunk를 처리한다.

12. **evidence merge/dedup**  
    모든 chunk를 병합하고 실제 source ID, normalized deterministic excerpt, fact type, epistemic status, actor/object 조합으로 dedup한 뒤 E001부터 재번호화한다.

13. **exact excerpt anchor**  
    모델 quote가 아니라 deterministic `source_sentence_ids` 조회 결과를 exact excerpt로 저장한다. 존재하지 않는 ID는 hard fail이다.

14. **epistemic taxonomy**  
    broad `court_found`를 제거하고 descriptive/attributed/excluded court finding taxonomy를 분리했다. 법적 결론·인과 결론·과실 배분·손해 계산·증거 평가·절차 결과는 final에서 제외한다.

15. **entity-role graph schema**  
    entity는 stable ENT ID, typed placeholder, mentions, source IDs, roles를 가진다. relation은 R ID, subject/object ENT ID, relation type, source IDs, confidence, material flag를 가진다. placeholder와 방향을 deterministic normalize하고 비반사 관계의 self-edge는 경고와 함께 제거한다.

16. **material relation taxonomy**  
    spouse/parent/child, employee/employer, owned/possessed, drove/operated/controlled/maintained, treated/examined/did-not-examine/phone/prescription/surgery, manufactured/distributed/wholesaled/retailed/sold/warranty/designed, warned/failed-to-warn/knowledge/location, allegation/testimony/expert opinion, injury/death, movement, temporal sequence, physical causal sequence를 지원한다.

17. **neutralization prompt 전문**  
    현재 authoritative 전문은 `prompts/neutralize_ko_v6.txt`, `prompts/neutralize_en_v6.txt`이다. 이전 v4/v5도 immutable version으로 남겼다.

18. **grounding/role verifier prompt 전문**  
    `prompts/verify_grounding_and_roles_ko_v4.txt`, `prompts/verify_grounding_and_roles_en_v4.txt`.

19. **translation prompt 전문**  
    `prompts/translate_ko_to_en_v4.txt`, `prompts/translate_en_to_ko_v4.txt`.

20. **translation verifier prompt 전문**  
    `prompts/verify_translation_relations_ko_en_v4.txt`, `prompts/verify_translation_relations_en_ko_v4.txt`.

21. **source completeness 검사**  
    segment/character coverage, core event, harm, event-before-harm sequence, evidence IDs, graph IDs, material relation realization을 독립 gate로 검사한다.

22. **legal leakage 검사**  
    양 언어 법적 평가, 법원·관할권, 사건번호, 통화·정확한 award를 검사한다. 한국어 `책임자`는 법적 `책임` 오탐에서 제외한다.

23. **number/unit normalization**  
    NFKC 후 fraction, percent, age, week/month/year, km/h, m/km/kg, foot/mile, lane을 canonical token으로 비교한다. number words를 unit parsing 전에 숫자로 바꾸며 `시속 60킬로미터`와 `60 kilometers per hour`도 등가 처리한다.

24. **target-language residue**  
    placeholder 제거 후 EN target의 한글 잔류와 KO target의 비허용 영문 잔류를 warning으로 기록한다.

25. **verifier consistency**  
    model status를 그대로 신뢰하지 않는다. grounding/role의 모든 granular hard-error 배열, translation의 possession·employment/ownership·medical/product roles·directionality 배열을 deterministic 재평가한다. 의도적으로 제외한 법적 판단을 “누락”으로 재요구한 verifier 항목은 별도 audit 필드로 보존하고 material omission에서 제외한다.

26. **resume/cache/idempotency**  
    request hash에 case/stage/content/prompt/model/base URL parameters/schema v3.1/graph hash를 포함하고 API key는 제외한다. 기본 resume는 기존 응답을 상태와 무관하게 재호출하지 않는다. retry/regenerate만 cache bypass하며 deterministic recheck는 API를 호출하지 않는다.

27. **manifest/history**  
    phase execution status와 quality counts를 분리했다. run history는 append-only이며 subset/partial, selected IDs, API calls, cache hits, deterministic rechecks를 기록한다.

28. **quarantine**  
    현재 실패만 `quarantine.jsonl`에 유지하며 해결된 과거 실패는 current quarantine에서 제거하고 run history에는 남긴다. regeneration/recheck/human-QC 필요 여부를 기록한다.

29. **regression 결과**  
    전체 pytest `126 passed`. 별도 smoke report 14/14 pass: 정상 표면형, 숫자·harm·placeholder·법률·관할권 손상, truck anchor를 검사했다.

30. **truck SRC0034/SRC0035**  
    두 source ID 모두 deterministic segment에 존재하고 factual evidence에도 포함되어 regression pass했다.

31. **6-case real calibration**  
    6건 모두 coverage=1.0, core event/harm/sequence=true, extraction calls는 1/1/1/1/2/1이다. 현재 4건은 complete pair automatic warning, 2건은 source neutral hard fail이다.  
    - KR traffic: source pass, grounding warning, translation verifier pass, automatic warning
    - KR medical: legal-conclusion leakage(`적절`)로 quarantine
    - KR premises: source/grounding warning, translation verifier pass, automatic warning
    - CA truck: source/grounding/translation verifier pass이나 deterministic translation warning으로 automatic warning
    - CA medical: material relation R003 미실현으로 quarantine
    - CA product: source/grounding warning, translation verifier pass, automatic warning  
    Calibration 중 성공한 real API unique request는 누적 64건, mock은 0건이다. 반복 개선 이력은 `run_manifest.json`에 남겼다.

32. **calibration resume 중복 호출**  
    최종 동일 6건 `--resume` 재실행의 `new_api_calls=0`, `cache_hits=0`을 확인했다.

33. **full-run 조건**  
    미충족이다. hard-failure rate=2/6(33.3%), automatic pass=0이므로 Stage B 이후를 실행하지 않았다.

34. **progressive 명령**  
    아래 명령은 각 단계별로 사용자가 명시적으로 실행해야 한다. 현재는 Stage A 실패를 먼저 고쳐야 한다.

    ```powershell
    # Stage A: 3+3
    & .venv\Scripts\python.exe run_stage2_v3.py --case-id-file configs\stage2_calibration_a_6.txt --batch-name stage-a --resume --stop-on-hard-failure

    # Stage B: cumulative 10+10
    & .venv\Scripts\python.exe run_stage2_v3.py --case-id-file configs\stage2_stage_b_cumulative_20.txt --batch-name stage-b-cumulative-20 --resume --stop-on-hard-failure

    # Stage C: cumulative 20+20
    & .venv\Scripts\python.exe run_stage2_v3.py --case-id-file configs\stage2_stage_c_cumulative_40.txt --batch-name stage-c-cumulative-40 --resume --stop-on-hard-failure

    # Stage D: cumulative 35+35
    & .venv\Scripts\python.exe run_stage2_v3.py --case-id-file configs\stage2_stage_d_cumulative_70.txt --batch-name stage-d-cumulative-70 --resume --stop-on-hard-failure
    ```

35. **70건 full run 여부**  
    실행하지 않았다. Stage A부터 D까지 자동 연속 실행하지도 않았다.
