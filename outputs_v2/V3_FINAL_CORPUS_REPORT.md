# V3 Final Corpus Report

- Corpus: `kr-us-highcourt-corpus-v3.0`
- Status: **frozen**
- Seed: `20260810`

## A. Starting review state

- Text corrections: 73
- Originally source-ineligible: 18
- Initially retainable: 182

## B. Source replacement

- Original audit-flagged: 18
- Additional source-ineligible: 37
- Additional balancing swaps: 2
- Final replacement set: 57

| Old case | New case | Country | Old state | New state | Category | Reason |
|---|---|---|---|---|---|---|
| KR_1490ae2a5caa81b4c7 | KR_0f2fb4048fdd1b2ba6 | KR |  |  | additional_source_ineligible | The opinion concerns contractual nonperformance and calculation of expectation damages under a construction agreement, making it a contract-only payment/damages dispute rather than a qualifying tort-liability domain. |
| KR_166fe3c2ec192a031c | KR_3e400c529b48b85317 | KR |  |  | additional_source_ineligible | The opinion concerns inheritance of a credit-guarantee reimbursement debt and the timeliness/effect of limited acceptance of inheritance, not a civil-liability damages merits issue. |
| KR_16747477bbb419c699 | KR_5be8caa54d55da9c2d | KR |  |  | additional_source_ineligible | 보험계약상 보험금 지급 및 소송대행의무 불이행에 따른 계약상 손해배상만을 다루는 사건으로, 보험-커버리지 및 계약상 지급 분쟁에 해당하며 독립적인 민사책임의 실체적 본안 쟁점이 아니다. |
| KR_17f99b470b67dcf2bd | KR_6623c47ec5cdef48bc | KR |  |  | additional_balancing_swap | minimum balancing swap required for exact KR-US domain equality |
| KR_2e57afc9436b5c385e | KR_872d527b52dfbd5927 | KR |  |  | additional_source_ineligible | 판결의 실질적 쟁점은 산업재해보상보험 유족급여·장의비에 관한 대위 및 합의서 해석이며, 민사상 손해배상책임의 성립이나 범위를 판단한 본안이 아니다. 특히 원심의 판단과 대법원의 파기 이유는 ‘처분문서인 이 사건 합의서의 문언’ 해석에 관한 것이다. |
| KR_55074ea924a215c67b | KR_882ce243b36c889fda | KR |  |  | additional_source_ineligible | Insurance-coverage-only dispute concerning application of an automobile-insurance exclusion clause, not the underlying civil-liability merits. |
| KR_655f2386b561c2a84a | KR_9dad33b6d6293b40f2 | KR |  |  | additional_source_ineligible | 계약상 품위유지약정 위반에 따른 채무불이행 및 손해배상만을 다루는 계약 전용 사건으로, 일반 불법행위·의료과실·제품책임의 실질적 쟁점이 없다. |
| KR_77616f4d33f49f160f | KR_a41002435a915c75d1 | KR |  |  | additional_source_ineligible | The opinion addresses only the statute-of-limitations accrual and interruption rules for a damages claim, without deciding a substantive civil-liability merits issue. |
| KR_8272e1d4abaf8ec55f | KR_ab2b68c40724d0aa92 | KR |  |  | additional_balancing_swap | minimum balancing swap required for exact KR-US domain equality |
| KR_8d7ca5d24afd11d1da | KR_adf561060b125d6b87 | KR |  |  | additional_source_ineligible | 계약상 상품권 제품 제공의무 불이행에 따른 전보배상만을 다루는 계약·지급 사건으로, 실질적인 불법행위상 민사책임이나 손해배상 merits 쟁점이 아니다. |
| KR_92b1ce3452f5861ba4 | KR_bcc14552a5ebdd0945 | KR |  |  | additional_source_ineligible | 소송의 판단 대상이 불법행위 손해배상책임의 성립이나 손해액 등 실체적 책임 merits가 아니라 소멸시효의 기산점 및 채무 승인에 따른 소멸시효 중단의 효력에 한정되어 있다. |
| KR_973642690caa8c1765 | KR_c2567a4fb9e71fc1f7 | KR |  |  | additional_source_ineligible | 실질적으로 대리사무계약상 자금집행순서와 계약금·선급금 지급에 관한 계약상 지급 분쟁이므로 contract-only payment 사건에 해당한다. |
| KR_9d2e86f030a6ea99c5 | KR_c93d64dedb05abdc5b | KR |  |  | additional_source_ineligible | 보험자의 책임보험금 지급한도와 대위·변제액 공제만을 다룬 보험·구상금 쟁송으로, underlying civil-liability merits를 판단하지 않는다. |
| KR_cabb12a8960b273308 | KR_d7658dab9202d2e025 | KR |  |  | additional_source_ineligible | The opinion concerns an insurer’s recourse claim against the workers’ compensation agency for payment of industrial accident benefits, rather than a substantive civil-liability damages merits issue. It is excluded as a workers’ compensation benefits and insurance-related reimbursement dispute. |
| KR_d5131e7c37df509266 | KR_e841c27e080fa0659c | KR |  |  | additional_source_ineligible | 보험자 상호 간의 중복보험 구상관계와 소멸시효만을 판단한 사건으로, 민사책임·손해배상 성립의 실체적 merits가 없고 보험 관련 청구 및 기간 쟁점에 해당한다. |
| KR_fbac05e4c9ae195c2d | KR_fdfefa335ed21439c0 | KR |  |  | additional_source_ineligible | 판결의 중심 쟁점은 교통사고 손해배상책임 자체가 아니라 의무보험금 산정 및 보험자의 초과지급금 반환·공제 여부인 보험·금액 문제이다. |
| US_02cd16f932838519ba | US_143504a555ba2b1572 | US | Louisiana | Louisiana | additional_source_ineligible | The opinion principally resolves prescription and does not adjudicate a substantive civil-liability merits issue; the discussion of negligence, product liability, Jones Act status, and maritime jurisdiction concerns alleged claims and vacated procedural findings. |
| US_162eb334cc7d5a8b91 | US_7f556627654c5d3f9a | US | Louisiana | Louisiana | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_18fd4288d7f32131ce | US_9acb209b5020b6dec4 | US | Louisiana | Louisiana | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_2e67d6775a92caddee | US_b24ef2018816af9946 | US | Louisiana | Louisiana | additional_source_ineligible | The controlling opinion resolves a nullity action concerning validity of service of process and appellate review, not a civil-liability or damages substantive-merits issue. The personal-injury action and damages award are only underlying procedural context. |
| US_39be177337afd8d1be | US_b2c7f8c81dc1b0c361 | US | Louisiana | Louisiana | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_5fb1681d1e52f4e709 | US_e05b7f2567e5800f56 | US | Louisiana | Louisiana | additional_source_ineligible | The controlling opinion resolves a nullity action concerning alleged fraud, ill practice, spoliation, diligence, and summary judgment—not a substantive civil-liability or damages merits issue. |
| US_a072ff18b300a13dfd | US_1aba36d98e0653f066 | US | Louisiana | Michigan | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_a83924c6ae07314b33 | US_1cfe591dc59be857da | US | Louisiana | Michigan | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_b1a79acce1af757b58 | US_60f81ce937468b4218 | US | Louisiana | Michigan | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_bf6cccdd9b8a33c4a5 | US_8636c5a0c5ede55860 | US | Louisiana | Michigan | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_e1da2d37a9dee8ec7b | US_99f0f4ac29a4da2463 | US | Louisiana | Michigan | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_00f7c6571f18d3c66e | US_9b0dcb923fdb190dd3 | US | Michigan | Michigan | additional_source_ineligible | The controlling opinion resolves a statutory notice and filing issue rather than a civil-liability substantive-merits issue. |
| US_02bde060086590cc77 | US_b732d6870be7eac6fb | US | Michigan | Michigan | additional_source_ineligible | The opinion addresses only the timeliness of a medical-malpractice claim under the statutory discovery rule and does not decide a civil-liability substantive merits issue. |
| US_0abf417417190c2e89 | US_e1738150309971f404 | US | Michigan | Michigan | additional_source_ineligible | The supplied opinion is an order denying reconsideration and addresses collateral attack and collateral estoppel, not a substantive civil-liability merits determination. |
| US_36fd5aae9fa72a7fbb | US_f5ad9cb96e57eb50d6 | US | Michigan | Michigan | additional_source_ineligible | The controlling opinion resolves only statute-of-limitations and notice-tolling issues and does not adjudicate the substantive medical-malpractice liability merits. |
| US_624564663f55368864 | US_04cf5031ab373b2ef6 | US | Michigan | Nevada | additional_source_ineligible | The controlling opinion resolves service-of-process, waiver, and limitations issues; it does not adjudicate the underlying medical-negligence or damages merits. |
| US_861576769645010c57 | US_0b0596e3f457282cc2 | US | Michigan | Nevada | additional_source_ineligible | The controlling opinion concerns expert-witness qualification and evidentiary gatekeeping under MCL 600.2169(1), and resolves the cases through dismissal and directed verdict for failure to present a qualified expert, without deciding substantive negligence, causation, or damages. |
| US_9652590cf5c0086fbf | US_231c5e7fb867d9b8c2 | US | Michigan | Nevada | additional_source_ineligible | Limitations-only proceeding; the opinion does not decide a substantive civil-liability or damages merits issue. |
| US_c494ba6285904ce918 | US_512d2cb96959fe1764 | US | Michigan | Nevada | additional_source_ineligible | Contract-only dispute concerning the applicable limitations period; the opinion expressly characterizes the claim as breach of contract rather than a civil-liability tort merits issue. |
| US_f360178db4c0409aa2 | US_8f953fe2f1a7911efc | US | Michigan | Nevada | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_f42baf15f4f65533de | US_98be2374bed153f34a | US | Michigan | Nevada | additional_source_ineligible | Ineligible because the opinion resolves only a statutory notice/service requirement and does not decide a civil-liability substantive-merits issue; additionally, the source identifies a Court of Appeals panel rather than a controlling Michigan Supreme Court merits opinion. |
| US_f72de5a85fdd45cf64 | US_9dc742b4409b555931 | US | Michigan | Nevada | additional_source_ineligible | The opinion resolves the sufficiency of a medical-malpractice notice of intent and its tolling effect, not a civil-liability substantive-merits issue. |
| US_0bdc6cb112770284df | US_a07db3e0a97840c5b1 | US | Nevada | Nevada | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_1bd83c629182d24617 | US_061c987e5f6b8cf2e8 | US | Nevada | Pennsylvania | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_514a1afebc7c89ad3f | US_c387cc2f242da3c0dc | US | Nevada | Pennsylvania | additional_source_ineligible | The opinion addresses only the procedural effect of an erroneous ex parte jury instruction and the scope of retrial, not a substantive civil-liability merits issue. |
| US_9f4d3cca3bba16c059 | US_00df5ba88ee9d43fd5 | US | Nevada | West Virginia | additional_source_ineligible | The opinion concerns contract payment, breach-of-contract, mechanic's-lien, and jury-verdict issues, not a substantive civil-liability damages claim within the eligible domains. |
| US_af3a72d6a4fbf2544a | US_193fdb5b19590cbcab | US | Nevada | West Virginia | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_bfd06f8cb32e6978f3 | US_37fae4c5d51bff450a | US | Nevada | West Virginia | additional_source_ineligible | The supplied opinion is expressly from the Nevada Court of Appeals rather than the state-highest court, making it an unusable controlling opinion under the eligibility requirement. In addition, the appellate decision resolves only attorney-misconduct/new-trial findings and remands for further findings, rather than deciding a civil-liability substantive merits issue. |
| US_c2f1eaa3640f0a9bbf | US_43352148b1cf063f58 | US | Nevada | West Virginia | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_2b3ba71069d1e5eab9 | US_45c1de8fa5ac8eefab | US | Pennsylvania | West Virginia | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_57290d297c1a43a6d4 | US_4bf2e6511806b25f10 | US | Pennsylvania | West Virginia | additional_source_ineligible | The controlling issue is statute of limitations accrual and the discovery rule, without adjudication of the underlying negligence liability merits. |
| US_5e1df8e8cc45419467 | US_5334a4131f192eab74 | US | Pennsylvania | West Virginia | additional_source_ineligible | The opinion resolves a procedural discontinuance and judgment-of-non-pros issue, not a civil-liability or damages substantive-merits issue. |
| US_9cf64af04d2e95a72d | US_6ba935b9156e1996aa | US | Pennsylvania | West Virginia | additional_source_ineligible | The controlling opinion resolves statutory interpretation of DPW's Medicaid reimbursement lien and the effect of the statute of limitations, rather than a substantive merits determination of medical liability or damages. |
| US_f6a3b4ef121a4e3b04 | US_73b5710930ed47d628 | US | Pennsylvania | West Virginia | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_07482ca42c075096bb | US_8026310f8d6a3819bd | US | West Virginia | West Virginia | additional_source_ineligible | The opinion addresses insurance coverage, duty to defend, and the procedural standard for reconsidering interlocutory orders, rather than a substantive civil-liability merits issue. |
| US_4e8bd43839d97e3e1f | US_94f469d91f6b1c555d | US | West Virginia | West Virginia | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_4ee87caff5124d4003 | US_ca3855b0593ebfc33e | US | West Virginia | West Virginia | additional_source_ineligible | Insurance-coverage-only declaratory judgment concerning policy exclusions and duties to defend or indemnify; the underlying wrongful-death facts are not decided as a substantive civil-liability merits issue. |
| US_911c57466fa2dd122c | US_d15af88bf67507bd93 | US | West Virginia | West Virginia | additional_source_ineligible | The controlling opinion resolves only a statute-of-limitations and pleading/amendment issue, expressly stating that the motion does not require resolution of the negligence claim on the merits. |
| US_ace4efc8607c66874a | US_d367706577b84edea9 | US | West Virginia | West Virginia | original_audit_flag | confirmed replacement required by external review and source recheck |
| US_bdeec2edbb47432c64 | US_e2121974270074f961 | US | West Virginia | West Virginia | additional_source_ineligible | The appeal concerns dismissal based on qualified immunity and pleading requirements, not a substantive civil-liability merits determination. |
| US_cf1fe4f8bf2f7fa43b | US_f52ee52e184d02cd06 | US | West Virginia | West Virginia | original_audit_flag | confirmed replacement required by external review and source recheck |

## C. Domain reclassification

- Changed labels: 109

| Old domain | New domain | Count |
|---|---|---:|
| general_negligence_personal_injury | None | 28 |
| general_negligence_personal_injury | general_negligence_personal_injury | 54 |
| general_negligence_personal_injury | other_civil_liability | 28 |
| medical_professional_liability | None | 19 |
| medical_professional_liability | general_negligence_personal_injury | 7 |
| medical_professional_liability | medical_professional_liability | 12 |
| medical_professional_liability | other_civil_liability | 16 |
| medical_professional_liability | product_liability | 2 |
| other_civil_liability | None | 1 |
| other_civil_liability | other_civil_liability | 9 |
| product_liability | None | 5 |
| product_liability | general_negligence_personal_injury | 2 |
| product_liability | other_civil_liability | 1 |
| product_liability | product_liability | 16 |

Changed cases:

- `KR_92b1ce3452f5861ba4`: `general_negligence_personal_injury` → `None`
- `KR_f92051e8c5598d8ccc`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_16747477bbb419c699`: `general_negligence_personal_injury` → `None`
- `KR_94e863154231995db3`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_c9b0b61cc030200a6a`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_0dcfdd46d74a8951e1`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_655f2386b561c2a84a`: `general_negligence_personal_injury` → `None`
- `KR_d5131e7c37df509266`: `general_negligence_personal_injury` → `None`
- `KR_bd5712bcc29293f8e7`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_c7ec53e12d00dabce6`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_b24448d722a0190549`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_68e0deec401c3edd58`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_7f8df551c38264fcbe`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_9ff8adb69f213e129e`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_55074ea924a215c67b`: `general_negligence_personal_injury` → `None`
- `KR_b9478acaee2bfa081a`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_86fdfbe3f16878dfcf`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_26b3a33d5ab3eb8be8`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_df9ae8ebb4355cf9ab`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_3d251e8be01dc48a0b`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_c6a564478fdecad238`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_29dbd6f831cc47236c`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_908095abe4f0d6d327`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_9d2e86f030a6ea99c5`: `general_negligence_personal_injury` → `None`
- `KR_251ece560a954ca2fc`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_b8c149d3a7eebfddda`: `general_negligence_personal_injury` → `other_civil_liability`
- `KR_166fe3c2ec192a031c`: `general_negligence_personal_injury` → `None`
- `KR_d614974ea605954ae6`: `medical_professional_liability` → `other_civil_liability`
- `KR_d35502ba94fbfc6c92`: `medical_professional_liability` → `other_civil_liability`
- `KR_f3fe352a4c02d16f80`: `medical_professional_liability` → `general_negligence_personal_injury`
- `KR_78134d127e8ed36369`: `medical_professional_liability` → `other_civil_liability`
- `KR_973642690caa8c1765`: `medical_professional_liability` → `None`
- `KR_9fa690d66617cf987e`: `medical_professional_liability` → `other_civil_liability`
- `KR_eee8ecec3165d10cc6`: `medical_professional_liability` → `general_negligence_personal_injury`
- `KR_b1623e21ca7095a358`: `medical_professional_liability` → `other_civil_liability`
- `KR_8d7ca5d24afd11d1da`: `medical_professional_liability` → `None`
- `KR_1490ae2a5caa81b4c7`: `medical_professional_liability` → `None`
- `KR_fade97268b1abdcfa2`: `medical_professional_liability` → `other_civil_liability`
- `KR_4ea4b7c552d041ba9e`: `medical_professional_liability` → `other_civil_liability`
- `KR_77616f4d33f49f160f`: `medical_professional_liability` → `None`
- `KR_38f0a0551f52400584`: `medical_professional_liability` → `general_negligence_personal_injury`
- `KR_f5a5d762462130dccb`: `medical_professional_liability` → `other_civil_liability`
- `KR_fbac05e4c9ae195c2d`: `medical_professional_liability` → `None`
- `KR_4b24e5a20e1f1ddead`: `medical_professional_liability` → `other_civil_liability`
- `KR_293763f5a50c79c279`: `medical_professional_liability` → `other_civil_liability`
- `KR_f1f9176685f8c1c0db`: `medical_professional_liability` → `general_negligence_personal_injury`
- `KR_1e296dbb0bcebaab8f`: `medical_professional_liability` → `other_civil_liability`
- `KR_af61f4590681c75f54`: `medical_professional_liability` → `other_civil_liability`
- `KR_cabb12a8960b273308`: `medical_professional_liability` → `None`
- `KR_b7cbce3536a0f4f763`: `medical_professional_liability` → `other_civil_liability`
- `KR_91bd2c03cf60df4426`: `medical_professional_liability` → `other_civil_liability`
- `KR_bca778095d79c8e711`: `medical_professional_liability` → `other_civil_liability`
- `KR_2e57afc9436b5c385e`: `other_civil_liability` → `None`
- `US_3f0101285fcc76e2b1`: `general_negligence_personal_injury` → `other_civil_liability`
- `US_46a80b1c5e06a72d45`: `general_negligence_personal_injury` → `other_civil_liability`
- `US_8e816087051c99374d`: `general_negligence_personal_injury` → `other_civil_liability`
- `US_f6a3b4ef121a4e3b04`: `general_negligence_personal_injury` → `None`
- `US_9cf64af04d2e95a72d`: `medical_professional_liability` → `None`
- `US_77921eca4293bb5c27`: `medical_professional_liability` → `product_liability`
- `US_38d924f1ad3609200a`: `medical_professional_liability` → `general_negligence_personal_injury`
- `US_5e1df8e8cc45419467`: `medical_professional_liability` → `None`
- `US_63ffb974f8488a5664`: `medical_professional_liability` → `other_civil_liability`
- `US_b362543043cab0106c`: `medical_professional_liability` → `general_negligence_personal_injury`
- `US_cd09a42ab8bc802821`: `medical_professional_liability` → `product_liability`
- `US_2b3ba71069d1e5eab9`: `medical_professional_liability` → `None`
- `US_57290d297c1a43a6d4`: `medical_professional_liability` → `None`
- `US_00f7c6571f18d3c66e`: `general_negligence_personal_injury` → `None`
- `US_f42baf15f4f65533de`: `general_negligence_personal_injury` → `None`
- `US_624564663f55368864`: `medical_professional_liability` → `None`
- `US_c494ba6285904ce918`: `medical_professional_liability` → `None`
- `US_0abf417417190c2e89`: `medical_professional_liability` → `None`
- `US_861576769645010c57`: `medical_professional_liability` → `None`
- `US_f72de5a85fdd45cf64`: `medical_professional_liability` → `None`
- `US_36fd5aae9fa72a7fbb`: `medical_professional_liability` → `None`
- `US_71162df598a424d867`: `medical_professional_liability` → `general_negligence_personal_injury`
- `US_02bde060086590cc77`: `medical_professional_liability` → `None`
- `US_9652590cf5c0086fbf`: `medical_professional_liability` → `None`
- `US_f360178db4c0409aa2`: `medical_professional_liability` → `None`
- `US_545cf69d7c7286aae7`: `product_liability` → `general_negligence_personal_injury`
- `US_a072ff18b300a13dfd`: `general_negligence_personal_injury` → `None`
- `US_39be177337afd8d1be`: `general_negligence_personal_injury` → `None`
- `US_162eb334cc7d5a8b91`: `general_negligence_personal_injury` → `None`
- `US_a83924c6ae07314b33`: `general_negligence_personal_injury` → `None`
- `US_8273a9a891358bad5d`: `general_negligence_personal_injury` → `other_civil_liability`
- `US_bf6cccdd9b8a33c4a5`: `general_negligence_personal_injury` → `None`
- `US_2e67d6775a92caddee`: `general_negligence_personal_injury` → `None`
- `US_b1a79acce1af757b58`: `general_negligence_personal_injury` → `None`
- `US_e1da2d37a9dee8ec7b`: `general_negligence_personal_injury` → `None`
- `US_18fd4288d7f32131ce`: `general_negligence_personal_injury` → `None`
- `US_02cd16f932838519ba`: `product_liability` → `None`
- `US_5fb1681d1e52f4e709`: `product_liability` → `None`
- `US_74c861440b330e42dc`: `general_negligence_personal_injury` → `other_civil_liability`
- `US_1bd83c629182d24617`: `general_negligence_personal_injury` → `None`
- `US_514a1afebc7c89ad3f`: `general_negligence_personal_injury` → `None`
- `US_0cb53a4c10c6b8a81c`: `general_negligence_personal_injury` → `other_civil_liability`
- `US_bb44001945931d8610`: `general_negligence_personal_injury` → `other_civil_liability`
- `US_c2f1eaa3640f0a9bbf`: `general_negligence_personal_injury` → `None`
- `US_0bdc6cb112770284df`: `general_negligence_personal_injury` → `None`
- `US_b160a9deca3c2a648b`: `product_liability` → `general_negligence_personal_injury`
- `US_9f4d3cca3bba16c059`: `product_liability` → `None`
- `US_bfd06f8cb32e6978f3`: `product_liability` → `None`
- `US_911c57466fa2dd122c`: `general_negligence_personal_injury` → `None`
- `US_bdeec2edbb47432c64`: `general_negligence_personal_injury` → `None`
- `US_ace4efc8607c66874a`: `general_negligence_personal_injury` → `None`
- `US_15e86d45f19b975195`: `general_negligence_personal_injury` → `other_civil_liability`
- `US_4ee87caff5124d4003`: `general_negligence_personal_injury` → `None`
- `US_07482ca42c075096bb`: `general_negligence_personal_injury` → `None`
- `US_0ed0e9c92a0a12ceca`: `product_liability` → `other_civil_liability`
- `US_4e8bd43839d97e3e1f`: `product_liability` → `None`

Final domain counts:

| Country | General | Medical/professional | Product | Other |
|---|---:|---:|---:|---:|
| KR | 37 | 14 | 10 | 39 |
| US | 37 | 14 | 10 | 39 |

## D. U.S. state distribution

| State | Count |
|---|---:|
| Pennsylvania | 24 |
| Michigan | 22 |
| Louisiana | 14 |
| Nevada | 17 |
| West Virginia | 23 |

## E. New neutral facts

- Replacement cases extracted: 57
- Replacement cases translated: 57
- Replacement cases QC-passed: 57
- Retained final-roster cases amended only after final QC: 84
- Retained cases regenerated through extraction/translation: 0

## F. Final QC

- Deterministic hard checks: 200/200 pass
- Final semantic raw pass: 195/200
- Repeated semantic false positives dismissed by direct source adjudication: 5
- Final resolved pass: 200/200
- Remaining manual-review flags: 0
- All listed failure categories after adjudication: 0

## G. Final invariants

- `total_cases_200`: **TRUE**
- `kr_cases_100`: **TRUE**
- `us_cases_100`: **TRUE**
- `all_kr_supreme`: **TRUE**
- `all_us_selected_state_court_type_s`: **TRUE**
- `all_dates_eligible`: **TRUE**
- `unique_case_id`: **TRUE**
- `unique_case_family_within_country`: **TRUE**
- `all_source_eligibility_resolved_pass`: **TRUE**
- `all_domains_source_validated`: **TRUE**
- `kr_us_domain_counts_equal`: **TRUE**
- `all_five_us_states_represented`: **TRUE**
- `us_state_counts_within_10_30`: **TRUE**
- `facts_match_case_roster`: **TRUE**
- `all_qc_checks_pass`: **TRUE**
- `all_semantic_flags_resolved`: **TRUE**
- `all_ko_en_nonempty`: **TRUE**
- `retained_changes_have_amendments`: **TRUE**
- `kr_development_20`: **TRUE**
- `kr_confirmatory_80`: **TRUE**
- `us_development_20`: **TRUE**
- `us_confirmatory_80`: **TRUE**

## H. Frozen artifacts

- `outputs_v2/source_replacement_validation_v3.jsonl` — `fa41a00d1a02c9f9cca7e6c92719cd508e7c232a2992d187393e5d08fe77898d`
- `outputs_v2/domain_reclassification_v3.jsonl` — `165a21e3aa9a1c66a9326ba57442e7634ecb57423924aa859804d343d0ff4183`
- `outputs_v2/domain_reclassification_changes_v3.csv` — `5ec72f894057ad4c66b8615f0cc7d23fedad3a1e465c02101bfe7d39d7ac794d`
- `outputs_v2/replacement_selection_v3.json` — `58f2fb645b9551054a89da48be4a8b1db4841cf56daa3fb215efa5c32bc220b4`
- `outputs_v2/provisional_final_cases_200_v3.jsonl` — `cfb2a469c1b2e6a898278c56bb06c7644406904c546fbe13633cb335bde735c4`
- `outputs_v2/replacement_fact_units_v3.jsonl` — `ac81c0989dadcb56bf57c79e91a0b8e5e230fcba674fcd976267d9a04f6f4d2b`
- `outputs_v2/replacement_fact_patterns_v3.jsonl` — `d67c2c235ba6d9d5ef0c080319905031fdad78e938c084f70a3bc5158776c8c0`
- `outputs_v2/replacement_neutral_fact_qc_v3.csv` — `f3fbb5dcad0c606513d6544de2d2324446334f2757fcb6e610be4e96c230509c`
- `outputs_v2/final_cases_200_v3.jsonl` — `2e5305950cfc5ce8011ef1dc00d1aa27478b394c479b7f3730e39fd6445ba9f2`
- `outputs_v2/final_fact_units_200_v3.jsonl` — `369bc7971e54a01c21a8b11cb3d8dff4bd2104437c5b292a68c2649d1ce6dc03`
- `outputs_v2/final_fact_patterns_200_v3.jsonl` — `d95a7ce3483a95542ab6f2124014c4d58b7be751437af27896dc5a7093dcec1e`
- `outputs_v2/final_manifest_v3.csv` — `5091e7884dd953ee5127114254eaebf6f0ade750f5682981884bd43970e23595`
- `outputs_v2/final_qc_audit_200_v3.jsonl` — `97bb5f6d1412bc678b8d11f0fe11ec0020905e3a092e9bbe784bc85389d42ae0`
- `outputs_v2/final_qc_summary_v3.json` — `90bc64868bc77b11c9bf0add07da8623b6e5461dc85780857dd3d4bb134b83f9`
- `outputs_v2/collection_summary_v3.json` — `f6c843db2d2c5475935b0f6d38528922ab3354d02f3c062c71226fc9c5f9ebbd`
- `outputs_v2/retained_fact_amendments_v3.jsonl` — `55bb78b0c1578320b6f44931f34bc556aa33b6df9e1fb5ff4e43e107e46e2d07`
- `outputs_v2/retained_text_integrity_v3.json` — `e353c1baaf53699ff0ef15cefff71a0ab77382eb973e0daa6be747efd30ff18b`

## I. Tests

- New v3 regression tests: 18 passed
- Existing v2 tests: 23 passed
- Legacy tests: 180 passed
- `git diff --check`: passed (line-ending warnings only)

## J. Remaining limitations

- The 143 retained final-roster records use the manually reviewed source-language master as one aggregate source-grounded unit because no corrected per-unit artifact accompanied the authoritative 182-record file.
- Five final semantic duplicate flags recurred after correction; independent direct source adjudication found that each repeated clause added distinct causal context, so they were explicitly dismissed rather than silently overridden.
- No Exp 1 generation, PCA, marker analysis, or downstream statistical experiment was run.
