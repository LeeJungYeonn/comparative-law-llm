from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pipeline_v2.io_utils import read_jsonl, write_jsonl


OUT = Path("outputs_v2")
WORK = OUT / "v4_repair"
VERSION = "kr-us-highcourt-corpus-v4.0"


def pair(ko: str, en: str) -> tuple[str, str]:
    return " ".join(ko.split()), " ".join(en.split())


NEW_FACTS = {
    "US_0bdc6cb112770284df": pair(
        "2006년 4월 [PERSON_B]는 [PERSON_C] 소유이고 [COMPANY_A]가 보험을 제공한 [VEHICLE_A]를 운전하다 [PERSON_A]의 차량과 충돌하여 [PERSON_A]를 다치게 하였다. [PERSON_B]와 [PERSON_C]는 사고를 [COMPANY_A]에 알리거나 사고 후 조사에 협조하지 않았다. [PERSON_A]가 부상과 의료기록을 알리자 [COMPANY_A]는 기록을 검토하고 보상 제안을 검토하겠다는 서신을 보냈다. [COMPANY_A]는 나중에 보장한도보다 [CURRENCY_AMOUNT_A] 많은 금액을 제안했으나 [PERSON_A]는 이를 받아들이지 않았다. [COMPANY_A]는 차량 소유자가 사고 후 조사에 협조하지 않아 보상 의무가 없다고 주장하였다. [PERSON_A]는 운전자와 소유자에게 귀속되는 손해액을 확정받은 뒤 [COMPANY_A]에 그 금액의 지급을 요구하였다.",
        "In April 2006, [PERSON_B] was driving [VEHICLE_A], which was owned by [PERSON_C] and insured by [COMPANY_A], when it collided with [PERSON_A]'s vehicle and injured [PERSON_A]. [PERSON_B] and [PERSON_C] did not notify [COMPANY_A] of the crash or cooperate with its post-accident investigation. After [PERSON_A] reported the injuries and provided medical records, [COMPANY_A] sent letters stating that it was reviewing the records and would consider a payment offer. [COMPANY_A] later offered [CURRENCY_AMOUNT_A] more than the coverage limit, but [PERSON_A] did not accept. [COMPANY_A] maintained that the owner's failure to cooperate with the post-accident investigation ended its payment obligation. After the amount of damage attributable to the driver and owner was fixed, [PERSON_A] demanded that amount from [COMPANY_A].",
    ),
    "US_59362b11f24be41b92": pair(
        "[PERSON_A]는 [INSTITUTION_A]에 수용되어 있던 중 다쳐 [INSTITUTION_B]에 입원하였다. 입원 중 [INSTITUTION_A] 소속 [PERSON_B]가 16시간 동안 [PERSON_A]를 감시하였다. [PERSON_A]는 그 시간에 [PERSON_B]가 자신을 성폭행했다고 보고하였고, [PERSON_B]는 성적 접촉 사실은 인정하면서도 합의에 따른 것이었다고 말했다. [PERSON_A]는 합의한 적이 없다고 일관되게 말했다. 보고를 받은 [INSTITUTION_A] 직원들은 [PERSON_A]를 [INSTITUTION_B]로 다시 보내 신체검사를 받게 하였다. [PERSON_A]는 [INSTITUTION_A]와 [INSTITUTION_B]가 적절한 감독과 보호 조치를 하지 않아 신체적·정신적 손상을 입었다고 주장하였다.",
        "[PERSON_A] was confined at [INSTITUTION_A] when an injury led to admission at [INSTITUTION_B]. During the admission, [PERSON_B], an employee of [INSTITUTION_A], guarded [PERSON_A] during a sixteen-hour shift. [PERSON_A] reported that [PERSON_B] sexually assaulted her during that shift; [PERSON_B] admitted sexual contact but said it was consensual. [PERSON_A] consistently denied consenting. After the report, employees of [INSTITUTION_A] returned [PERSON_A] to [INSTITUTION_B] for a physical examination. [PERSON_A] alleged physical and emotional harm resulting from inadequate supervision and protection by [INSTITUTION_A] and [INSTITUTION_B].",
    ),
    "US_a15bd1ea6aa4ea1ed9": pair(
        "1993년 9월 10일 [PERSON_A]는 예약된 진료를 위해 [INSTITUTION_A]에 갔다. 주출입구로 이어지는 경사로를 내려가던 중 미끄러져 넘어져 다쳤다. [INSTITUTION_A]는 그 건물을 소유·관리하였다. 측정 결과 경사로의 기울기는 당시 적용되던 안전 기준보다 가팔랐다는 증언이 있었고, [INSTITUTION_A]는 그 측정 자체를 반박하지 않았으나 기울기가 낙상의 원인이 아니었다고 다투었다.",
        "On September 10, 1993, [PERSON_A] went to [INSTITUTION_A] for a scheduled appointment. While negotiating the ramp leading to the main entrance, [PERSON_A] slipped, fell, and was injured. [INSTITUTION_A] owned and managed the premises. Testimony indicated that the ramp was steeper than the applicable safety specification; [INSTITUTION_A] did not dispute the measurement but disputed that the slope caused the fall.",
    ),
    "US_5374560fdfdf536ed1": pair(
        "[PERSON_A]는 2002년 3월부터 2007년 11월까지 [INSTITUTION_A]의 경찰 책임자로 근무하였다. 내부 규정은 사용하지 않은 병가·개인휴가·연차에 대한 금전 지급을 허용했지만, 예산 부족 때문에 여러 관리자가 지급을 포기하기로 하였다는 논쟁이 있었다. 2004년 초 [PERSON_A]는 미사용 휴가대금이 지급되지 않은 것이 내부 규정에 어긋난다고 [PERSON_B], 행정담당자 및 법률담당자에게 반복하여 알렸고, 형사 보고 가능성도 언급하였다. [INSTITUTION_A]는 결국 그 금액을 지급하였다. [PERSON_B]는 [PERSON_A]가 개인적 금전 문제를 강하게 제기한 데 불만을 표시했고, 2007년 재선 뒤 [PERSON_A]를 다시 임명하지 않았다. [PERSON_A]는 반복된 보고가 그 결정의 원인이었다고 주장한 반면, [PERSON_B]는 업무수행의 여러 측면에 대한 불만 때문이었다고 말했다.",
        "[PERSON_A] served as the police chief of [INSTITUTION_A] from March 2002 through November 2007. An internal rule allowed payment for unused sick, personal, and vacation leave, while there was a dispute whether several administrators had agreed to forgo payment during a budget shortfall. In early 2004, [PERSON_A] repeatedly told [PERSON_B], an administrator, and a legal adviser that withholding the accrued payment conflicted with the internal rule, and mentioned a possible criminal report. [INSTITUTION_A] eventually made the payment. [PERSON_B] expressed anger about [PERSON_A]'s pursuit of the personal payment and, after reelection in 2007, did not reappoint [PERSON_A]. [PERSON_A] alleged that the repeated reports caused that decision; [PERSON_B] attributed it to dissatisfaction with several aspects of [PERSON_A]'s performance.",
    ),
    "US_4a0f056bcec6698d7d": pair(
        "[INSTITUTION_A]는 정수시설 개선 장비를 마련하기 위해 [COMPANY_A]가 운영하는 임대·구매 금융 프로그램을 이용하였다. [COMPANY_A]는 계약을 [COMPANY_B]에 양도했고, [COMPANY_B]는 [COMPANY_A]에 [CURRENCY_AMOUNT_A]를 송금하였다. [COMPANY_A]는 2009년 9월부터 2010년 3월까지 장비 공급자들에게 [CURRENCY_AMOUNT_B]를 지급한 뒤 나머지 자금을 전용하고 파산하였다. [INSTITUTION_A]는 공급자들에게 [CURRENCY_AMOUNT_C]를 별도로 지급하면서도 [COMPANY_B]에 매월 [CURRENCY_AMOUNT_D]를 계속 지급해야 했다. [INSTITUTION_A]는 프로그램을 후원·홍보한 [ORGANIZATION_A]가 [COMPANY_A]를 선택·추천할 때 적절히 검토하지 않았고 알려진 이상 징후도 알리지 않았다고 주장하였다. [ORGANIZATION_A]는 [INSTITUTION_A]와 [COMPANY_A] 사이의 금융 거래에 관여하지 않았다고 말했다.",
        "[INSTITUTION_A] used a lease-purchase financing program operated by [COMPANY_A] to obtain equipment for upgrades to a water-treatment facility. [COMPANY_A] assigned the agreement to [COMPANY_B], which transferred [CURRENCY_AMOUNT_A] to [COMPANY_A]. Between September 2009 and March 2010, [COMPANY_A] paid equipment vendors [CURRENCY_AMOUNT_B], diverted the remaining funds, and entered bankruptcy. [INSTITUTION_A] paid vendors an additional [CURRENCY_AMOUNT_C] while remaining obligated to make monthly payments of [CURRENCY_AMOUNT_D] to [COMPANY_B]. [INSTITUTION_A] alleged that [ORGANIZATION_A], which sponsored and marketed the program, failed to use adequate care when selecting and endorsing [COMPANY_A] and failed to disclose known irregularities. [ORGANIZATION_A] said it was not involved in the financing transactions between [INSTITUTION_A] and [COMPANY_A].",
    ),
    "US_3aa1784247c016e937": pair(
        "2000년 1월 13일 12시 51분경 [PERSON_A]는 공사 구간의 왼쪽 차로에서 [VEHICLE_A]를 시속 약 97km로 운전하고 있었다. 앞쪽 오른쪽 차로에서 [PERSON_B]가 대형 [VEHICLE_B]를 시속 약 64~72km에서 약 8~16km로 감속한 뒤 실선을 넘어 왼쪽 차로로 이동하였다. [PERSON_A]는 오른쪽으로 피하려다 바퀴를 바로잡고 급제동했지만 약 31m를 미끄러진 뒤 [VEHICLE_B]의 뒤를 들이받았다. [VEHICLE_A]의 앞부분이 [VEHICLE_B] 아래에 깔렸다. 당시 23세였던 [PERSON_A]는 뇌 전두측두부 손상을 입어 발작, 단기기억 저하, 지능 저하, 성격 변화와 행동 억제 저하가 남았고 장기간의 주거형 치료가 필요하게 되었다. [PERSON_A]가 충분히 주시하고 제때 제동했는지와 [PERSON_B]가 두 차로를 막는 위험한 진로변경을 했는지가 서로 다투어졌다.",
        "At about 12:51 p.m. on January 13, 2000, [PERSON_A] was driving [VEHICLE_A] at about 97 km/h in the left lane of a construction zone. Ahead in the right lane, [PERSON_B] slowed a large [VEHICLE_B] from about 64-72 km/h to about 8-16 km/h and moved across a solid line into the left lane. [PERSON_A] steered right, straightened the wheels, and braked hard, but skidded about 31 meters before striking the rear of [VEHICLE_B]. The front of [VEHICLE_A] was crushed beneath [VEHICLE_B]. [PERSON_A], then age twenty-three, sustained an anterior temporal brain injury followed by seizures, short-term-memory deficits, reduced intellectual function, personality changes, and disinhibition requiring long-term residential treatment. The parties disputed whether [PERSON_A] kept an adequate lookout and braked in time and whether [PERSON_B]'s lane change created a hazard across both lanes.",
    ),
    "US_979762a92670a0552a": pair(
        "2004년 6월 4일 [PERSON_A]는 도로변에 차를 세우고 타이어를 수리하고 있었고, 동료 [PERSON_B]가 도우러 왔다. 두 사람이 차량 사이에서 물건을 옮기던 중 [PERSON_C]가 운전한 [VEHICLE_A]가 이들을 들이받았다. [PERSON_A]는 사망했고 [PERSON_B]는 중상을 입었다. [PERSON_C]는 통제약물의 영향을 받은 상태로 운전한 사실로 체포되었다. 2003년 6월 [ORGANIZATION_A]는 [PERSON_C]가 2002년 5월부터 2003년 5월까지 13개 약국에서 약 4,500정의 하이드로코돈을 받은 기록을 해당 약국들과 처방 의료인들에게 알렸다. [COMPANY_A]를 포함한 약국들은 이 통지를 받은 뒤에도 [PERSON_C]에게 사고 전에 사용한 통제약물을 계속 조제하였다. 각 처방전 자체가 위조되었거나 표시된 용량대로 복용할 경우 개별적으로 위험했다는 주장은 없었다. 피해자 측은 대량·반복 조제 기록을 통지받은 약국들이 추가 확인이나 경고 없이 계속 조제한 것이 사고에 기여했다고 주장했고, 약국들은 알지 못하는 도로 이용자에게까지 [PERSON_C]의 운전을 통제할 수는 없었다고 다투었다.",
        "On June 4, 2004, [PERSON_A] had stopped beside a highway to repair a tire, and coworker [PERSON_B] arrived to help. While the two moved items between their vehicles, [VEHICLE_A], driven by [PERSON_C], struck them. [PERSON_A] died and [PERSON_B] was seriously injured. [PERSON_C] was arrested for driving while under the influence of controlled substances. In June 2003, [ORGANIZATION_A] notified the pharmacies and prescribers involved that records showed [PERSON_C] had obtained about 4,500 hydrocodone pills from thirteen pharmacies between May 2002 and May 2003. After receiving the notice, pharmacies including [COMPANY_A] continued dispensing to [PERSON_C] the controlled substances she used before the crash. It was not alleged that the individual prescriptions were forged or that any one prescription, taken as written, presented a harmful dosage. The affected people alleged that continued dispensing without further inquiry or warning after notice of the repeated large-volume activity contributed to the crash; the pharmacies disputed that they could control [PERSON_C]'s driving for the benefit of unidentified road users.",
    ),
    "US_6fc55880fb74718547": pair(
        "1998년 2월 12일 [PERSON_A]가 [INSTITUTION_A]에 있던 중 선반이 부러지면서 쌓여 있던 플라스틱 접시들이 떨어졌고, 그중 하나 이상이 [PERSON_A]의 이마를 쳤다. 이후 [PERSON_A]는 신체적·심리적 증상과 소득상실을 호소하고 장기간의 치료와 돌봄이 필요하다고 주장하였다. 여러 의료전문가들은 해당 충격이 증상을 일으키거나 악화했는지에 관해 서로 다른 의견을 제시하였다. [INSTITUTION_A]는 선반 사고 자체는 다투지 않았지만 이후 증상과 필요한 장래 치료의 범위가 그 사고에서 비롯되었는지는 다투었다.",
        "On February 12, 1998, a shelf broke while [PERSON_A] was inside [INSTITUTION_A], causing stacked plastic platters to fall and one or more to strike [PERSON_A]'s forehead. [PERSON_A] later reported physical and psychological symptoms, loss of income, and a need for long-term treatment and care. Several medical specialists gave differing opinions about whether the impact caused or aggravated those conditions. [INSTITUTION_A] did not dispute the shelf incident but disputed whether the later symptoms and the asserted scope of future care resulted from it.",
    ),
    "US_f57bc43fc6478fd8b2": pair(
        "[COMPANY_A]는 약 50kg의 쌀 포대를 팔레트마다 36~42개씩 교차 적재하고 층 사이를 접착하였다. 운송 때에는 팔레트를 한 층으로 싣고 차량에 묶었지만, 창고에서는 하역업체가 팔레트를 세 층으로 쌓았다. 1994년 11월 3일 창고에서 일부 포대가 떨어지자 장기하역 노동자인 [PERSON_A]가 이를 정리하였다. [PERSON_A]가 등을 돌린 채 떨어진 포대를 줍던 중 팔레트 두 개가 무너져 포대들이 [PERSON_A]를 완전히 덮쳤다. [PERSON_A]는 병원으로 이송된 뒤 탈장수술과 두 부위의 경추수술을 받았으나 경추 유합이 실패했고 흉곽출구증후군도 진단받았다. 이후 업무에 복귀하지 못했고 허리와 오른쪽 무릎 통증도 계속되었다. 포대의 원래 적재·접착 방식과 창고의 삼단 적재 중 무엇이 붕괴에 기여했는지가 다투어졌다.",
        "[COMPANY_A] filled approximately 50-kilogram rice sacks, stacked 36 to 42 sacks on each pallet in cross-tied layers, and applied glue between the layers. The pallets were shipped one layer high and strapped to trucks, but a warehouse operator stacked them three pallets high after unloading. On November 3, 1994, sacks had fallen in the warehouse, and [PERSON_A], a longshore worker, was assigned to collect them. While [PERSON_A] was picking up sacks with his back to the stack, two pallets collapsed and completely covered him. [PERSON_A] was taken to a hospital and later underwent surgery for a hernia and for cervical discs at two levels; the cervical fusions failed, and thoracic outlet syndrome was also diagnosed. [PERSON_A] did not return to work and continued to have lower-back and right-knee pain. The parties disputed whether the mill's original stacking and gluing or the warehouse's three-high stacking contributed to the collapse.",
    ),
    "US_384364b54638f78623": pair(
        "2007년 4월 [PERSON_A]는 신체 이상으로 [INSTITUTION_A] 응급실에 갔다. 응급실 의사는 [PERSON_A]를 [PERSON_B]에게 의뢰했고, [PERSON_B]는 입원시켜 치료하였다. [PERSON_B]가 자리를 비운 동안 같은 진료조직의 [PERSON_C]가 치료하고 [PERSON_D]에게 외과 협진을 요청하였다. [PERSON_D]는 입원 며칠 뒤 탐색수술을 시행하였다. 수술 뒤 [PERSON_A]에게 중증 감염이 발생해 여러 차례 추가 수술을 받았고 영구적 손상이 남았다고 주장하였다. [PERSON_B]와 [PERSON_C]는 [COMPANY_A]에 고용되어 계약에 따라 [INSTITUTION_A] 환자만 진료했고, [PERSON_D]도 별도 회사와의 계약에 따라 [INSTITUTION_A]에서 수술서비스를 제공하였다. [PERSON_A]는 이 조직·계약 관계 때문에 [INSTITUTION_A]도 의료진의 행위와 감염 결과에 책임이 있다고 주장했고, [INSTITUTION_A]는 해당 의료진이 독립적으로 일했다고 다투었다.",
        "In April 2007, [PERSON_A] went to the emergency department of [INSTITUTION_A] with a physical ailment. An emergency physician referred [PERSON_A] to [PERSON_B], who admitted and treated the patient. While [PERSON_B] was away, [PERSON_C] from the same practice provided care and requested a surgical consultation from [PERSON_D]. A few days after admission, [PERSON_D] performed exploratory surgery. [PERSON_A] developed a serious infection after surgery, underwent several follow-up operations, and alleged permanent injury. [PERSON_B] and [PERSON_C] were employed by [COMPANY_A] and treated only patients of [INSTITUTION_A] under a contract; [PERSON_D] likewise provided surgical services there under a separate company's contract. [PERSON_A] alleged that these organizational and contractual relationships made [INSTITUTION_A] responsible for the clinicians' conduct and the infection, while [INSTITUTION_A] maintained that the clinicians worked independently.",
    ),
    "US_b675276d7ca44f64e2": pair(
        "2002년 10월 29일 임신 28.4주였던 [PERSON_A]는 2002년 11월 3일 심한 복부경련과 출혈로 [INSTITUTION_A] 응급실에 갔다. 의료진은 태아 심박수 143회를 기록했지만 [INSTITUTION_A]에 필요한 감시장비와 수술장비가 없어 더 높은 수준의 신생아 진료가 가능한 [INSTITUTION_B]로 이송하기로 하였다. 05시 30분 이송서에는 상태 악화 가능성이 없다고 기재되었고, 이송 중 태아 심박수는 170회였다. 06시 29분 도착 직후에는 심박동이 잡히지 않았으나 이후 60~144회와 160~180회의 기록이 있었다. [PERSON_B]는 06시 55분 초음파에서 심박동을 확인하지 못했고, 07시 13분 두피전극에서 60~144회가 측정된 뒤 07시 14분 제왕절개를 지시하였다. 수술은 07시 25분 시작되었고 07시 33분 사산아가 분만되었다. [PERSON_A]는 [PERSON_B]의 도착과 수술이 지연되어 태아가 사망했다고 주장했지만, [PERSON_B]는 태아가 이송 중 이미 심한 상태 악화를 겪었고 더 이른 조치도 결과를 바꾸지 못했을 것이라고 다투었다.",
        "On October 29, 2002, [PERSON_A] was 28.4 weeks pregnant. On November 3, 2002, she went to the emergency department of [INSTITUTION_A] with severe abdominal cramping and bleeding. Staff recorded a fetal heart rate of 143, but [INSTITUTION_A] lacked the monitoring and surgical equipment needed for the pregnancy, so a transfer was arranged to [INSTITUTION_B] for higher-level neonatal care. A 5:30 a.m. transfer form stated that material deterioration was not expected, and the fetal heart rate was 170 during transport. No heartbeat was detected immediately after arrival at 6:29 a.m., although later readings ranged from 60 to 144 and from 160 to 180. [PERSON_B] found no heartbeat by ultrasound at 6:55 a.m.; after a scalp electrode recorded 60 to 144 at 7:13 a.m., [PERSON_B] ordered a cesarean delivery at 7:14 a.m. Surgery began at 7:25 a.m., and a stillborn fetus was delivered at 7:33 a.m. [PERSON_A] alleged that delay in [PERSON_B]'s arrival and surgery caused the death; [PERSON_B] disputed this and said the fetus had already deteriorated during transfer and earlier intervention would not have changed the outcome.",
    ),
    "US_472f038dea6238a51e": pair(
        "2011년 9월 [PERSON_A]는 학교 조리실에서 일하다 넘어져 왼쪽 팔꿈치를 다쳤다. 영상검사에서 요골두 아탈구, 여러 골절, 중등도 관절염 및 외상 후 변형이 보였다. 두 차례 정형외과 진료 뒤 [PERSON_A]는 2011년 12월 [PERSON_B]에게 진료받았고, [PERSON_B]는 팔꿈치 전치환술을 권하였다. 동의서에는 신경 손상을 포함한 위험과 수련의 등 다른 의료인이 수술 일부를 맡을 수 있다는 내용이 있었다. 2012년 2월 12일 [PERSON_B]와 수련의가 수술한 뒤 [PERSON_A]는 팔과 손의 무감각을 겪었고, 검사에서 척골신경의 감각 부위 손상이 확인되었다. [PERSON_B]는 신경병증이 수술로 생긴 흔한 합병증이라고 말했고, [PERSON_A] 측 전문가는 전치환술 대신 척골신경을 노출하지 않는 요골두 절제술을 했어야 한다고 의견을 밝혔다.",
        "In September 2011, [PERSON_A] fell while working in a school kitchen and injured her left elbow. Imaging showed radial-head subluxation, several fractures, moderate arthritis, and post-traumatic deformity. After seeing two orthopedists, [PERSON_A] was evaluated by [PERSON_B] in December 2011, and [PERSON_B] recommended a total elbow replacement. The consent form listed nerve damage among the risks and stated that a resident or other clinician might perform part of the procedure. On February 12, 2012, [PERSON_B] and a resident performed the surgery, after which [PERSON_A] experienced numbness in her arm and hand; testing showed damage to the sensory portion of the ulnar nerve. [PERSON_B] described the neuropathy as a common complication caused by the operation, while an expert for [PERSON_A] said a radial-head excision that did not expose the ulnar nerve should have been used instead of total replacement.",
    ),
    "US_a4ba9463b694840f34": pair(
        "65세인 [PERSON_A]는 만성 폐질환, 수면무호흡과 이산화탄소 저류가 있었고, 과거 같은 증상으로 입원했을 때 양압환기와 호흡치료를 받고 회복하였다. 2010년 9월 3일 같은 폐 증상이 악화되어 [INSTITUTION_A]에 입원했으며, 기록에는 [MEDICATION_A] 50mg에 과도하게 진정된 과거 반응이 기재되어 있었다. 의료진은 먼저 [MEDICATION_B] 5mg을 투여하고 약 두 시간 뒤 [MEDICATION_A] 100mg을 투여하였다. [PERSON_A]는 더 혼란스럽고 흥분한 뒤 심하게 진정되어 손목을 고정한 채 누워 있었고, 다음 날 [MEDICATION_A] 25mg을 추가로 받았다. 초기 검사에서 혈중 이산화탄소가 매우 높았지만 후속 동맥혈가스 검사는 없었고 호흡 전문의도 참여하지 않았다. 필요한 양압환기 처방은 9월 4일 22시에야 이루어졌다. [PERSON_A]는 완전히 깨어나지 못한 채 9월 5일 부정맥, 서맥과 심정지를 겪고 07시에 사망하였다. 사망기록에는 만성 고탄산혈증성 호흡부전의 악화와 [MEDICATION_A] 부작용이 기재되었다.",
        "[PERSON_A], age sixty-five, had chronic lung disease, sleep apnea, and carbon-dioxide retention and had recovered during an earlier admission after positive-pressure ventilation and breathing treatments. On September 3, 2010, the same pulmonary condition worsened and [PERSON_A] was admitted to [INSTITUTION_A]; the record noted a prior excessive-sedation reaction to 50 mg of [MEDICATION_A]. Staff first administered 5 mg of [MEDICATION_B] and about two hours later administered 100 mg of [MEDICATION_A]. [PERSON_A] became more confused and agitated, then heavily sedated while lying in wrist restraints, and received another 25 mg of [MEDICATION_A] the next day. Initial testing showed very high blood carbon-dioxide levels, but no follow-up arterial-blood-gas study was ordered and no pulmonary specialist was consulted. Positive-pressure ventilation was not ordered until 10:00 p.m. on September 4. Without fully awakening, [PERSON_A] developed arrhythmia, bradycardia, and cardiac arrest and died at 7:00 a.m. on September 5. The death record identified acute-on-chronic hypercapnic respiratory failure associated with an adverse reaction to [MEDICATION_A].",
    ),
    "US_0ffb071db74bea0121": pair(
        "2008년 8월 [PERSON_A]는 술과 약물을 섭취한 저녁 뒤 [PERSON_B]를 태운 [VEHICLE_A]를 운전하다 기둥을 들이받았다. [PERSON_B]는 그날 새벽 엉덩이 통증으로 응급실에 갔지만, 당시 진료기록에는 친구의 계단 낙상을 막다가 다쳤다는 설명이 적혔고 엑스레이에서는 골절이 보이지 않았다. [PERSON_B]는 [PERSON_A]의 친척이 사실을 숨기자고 부탁해 그 설명에 따랐다고 말했다. 통증이 계속되어 추가 진료를 받은 [PERSON_B]는 2009년 9월 15일 고관절 골절을 진단받고 결국 고관절 전치환술을 받았다. [PERSON_A]는 자신이 운전하지 않았다고 말했지만 목격자는 운전자와 비슷한 사람이 현장을 떠났다고 보고했고, [PERSON_A]의 배우자는 차량 도난신고가 거짓이었다고 인정하였다. 의료진은 해당 유형의 골절이 자동차 충돌과 일치하고 초기 엑스레이에서 누락될 수 있다고 말했지만, 골절의 정확한 원인을 확정하지는 못했다.",
        "In August 2008, after an evening of alcohol and drug use, [PERSON_A] drove [VEHICLE_A] with [PERSON_B] as a passenger and crashed into a pole. [PERSON_B] went to an emergency department that morning with hip pain, but the medical history described an injury while preventing a friend from falling down stairs, and an x-ray showed no fracture. [PERSON_B] said he followed that account after a relative of [PERSON_A] asked him to conceal the crash. Continued pain led to further treatment; on September 15, 2009, [PERSON_B] was diagnosed with a hip fracture and ultimately underwent total hip replacement. [PERSON_A] denied being the driver, but a witness reported that a person matching [PERSON_A]'s description left the scene, and [PERSON_A]'s spouse admitted that a vehicle-theft report was false. Clinicians said this type of fracture was consistent with a motor-vehicle collision and could be missed on an initial x-ray, but they could not identify its precise cause.",
    ),
    "US_2999172a4de14d7523": pair(
        "1995년 전시회 준비 중 [PERSON_A]는 조명기사로서 전시 부스 조명 설치를 감독하였다. [COMPANY_A] 직원들은 조명용 트러스 구조물을 조립해 나일론 끈으로 임시 고정하고, 최종 고정에 필요한 볼트나 클램프가 아직 설치되지 않았다고 다른 작업자들에게 알렸다. 그럼에도 다른 업체 직원들의 감독 아래 조명 설치가 계속되었다. 두 번째 조명이 매달린 직후 트러스가 흔들리며 [PERSON_A] 쪽으로 무너졌고, [PERSON_A]는 약 4m 높이의 지게차에서 얼굴부터 바닥으로 떨어졌다. [PERSON_A]는 두개골, 부비동, 양쪽 손목과 팔꿈치 골절 등 중상을 입었다. 트러스의 임시 고정 상태를 알린 [COMPANY_A]와 고정 전 설치를 진행한 다른 업체들의 각 행위가 사고에 얼마나 기여했는지가 다투어졌다.",
        "During preparation for a 1995 trade show, [PERSON_A] worked as a lighting technician supervising installation at an exhibit booth. Employees of [COMPANY_A] assembled a lighting truss, temporarily secured it with nylon straps, and told other workers that the bolts or clamps needed for final stabilization had not yet been installed. Lighting installation nevertheless continued under the supervision of employees from other companies. Immediately after the second light was hung, the truss swayed and collapsed toward [PERSON_A], knocking him face-first from a forklift about four meters to the ground. [PERSON_A] sustained a fractured skull, fractured sinuses, fractures of both wrists and an elbow, and other serious injuries. The parties disputed how much the temporary securing by [COMPANY_A] and the other companies' decision to proceed before final stabilization contributed to the accident.",
    ),
    "US_107f5beb1553bc775c": pair(
        "[COMPANY_A]의 트럭이 발전기를 견인하던 중 발전기가 [PERSON_A]가 타고 있던 [VEHICLE_A]를 충격하였다. [PERSON_A]는 당시 업무 중이었고 충돌로 신체 상해를 입어 치료를 받았다. 의료기관들은 치료비를 청구했으나 별도 보상기관과의 협의로 청구액보다 적은 금액을 전액 변제로 받았다. [PERSON_A]는 청구된 의료비와 그 밖의 상해 손실을 요구했고, [COMPANY_A]는 실제 지급된 치료비와 제3자가 지급한 보상액도 함께 고려해야 한다고 다투었다.",
        "A truck operated by [COMPANY_A] was towing a generator when the generator struck [VEHICLE_A], occupied by [PERSON_A]. [PERSON_A] was working at the time and sustained bodily injuries requiring medical treatment. Medical providers issued bills but accepted lower negotiated amounts as full payment through a separate compensation provider. [PERSON_A] sought the billed medical expenses and other injury losses, while [COMPANY_A] disputed the amount and said the sums actually paid and the third-party benefits should also be considered.",
    ),
    "US_b267b8668935378e25": pair(
        "2011년 9월 13일 [PERSON_A]가 퇴근하며 [VEHICLE_A]를 운전하던 중 [PERSON_B]의 밴이 교차로에서 앞을 가로질러 좌회전하였다. [PERSON_A]는 밴과의 충돌을 피하지 못했고, 충격으로 [VEHICLE_A]가 회전해 세 번째 차량과도 부딪혔다. [PERSON_A]는 현장에서 응급처치를 받고 병원으로 이송된 뒤 목과 허리 통증, 팔로 뻗치는 통증과 작열감을 계속 호소하였다. 정형외과 진료, 물리치료, 통증조절기 사용, 약 30회의 척추교정치료와 주사치료를 받았고 한동안 일하지 못했다. 업무에 복귀한 뒤에도 통증과 운동범위 제한이 남았으며, 한 의료인은 손상이 영구적이고 악화가 반복될 수 있다고 말했다. [PERSON_B]는 자신의 운전이 충돌을 일으킨 점은 인정했지만, [PERSON_A]의 증상과 손실의 범위는 다투었다.",
        "On September 13, 2011, [PERSON_A] was driving [VEHICLE_A] home from work when a van driven by [PERSON_B] turned across her path at an intersection. [PERSON_A] could not avoid the van, and the impact spun [VEHICLE_A] into a third vehicle. [PERSON_A] received emergency care at the scene and was transported to a hospital, then continued to report neck and back pain, radiating arm pain, and burning sensations. Treatment included orthopedic care, physical therapy, a pain-control device, about thirty chiropractic sessions, and injections, and [PERSON_A] remained off work for a period. Pain and reduced range of motion persisted after she returned to work, and one clinician described the injury as permanent with recurring flare-ups. [PERSON_B] admitted that his driving caused the collision but disputed the extent of [PERSON_A]'s symptoms and losses.",
    ),
    "US_97456da392d82990a3": pair(
        "1996년 3월 7일 [PERSON_A]에게 매일 [MEDICATION_A] 5mg을 투여하라는 처방이 내려졌다. [PHARMACY_A]의 [PERSON_B]는 라벨에 매일 20mg을 투여하도록 잘못 기재하였다. [PERSON_A]의 부모는 3월 9일부터 11일까지 매일 20mg을 투여했고, [PERSON_A]가 점점 공격적이고 비합리적으로 변해 타인과 자신을 해치겠다고 말하였다고 보고하였다. 3월 10일 부모가 용량을 확인해 달라고 전화하자 [PHARMACY_A]의 관리자는 원 처방을 확인하지 않은 채 20mg이 맞다고 답하였다. 3월 11일 치료자에게 확인한 결과 원 처방은 5mg이었음이 드러났고, [PERSON_A]는 그날 진료기관에 입원하여 관찰과 검사를 받은 뒤 다음 날 퇴원하였다. [PHARMACY_A]는 라벨 오류는 인정했지만, 이후 행동 변화와 장기적 손상이 과량 투여에서 비롯되었는지는 다투었다.",
        "On March 7, 1996, [PERSON_A] was prescribed 5 mg of [MEDICATION_A] once daily. [PERSON_B] at [PHARMACY_A] incorrectly labeled the medication for a 20 mg daily dose. [PERSON_A]'s parents administered 20 mg on March 9, 10, and 11 and reported that [PERSON_A] became increasingly aggressive and irrational and threatened harm to others and himself. When a parent called on March 10 to verify the dosage, a manager at [PHARMACY_A] did not check the original prescription and said that 20 mg was correct. A treating professional confirmed on March 11 that the prescription was for 5 mg, and [PERSON_A] was admitted that day for observation and testing and released the next afternoon. [PHARMACY_A] admitted the labeling error but disputed whether the subsequent behavioral changes and any lasting harm resulted from the excess doses.",
    ),
    "KR_c0c025922aca493f10": pair(
        "[PERSON_A]는 [COMPANY_A]가 운영하는 숙박시설의 1박 이용권을 구입했으며, 이용권에는 1명의 무료 승마체험이 포함되어 있었다. 2014년 11월 15일 09시경 [COMPANY_A]의 관리자는 현장에 있던 [PERSON_B]에게 [PERSON_A]의 승마를 지도해 달라고 부탁하였다. [PERSON_B]는 [PERSON_A]의 경험을 간단히 확인하고 온순한 말을 골라 손잡이 안장을 얹었지만 사전 안전교육을 하지 않았고 헬멧과 신발 등 기본 안전장비 착용도 확인하지 않았다. [PERSON_B]는 고삐를 잡고 걷기와 속보를 지도한 뒤 구보를 시작한다고 알렸으나, [PERSON_A]의 신체 상태와 의사를 충분히 확인하거나 준비시간을 주지 않았다. 말이 구보를 위해 도움닫기를 하는 순간 [PERSON_A]가 손잡이를 놓쳐 떨어졌다. [PERSON_A]는 오른쪽 상완골과 골반 골절 등을 입어 2014년 11월 16일부터 12월 2일까지 수술과 입원치료를 받았다.",
        "[PERSON_A] bought a one-night stay from lodging operated by [COMPANY_A], with a complimentary horse-riding experience for one guest. At about 9:00 a.m. on November 15, 2014, a manager of [COMPANY_A] asked [PERSON_B], who was present at the property, to guide [PERSON_A]'s ride. [PERSON_B] briefly asked about [PERSON_A]'s experience, selected a gentle horse, and fitted a saddle with a handhold, but gave no advance safety instruction and did not confirm use of basic protective gear such as a helmet and suitable shoes. Holding the reins, [PERSON_B] guided the horse at a walk and trot, then announced a canter without adequately checking [PERSON_A]'s physical condition or wishes or allowing sufficient preparation time. As the horse accelerated into the canter, [PERSON_A] lost the handhold and fell. [PERSON_A] sustained fractures of the right humerus and pelvis and received surgery and inpatient treatment from November 16 through December 2, 2014.",
    ),
    "KR_d47698374deaa59285": pair(
        "2006년 10월 3일 07시 40분경 짙은 안개 속에서 [PERSON_A]가 25톤 [VEHICLE_A]를 운전하다 앞서 서행하던 1톤 [VEHICLE_B]를 들이받고 [VEHICLE_A]를 2차로에 세워 둔 채 후방 경고표지 등 안전조치를 하지 않았다. 07시 41분부터 여러 차량이 연속해 충돌했고 일부 탑승자들은 갓길로 대피하였다. 07시 53분경 [PERSON_B]의 화물차가 갓길과 3차로에 걸쳐 있던 탱크로리를 들이받았고, 돌출된 엔진과 프레임에서 불꽃이 발생해 앞선 차량들에서 유출된 연료에 불이 붙었다. 갓길로 대피한 [PERSON_C]와 [PERSON_D]는 밀려난 차량들 사이에 갇혀 질식 또는 화상으로 사망했고, 다른 차량에서 내려 도로에 있던 [PERSON_E]도 화재 연기를 흡입해 사망하였다. 최초 충돌 뒤의 정차와 안전조치 부재, 이어진 운전자들의 전방주시 및 안전거리 부족이 연쇄충돌과 화재에 각각 기여하였다.",
        "At about 7:40 a.m. on October 3, 2006, in dense fog, [PERSON_A] drove a 25-ton [VEHICLE_A] into a slower one-ton [VEHICLE_B], left [VEHICLE_A] stopped in the second lane, and did not place a rear warning marker or take other safety measures. From 7:41 a.m., several vehicles collided in sequence and some occupants moved to the shoulder. At about 7:53 a.m., a cargo truck driven by [PERSON_B] struck a tanker stopped partly on the shoulder and partly in the third lane; exposed engine and frame components produced sparks that ignited fuel spilled from earlier vehicles. [PERSON_C] and [PERSON_D], who had moved to the shoulder, became trapped among displaced vehicles and died from smoke inhalation or burns, while [PERSON_E], who had exited another vehicle and was on the roadway, also died from fire smoke. The initial collision, the stopped vehicle and missing warnings, and later drivers' failures to maintain lookout and following distance each contributed to the chain collisions and fire.",
    ),
    "KR_25ab15d21ac1967afd": pair(
        "1986년생인 [PERSON_A]는 세 살 때 발견된 선천성 척추측만증으로 진료받다가 2004년 11월 8일 [INSTITUTION_A]에 입원하였다. 당시 척추측만각은 66도였고 양쪽 다리의 근력과 감각은 정상이었다. 의료진은 향후 신경손상을 예방하기 위해 11월 12일 07시 30분부터 13시 30분까지 척추 일부와 디스크를 제거하고 봉으로 척추를 교정하는 수술을 시행하였다. 수술 직후 [PERSON_A]는 양쪽 다리의 근력과 감각을 완전히 잃었다. 의료진은 원인을 찾기 위해 11월 13일과 15일 두 차례 추가 수술을 했지만 활동성 혈종, 출혈이나 척수 압박을 발견하지 못하였다. 일부 기능은 회복되었으나 영구적 양하지 부전마비와 대소변 장애가 남아 보행이 불가능하고, 휠체어와 하루 12시간의 성인 돌봄이 필요하게 되었다. 첫 수술 중 신경감시장치를 사용했지만 저장자료 일부가 손상되어 결과를 제출할 수 없다는 설명이 있었다.",
        "[PERSON_A], born in 1986, had received care for congenital scoliosis found at age three and was admitted to [INSTITUTION_A] on November 8, 2004. The spinal curvature measured 66 degrees, and strength and sensation in both legs were normal. To prevent future neurological injury, clinicians operated from 7:30 a.m. to 1:30 p.m. on November 12, removing portions of vertebrae and discs and correcting the spine with rods. Immediately after surgery, [PERSON_A] completely lost strength and sensation in both legs. Clinicians performed two further operations on November 13 and 15 to find the cause but observed no active hematoma, bleeding, or spinal-cord compression. Some function returned, but permanent partial paralysis of both legs and bowel and bladder impairment remained; [PERSON_A] could not walk and required a wheelchair and twelve hours of adult assistance each day. Neural monitoring was used during the first operation, but [INSTITUTION_A] said part of the stored data had been damaged and could not be produced.",
    ),
    "KR_e61f8571b55fd48e4c": pair(
        "2014년 9월 7일 [PERSON_A]는 혈중알코올농도 0.170% 상태로 [VEHICLE_A]를 운전하다 횡단보도를 건너던 [PERSON_B]를 들이받았다. [PERSON_B]는 2014년 9월 18일 사망하였다. 당시 [PERSON_B]는 만 24세 5개월로 의과대학 본과 3학년 2학기에 재학 중이었고, 예과 평균학점은 3.16, 본과 평균학점은 3.01이었다. 같은 단계까지 휴학이나 유급 없이 재학한 동급 학생들의 2012년부터 2015년까지 의사자격시험 합격률은 92~100%였다. [PERSON_B]의 부모는 사망으로 인한 소득상실과 정신적 손상을 주장하였다.",
        "On September 7, 2014, [PERSON_A], with a blood-alcohol concentration of 0.170 percent, drove [VEHICLE_A] into [PERSON_B] while [PERSON_B] was crossing at a marked crossing. [PERSON_B] died on September 18, 2014. [PERSON_B] was twenty-four years and five months old and in the second semester of the third year of medical school, with a 3.16 premedical grade average and a 3.01 medical-course average. Between 2012 and 2015, students who reached the same stage without leave or repeating a year had professional-licensing pass rates of 92 to 100 percent. [PERSON_B]'s parents alleged lost future income and emotional harm resulting from the death.",
    ),
    "KR_1346b7ea49f678b93c": pair(
        "[PERSON_GROUP_A]는 1988년 12월경부터 1998년 9월경 사이에 [INSTITUTION_A] 또는 [INSTITUTION_B]에 수용되었다. 일부는 적법한 절차 없이 감금되고 강제노역을 하였으며, 이를 거부하거나 항의할 때 폭행이나 진정제 투여를 당했다고 진술하였다. 1997년 2월 1일 [PERSON_A]가 가족의 도움으로 퇴소한 뒤, 2월 3일 담당 공무원 [PERSON_B]에게 불법 감금, 강제노역과 폭행을 알리고 시정을 요구하였다. [PERSON_B]는 보상이나 시정을 언급하며 [PERSON_A]를 돌려보냈지만 시설 조사, 관계자 질문, 기록 점검이나 그 밖의 후속조치를 하지 않았다. 그날 [PERSON_A]는 시설 직원들에게 폭행당해 다시 끌려갔고, 그 뒤에도 일부 수용자에 대한 폭행 등 부당한 대우가 이어졌다고 보고되었다. [PERSON_GROUP_A]는 시설 운영자들의 행위와 감독 담당자의 부작위 때문에 신체적·정신적 손상을 입었다고 주장하였다.",
        "[PERSON_GROUP_A] were confined at [INSTITUTION_A] or [INSTITUTION_B] at various times between about December 1988 and September 1998. Some described confinement without proper process, forced labor, and beatings or administration of sedatives when they refused the labor or protested. After [PERSON_A] left with family assistance on February 1, 1997, [PERSON_A] told the responsible official, [PERSON_B], on February 3 about unlawful confinement, forced labor, and beatings and requested corrective action. [PERSON_B] mentioned compensation or correction and sent [PERSON_A] away but did not inspect the facilities, question those involved, examine records, or take other follow-up action. That day facility employees assaulted [PERSON_A] and took [PERSON_A] back, and mistreatment including assaults on some confined people reportedly continued. [PERSON_GROUP_A] alleged physical and emotional harm from the operators' conduct and the supervising official's inaction.",
    ),
    "KR_6a3e8f577c00b9c20c": pair(
        "당시 11세 7개월인 [PERSON_A]는 모야모야병 치료를 위해 2016년 6월 17일 [INSTITUTION_A]에 내원하였다. [PERSON_A]의 어머니 [PERSON_B]는 우회로 조성술 전 뇌혈관조영술이 필요하다는 설명을 듣고 동의서에 서명하였다. [PERSON_A]는 6월 30일 입원한 뒤 7월 1일 09시부터 10시 20분까지 조영술을 받고 10시 37분 병실로 옮겨졌다. 12시 2분부터 입술이 실룩이는 간헐적 경련이 시작되어 16시 1분 잠시 가라앉았다가 16시 20분 다시 나타났다. 17시 26분 촬영한 MRI에서 왼쪽 중대뇌동맥의 급성 뇌경색이 보였고, 18시 52분 중환자실로 옮겨졌다. 7월 13일 우회로 조성술을 받고 7월 20일 퇴원했지만 영구적인 오른쪽 편마비와 언어기능 저하가 남았다. [PERSON_A]는 조영술의 위험과 시행 후 관찰에 관한 설명 및 조치가 충분하지 않았다고 주장하였다.",
        "[PERSON_A], then eleven years and seven months old, went to [INSTITUTION_A] on June 17, 2016, for treatment of moyamoya disease. [PERSON_A]'s mother, [PERSON_B], was told that cerebral angiography was needed before bypass surgery and signed the consent form. After admission on June 30, [PERSON_A] underwent angiography from 9:00 to 10:20 a.m. on July 1 and returned to the room at 10:37 a.m. Intermittent lip-twitching seizures began at 12:02 p.m., appeared to subside at 4:01 p.m., and returned at 4:20 p.m. An MRI at 5:26 p.m. showed an acute infarction in the left middle cerebral artery, and [PERSON_A] was transferred to intensive care at 6:52 p.m. Bypass surgery was performed on July 13, and [PERSON_A] was discharged on July 20 with permanent right-sided paralysis and reduced language function. [PERSON_A] alleged that the explanation of the procedure's risks and the monitoring and response after it were insufficient.",
    ),
    "KR_a1301398c0525889ea": pair(
        "[PERSON_A]는 만 4세의 아동으로 보호자와 함께 [INSTITUTION_A]가 운영하는 수영시설을 방문하였다. [PERSON_A]는 보호자가 동반하지 않은 상태에서 구명조끼나 튜브를 착용하지 않고 수영장에 들어갔다. 시설 운영자와 현장 관리인들은 어린 이용자의 출입과 안전장비 착용을 충분히 확인하거나 위험을 막지 못하였다. [PERSON_A]는 물에 빠져 사망하였다. 보호자 측과 시설 측은 보호자의 주의 부족과 시설의 안전관리 부족이 사고에 각각 얼마나 기여했는지를 다투었다.",
        "[PERSON_A], a four-year-old child, visited a swimming facility operated by [INSTITUTION_A] with a caregiver. [PERSON_A] entered the pool without the caregiver and without a life jacket or flotation tube. The operator and on-site staff did not adequately monitor the young child's access or use of safety equipment or prevent the danger. [PERSON_A] drowned and died. The caregiver and the facility disputed the degree to which inadequate parental supervision and inadequate facility safety management each contributed to the death.",
    ),
    "KR_fd8f69f92c35adf0ac": pair(
        "2012년 6월 5일 당시 만 16세인 [PERSON_A]는 술에 취한 상태로 [VEHICLE_A]를 운전하다 횡단보도에 인접한 도로를 건너던 [PERSON_B]를 들이받았다. [PERSON_B]는 경부척수 손상으로 사지가 마비되는 등 중상을 입었다. [PERSON_A]의 부모인 [PERSON_C]와 [PERSON_D]가 미성년자의 운전과 차량 사용을 어떻게 감독했는지도 문제 되었다. 전체 치료비는 [CURRENCY_AMOUNT_A]였고, 그중 [ORGANIZATION_A]가 [CURRENCY_AMOUNT_B]를 부담하였다. [PERSON_B]가 횡단보도 바로 옆을 건넌 행위와 [PERSON_A]의 음주운전 및 보호자들의 감독 부재가 손해에 각각 기여한 정도가 다투어졌다.",
        "On June 5, 2012, [PERSON_A], then age sixteen, drove [VEHICLE_A] while intoxicated and struck [PERSON_B], who was crossing a roadway next to a marked crossing. [PERSON_B] sustained severe injuries, including paralysis of all four limbs from a cervical spinal-cord injury. The extent to which [PERSON_A]'s parents, [PERSON_C] and [PERSON_D], supervised the minor's driving and vehicle use was also disputed. Total treatment charges were [CURRENCY_AMOUNT_A], of which [ORGANIZATION_A] paid [CURRENCY_AMOUNT_B]. The parties disputed how much [PERSON_B]'s decision to cross beside the marked crossing, [PERSON_A]'s intoxicated driving, and the parents' lack of supervision each contributed to the harm.",
    ),
    "KR_6078abc75c440541ab": pair(
        "무면허인 [PERSON_A]는 [COMPANY_A]가 운영하는 대여점에서 [VEHICLE_A]를 빌려 [PERSON_B]를 태우고 운전하였다. [PERSON_A]가 운전 중 도로 경계석 등을 들이받아 [PERSON_A]는 왼쪽 대퇴골 골절 등을, [PERSON_B]는 안와 골절 등을 입었다. [PERSON_A]는 2008년 5월 1일부터 12월 27일까지, [PERSON_B]는 2008년 5월 1일부터 2009년 5월 25일까지 입원 및 통원치료를 받았다. [PERSON_A]의 치료비는 [CURRENCY_AMOUNT_A], [PERSON_B]의 치료비는 [CURRENCY_AMOUNT_B]였다. [COMPANY_A]가 면허가 없는 사람에게 [VEHICLE_A]를 대여한 행위와 [PERSON_A]의 운전 부주의 및 [PERSON_B]의 동승이 사고와 손해에 기여한 정도가 다투어졌다.",
        "Without a driver's license, [PERSON_A] rented [VEHICLE_A] from a rental business operated by [COMPANY_A] and drove it with [PERSON_B] as a passenger. [PERSON_A] struck a road curb or similar boundary, sustaining a left-femur fracture and other injuries, while [PERSON_B] sustained an orbital fracture and other injuries. [PERSON_A] received inpatient and outpatient treatment from May 1 through December 27, 2008, and [PERSON_B] from May 1, 2008, through May 25, 2009. Treatment charges were [CURRENCY_AMOUNT_A] for [PERSON_A] and [CURRENCY_AMOUNT_B] for [PERSON_B]. The parties disputed the respective contributions of [COMPANY_A]'s rental to an unlicensed person, [PERSON_A]'s inattentive driving, and [PERSON_B]'s decision to ride.",
    ),
    "KR_8f9a8d4ff8f0f379fb": pair(
        "2006년 10월 3일 07시 40분경 짙은 안개 속에서 25톤 [VEHICLE_A]가 앞서 서행하던 1톤 트럭을 들이받은 뒤 2차로에 멈췄고 후방 경고표지 등 안전조치가 이루어지지 않았다. 뒤따르던 승합차와 여러 승용차가 연이어 충돌했고, 일부 차량도 안전조치 없이 차로에 남았다. [PERSON_A]는 동승 차량에서 내려 3차로의 화재를 피해 1차로 쪽으로 이동하였다. 1차로를 진행하던 카캐리어가 [PERSON_A]의 발을 뒷바퀴로 충격해 골반골절과 오른쪽 다리 절단 등의 상해를 입혔다. 3차로에서는 탱크로리와 후속 차량들의 충돌로 화재가 발생해 다수의 사상자가 나왔다. 최초와 후속 운전자들의 전방주시 및 안전거리 부족과 정차 후 안전조치 부재가 [PERSON_A]의 대피와 후행 충돌에 얼마나 기여했는지가 다투어졌다.",
        "At about 7:40 a.m. on October 3, 2006, in dense fog, a 25-ton [VEHICLE_A] struck a slower one-ton truck, stopped in the second lane, and remained without a rear warning marker or other safety measures. A van and several passenger vehicles then collided in sequence, and some also remained in the travel lanes without warnings. [PERSON_A] exited one of the vehicles and moved toward the first lane to escape a fire in the third lane. A car carrier traveling in the first lane struck [PERSON_A]'s foot with a rear wheel, causing a pelvic fracture, amputation of the right leg, and other injuries. In the third lane, collisions involving a tanker and later vehicles caused a fire with multiple casualties. The parties disputed how the initial and later drivers' failures to keep lookout and following distance and their failure to secure stopped vehicles contributed to [PERSON_A]'s flight and the later impact.",
    ),
    "KR_de99291f16b6fe1b2a": pair(
        "2012년 한 주점에서 화재가 발생하였다. 2011년 6월경부터 주출입구 옆의 제2비상구 연결 통로에는 문이 설치되고 술 상자가 쌓여 사실상 창고로 사용되어 통행이 어려웠다. 화재 당시 생존자들은 주출입구를 통해 대피했지만, 사망자들은 주출입구나 제2비상구 통로 입구에 이르지 못한 채 내부 복도에서 유독가스를 흡입해 사망하였다. 제2비상구 표시도 그 전 복도에서는 발견하기 어려운 구조였다. 시설 관계자, 건물 안전관리자와 소방 담당자들이 비상통로, 조명, 차단장치, 훈련 및 점검을 적절히 관리했는지와, 제2비상구가 열려 있었더라도 사망을 막을 수 있었는지가 다투어졌다.",
        "A fire occurred in a drinking establishment in 2012. Since about June 2011, a door and stacked beverage boxes had turned the passage connecting to a second emergency exit beside the main entrance into de facto storage and made passage difficult. During the fire, survivors escaped through the main entrance, but the people who died did not reach either the main entrance or the passage to the second exit and died after inhaling toxic gases in an interior corridor. The layout also made the second-exit sign difficult to see before reaching that corridor. The parties disputed whether the establishment operators, building safety manager, and fire-safety personnel adequately maintained the exit route, lighting, shutoff devices, training, and inspections, and whether an open second exit would have prevented the deaths.",
    ),
    "KR_46eecae742ccab2781": pair(
        "2014년 10월 2일 [PERSON_A]는 허리 통증으로 [INSTITUTION_A] 응급실에 갔다. [PERSON_B]는 자기공명영상에서 요추 4-5번 척추관 협착증과 좌측 추간판 탈출증을 확인하고, 휴일 동안 수술 없이 증상 완화 치료만 가능하다고 설명한 뒤 [PERSON_A]를 집 근처 [INSTITUTION_B]로 옮겼다. 영상 판독에는 흉추 12번부터 요추 1번에 걸친 상당량의 척추 경막외 혈종과 중등도 이상의 척수 압박도 기록되어 있었지만, [PERSON_B]가 작성한 전원 문서에는 협착증과 추간판 탈출증만 적혀 있었다. [INSTITUTION_B]의 기록에도 혈종은 기재되지 않았다. [PERSON_A]는 통증 조절 치료를 받다가 10월 4일 통증이 심해지고 다리 마비가 나타났다. 10월 6일 [INSTITUTION_A]로 돌아왔을 때 출혈은 흉추 9번부터 12번까지 확대되어 있었고, 혈종 제거수술 중 출혈성 약물과 관련된 대량 출혈도 발생하였다. 이후 [PERSON_A]는 서거나 걸을 수 없는 영구적인 하지마비 상태가 되었다. 혈종을 조기에 확인해 약물·응고상태를 검사하고 세밀히 관찰했는지, 전원받은 의료진과 환자·보호자에게 혈종과 응급수술 가능성을 충분히 알렸는지, 더 빠른 수술이 마비를 막을 수 있었는지가 다투어졌다.",
        "On October 2, 2014, [PERSON_A] went to the emergency department of [INSTITUTION_A] with lower-back pain. [PERSON_B] identified spinal-canal stenosis at lumbar levels 4-5 and a left-sided herniated disc on magnetic-resonance images and, after explaining that only symptom-relief treatment would be available during the holiday period, transferred [PERSON_A] to nearby [INSTITUTION_B]. The imaging report also recorded a substantial spinal epidural hematoma from thoracic level 12 to lumbar level 1 with at least moderate spinal-cord compression, but [PERSON_B]'s transfer document listed only the stenosis and herniated disc. [INSTITUTION_B]'s record likewise did not mention the hematoma. While receiving pain-control treatment, [PERSON_A] developed worse pain and leg paralysis on October 4. When [PERSON_A] returned to [INSTITUTION_A] on October 6, the bleeding extended from thoracic levels 9 through 12, and heavy bleeding associated with a blood-thinning medication occurred during surgery to remove the hematoma. [PERSON_A] was left with permanent lower-body paralysis and could no longer stand or walk. The parties disputed whether the hematoma was identified early enough for medication and coagulation checks and close monitoring, whether the receiving clinicians and the patient and family were adequately told about the hematoma and possible emergency surgery, and whether earlier surgery could have prevented the paralysis.",
    ),
}


CORRECTIONS = {
    "US_15e86d45f19b975195": pair(
        "2010년 12월 [PERSON_A]와 [PERSON_B]는 손자녀 [PERSON_C]의 후견을 맡았고, [PERSON_C]는 두 사람과 [LOCATION_A]에서 살았다. [ORGANIZATION_A]가 작성한 초기 문서에는 [PERSON_F]가 다른 자녀 앞에서 폭력을 행사하고 자녀들에게 필요한 위생, 음식, 의복, 감독과 주거를 제공하지 않았다는 내용이 있었고, [PERSON_C]의 주소로 [LOCATION_A]가 적혀 있었지만 [PERSON_A]와 [PERSON_B]가 후견인이라는 점은 표시되지 않았다. 같은 날 [INSTITUTION_A]는 자녀들을 [ORGANIZATION_A]가 임시로 보호하도록 정했으나 [PERSON_C]는 계속 두 후견인과 지냈다. 나흘 뒤 [ORGANIZATION_A] 직원들은 [PERSON_A]와 [PERSON_B]에게 [PERSON_C]를 데리고 예정된 모임에 오라고 했고, 도착하자 [PERSON_C]를 두 사람에게서 데려가 보호하기 시작했다. [PERSON_A]와 [PERSON_B]는 [ORGANIZATION_A]가 자신들의 후견관계와 [PERSON_C]가 집에서 적절히 돌봄받고 있다는 사실을 알고도 이를 초기 문서에서 빠뜨리고, 보호 변경 계획을 분명히 알리지 않은 채 아이를 데려오게 하여 분리와 정서적 손상을 초래했다고 주장하였다. [ORGANIZATION_A]는 집에 관한 학대·방임 정보와 임시 보호 변경을 두 사람에게 알렸으며, 두 사람이 이후 관련 문서를 받고 의견을 제출할 기회도 가졌다고 다투었다.",
        "In December 2010, [PERSON_A] and [PERSON_B] became guardians of their grandchild, [PERSON_C], who lived with them at [LOCATION_A]. An initial document prepared by [ORGANIZATION_A] alleged that [PERSON_F] used violence in front of another child and failed to provide the children with necessary hygiene, food, clothing, supervision, and housing. The document listed [LOCATION_A] as [PERSON_C]'s address but did not identify [PERSON_A] and [PERSON_B] as guardians. The same day, [INSTITUTION_A] authorized [ORGANIZATION_A] to take temporary care of the children, although [PERSON_C] continued living with the two guardians. Four days later, personnel from [ORGANIZATION_A] directed [PERSON_A] and [PERSON_B] to bring [PERSON_C] to a scheduled meeting; when they arrived, [ORGANIZATION_A] removed [PERSON_C] from them and began providing custody. [PERSON_A] and [PERSON_B] alleged that [ORGANIZATION_A] knew of their guardianship and knew [PERSON_C] was receiving adequate care in their home, yet omitted those facts from the initial document and induced them to bring the child without clearly disclosing the planned custody change, causing separation and emotional harm. [ORGANIZATION_A] disputed this, stating that it told them about abuse and neglect information concerning the home and the temporary custody change and that they later received the relevant documents and had an opportunity to respond.",
    ),
    "KR_7f8df551c38264fcbe": pair(
        "소년인 [PERSON_A], [PERSON_C], [PERSON_F], [PERSON_I]를 포함한 10명은 [ORGANIZATION_A] 소속 조사관의 조사 방식으로 정신적 고통을 입었다고 주장하였다. [PERSON_K]는 [PERSON_C]와 [PERSON_F]에게 범행의 시기·장소·준비과정과 세부내용을 구체적으로 질문한 뒤, 이들이 한 짧은 답변을 자발적으로 이어진 구체적 진술처럼 보이도록 문답 내용을 바꾸어 기록하였다. 그 기록은 이후의 심문과 조사에서 네 소년이 질문 내용과 기록된 답변의 차이를 해명하고 대응하는 데 불리하게 작용하였다. 이들은 조사 때 신뢰하는 성인의 동석이 배제되었고, 답변하지 않거나 조력자를 참여시킬 수 있다는 설명과 실질적인 기회도 충분하지 않았다고 추가로 주장하였다. [ORGANIZATION_A]는 기록 변경 외의 추가 조사방식과 그로 인한 손해에 관한 주장들을 다투었다.",
        "Ten affected people, including the juveniles [PERSON_A], [PERSON_C], [PERSON_F], and [PERSON_I], reported emotional distress from the manner in which an investigator employed by [ORGANIZATION_A] conducted interviews. After [PERSON_K] asked [PERSON_C] and [PERSON_F] specific questions about the time, place, preparation, and details of the suspected conduct, [PERSON_K] rewrote their short answers so that the record appeared to contain a detailed, continuous, voluntary account. During later questioning and investigation, the discrepancy between the questions, the short answers, and the written account made it harder for the four juveniles to explain and respond. They further alleged that a trusted adult was excluded and that they were not adequately told or given a meaningful opportunity to decline to answer or to have an adviser present. [ORGANIZATION_A] disputed the additional allegations about the interview process and the resulting harm beyond the altered records.",
    ),
    "US_4bf2e6511806b25f10": pair(
        "[PERSON_A]와 [PERSON_C]는 3년간 관계를 유지하였다. 관계가 끝나갈 무렵 임신한 [PERSON_A]는 임신이 인공수정으로 이루어졌고 [PERSON_C]는 2001년 3월 태어난 [PERSON_E]의 아버지가 아니라고 말했다. 2010년 8월 [PERSON_C]는 친자관계를 확인하는 절차를 시작했고, DNA 검사 결과 자신이 아버지임을 알게 되어 2011년 9월 11일 이를 공개하였다. [PERSON_C]는 [PERSON_A]가 친자관계를 알고도 숨기고 [PERSON_E]와의 관계 형성을 방해했다고 주장하였다. [PERSON_A]는 이 분쟁에 대응하도록 [PERSON_B]를 선임하였다. [PERSON_B]는 [PERSON_C]가 친자관계를 의심하거나 알게 된 시점을 뒷받침하는 자료를 초기 단계에 충분히 제시하지 않았고, 그 시점에 관한 [PERSON_C]의 선서진술도 확보하지 않았다. [PERSON_A]는 이러한 누락 때문에 비용이 늘고 재정적·정서적 손해를 입었다고 주장하였다. [PERSON_B]는 해당 시점의 문제를 계속 제기할 수 있도록 남겨 두었고 자신의 조치가 불리한 결과를 초래하지 않았다고 다투었다.",
        "[PERSON_A] and [PERSON_C] had a three-year relationship. Near its end, [PERSON_A] became pregnant and told [PERSON_C] that the pregnancy resulted from artificial insemination and that he was not the father of [PERSON_E], who was born in March 2001. In August 2010, [PERSON_C] began a process to determine paternity; DNA testing showed that he was the father, and he announced that fact on September 11, 2011. [PERSON_C] alleged that [PERSON_A] knowingly concealed the paternity and interfered with his efforts to form a relationship with [PERSON_E]. [PERSON_A] retained [PERSON_B] to respond to the dispute. At the initial stage, [PERSON_B] did not sufficiently present material showing when [PERSON_C] suspected or learned of the paternity and did not obtain [PERSON_C]'s sworn account of that timing. [PERSON_A] alleged that those omissions increased her expenses and caused financial and emotional harm. [PERSON_B] disputed causation, stating that the timing issue remained available and that her actions did not cause the adverse result.",
    ),
    "KR_cda2f7ae1f9162201f": pair(
        "1997년 2월 3일 18시경 [COMPANY_A] 소속 주차관리원인 [PERSON_A]는 [LOCATION_A]에 세워진 [COMPANY_B] 소유의 [VEHICLE_A]를 옮기려고 시동을 켜고 변속 선택레버를 주차에서 전진으로 이동하였다. 직후 [VEHICLE_A]가 갑자기 앞으로 나아가 다른 주차차량들과 [PROPERTY_A]를 연이어 충격해 여러 차량과 시설 일부가 파손되었다. [VEHICLE_A]는 [COMPANY_C]가 1996년에 제조하였다. 사고 전에는 엔진, 변속장치, 브레이크나 전자제어장치의 이상 및 급가속 이력이 없었고, 사고 후 점검에서도 부품 이상이 발견되지 않았다. [PERSON_A]는 엔진제어장치의 설치 위치, 전자파 차폐장치 부재 및 전자파 시험 부재 때문에 차량이 갑자기 움직였다고 주장하였다. [COMPANY_C]는 [PERSON_A]가 변속 과정에서 액셀러레이터를 밟았을 가능성을 제시하였다. 사용설명서에는 페달 위치를 확인하고 브레이크를 밟은 상태에서 시동과 변속을 하라는 지시가 있었다.",
        "At about 6:00 p.m. on February 3, 1997, [PERSON_A], a parking attendant employed by [COMPANY_A], entered [VEHICLE_A], owned by [COMPANY_B] and parked at [LOCATION_A], started it, and moved the transmission selector from park to drive. [VEHICLE_A] immediately moved forward, struck other parked vehicles and [PROPERTY_A] in sequence, and damaged several vehicles and part of the facility. [COMPANY_C] had manufactured [VEHICLE_A] in 1996. Before the incident, there was no known problem with the engine, transmission, brakes, or electronic control unit and no prior sudden-acceleration event; a post-incident inspection likewise found no component abnormality. [PERSON_A] alleged that the placement of the engine-control unit, absence of added electromagnetic shielding, and lack of electromagnetic-interference testing caused the movement. [COMPANY_C] raised the possibility that [PERSON_A] pressed the accelerator while shifting. The owner's manual instructed the driver to check pedal positions and keep the brake depressed while starting and shifting.",
    ),
    "US_ca3855b0593ebfc33e": pair(
        "[COMPANY_A]는 세르트랄린 염산염을 항우울제 [PRODUCT_A]로 제조·판매하고, [COMPANY_C]는 그 제네릭 제품을 판매한다. [PERSON_B]가 임신 중 [PRODUCT_A]를 복용한 뒤 [PERSON_A]에게 자궁 내 손상과 선천적 결함이 발생했다고 주장되었다. 2009년 제품표시에는 임신 중 사용과 선천적 결함 위험 증가의 관련성이나 복용 중 피임 필요성에 관한 문구가 없었다. [PERSON_A]는 [COMPANY_A]가 임신 중 사용 위험을 시사하는 보고서를 보유하고 다른 지역 판매 포장에는 피임 관련 문구를 넣으면서도 해당 표시에는 넣지 않았다고 주장하였다. [COMPANY_A]는 [PRODUCT_A]가 [INSTITUTION_A]의 승인을 받았고 요구된 표시 기준을 따랐으며, 복용과 선천적 결함 사이의 관련성도 확정되지 않았다고 다투었다.",
        "[COMPANY_A] manufactures and markets sertraline hydrochloride as the antidepressant [PRODUCT_A], and [COMPANY_C] sells a generic version. It was alleged that after [PERSON_B] took [PRODUCT_A] during pregnancy, [PERSON_A] sustained in-utero injury and resulting birth defects. The 2009 product label did not state that use during pregnancy was associated with an increased risk of birth defects or that contraception should be used while taking the medication. [PERSON_A] alleged that [COMPANY_A] possessed reports suggesting risks from use during pregnancy and included contraception language on packaging sold elsewhere but omitted it from the relevant label. [COMPANY_A] disputed this, noting that [PRODUCT_A] had been approved by [INSTITUTION_A], that the required labeling specifications were followed, and that an association between exposure and the birth defects had not been established.",
    ),
    "US_512d2cb96959fe1764": pair(
        "2002년 7월 21일 [PERSON_A]는 [PERSON_B]가 운전하는 [VEHICLE_A]에 승객으로 타고 있었다. [PERSON_B]는 술에 취한 채 과속하다 차량을 도로 밖으로 벗어나게 했고, [VEHICLE_A]는 전복되어 여러 차례 굴렀다. 충돌 당시 [PERSON_B]의 혈중알코올농도는 0.31%였고 체내에서 코카인 대사물질도 검출되었다. [PERSON_A]는 척추뼈 두 개, 갈비뼈 네 개, 손목과 쇄골 골절을 입었고, 손·팔·다리에 중증 장애가 남는 불완전 사지마비 상태가 되었다. [PERSON_B]는 [PERSON_A]가 음주와 과속을 알면서도 탑승한 행위도 손해 발생에 기여했다고 주장하였다.",
        "On July 21, 2002, [PERSON_A] was a passenger in [VEHICLE_A], driven by [PERSON_B]. While intoxicated and speeding, [PERSON_B] drove off the road, and [VEHICLE_A] overturned and rolled several times. At the time of the crash, [PERSON_B]'s blood-alcohol concentration was 0.31 percent and cocaine metabolite was present in his system. [PERSON_A] sustained fractures of two vertebrae, four ribs, a wrist, and a collarbone and was left with incomplete quadriplegia and severe impairment of the hands, arms, and legs. [PERSON_B] alleged that [PERSON_A]'s decision to ride despite knowing about the alcohol use and speeding also contributed to the harm.",
    ),
    "US_0ed0e9c92a0a12ceca": pair(
        "2002년 2월 4일 [PERSON_A]가 [VEHICLE_A]를 운전하던 중 차량이 통제력을 잃고 전복되어 [PERSON_A]가 다쳤다. 사고 몇 시간 안에 [PERSON_B]는 보험사 [COMPANY_B]에 알렸다. [COMPANY_B]는 차량을 검사해 전손으로 정하고 소유권을 넘겨받아 [CURRENCY_AMOUNT_A]를 지급한 뒤, 2002년 4월 11일 해체업체 [COMPANY_D]에 매각하였다. 차량은 분해되어 부품과 고철로 처분되었고 서스펜션 일부도 없어졌다. [PERSON_A]와 [PERSON_B]는 서스펜션 설계 때문에 차량이 불안정하고 전복 저항성이 부족했으며, 부품 처분으로 그 원인을 검사할 기회를 잃었다고 주장하였다. [COMPANY_B]는 매각 전 보존 요청이나 결함 조사 계획을 전달받지 못했다고 말했다. [COMPANY_B]는 이전 10년 동안 같은 모델과 관련된 약 500건을 처리해 총 [CURRENCY_AMOUNT_B]를 지급한 기록이 있었지만, 그 기록이 각 전복의 원인을 보여주는지는 다투어졌다.",
        "On February 4, 2002, [PERSON_A] was driving [VEHICLE_A] when it went out of control and rolled over, injuring [PERSON_A]. Within hours, [PERSON_B] notified the insurer, [COMPANY_B]. [COMPANY_B] inspected the vehicle, declared it a total loss, obtained title, paid [CURRENCY_AMOUNT_A], and sold it to dismantler [COMPANY_D] on April 11, 2002. The vehicle was broken apart for parts and scrap, and some suspension components disappeared. [PERSON_A] and [PERSON_B] alleged that the suspension design made the vehicle unstable and insufficiently resistant to rollover and that disposal of the parts eliminated an opportunity to examine the cause. [COMPANY_B] said it had received no preservation request or notice of a planned defect investigation before the sale. Records showed that [COMPANY_B] had handled about 500 matters involving the same model during the preceding ten years and paid a total of [CURRENCY_AMOUNT_B], but the parties disputed whether those records showed the causes of the prior rollovers.",
    ),
    "US_193fdb5b19590cbcab": pair(
        "1996년 3월 26일 [COMPANY_A] 소속 검사실 기술자 [PERSON_C]는 전혈구 검사를 위해 [PERSON_A]의 왼팔에서 채혈하였다. [PERSON_A]는 주사침 삽입 때 극심한 통증이 있었다고 말했고, 4월 1일에는 왼손의 멍, 통증과 무감각으로 다시 진료받아 왼팔 혈종 진단을 받았다. 4월 8일에도 통증과 무감각이 계속되어 혈관외과 진료와 신경검사를 받았다. 1996년 8월 21일 왼쪽 전완 신경유리술과 손목터널 유리술을 받았지만 증상이 계속되었고, 1997년 3월에는 왼팔이 자줏빛을 띠고 차가워졌다. 1997년 7월 [PERSON_A]는 복합부위통증증후군 진단을 받았다. [PERSON_A] 측 의료인은 주사침이 신경을 손상시켜 증후군이 발생했을 가능성을 제시한 반면, 다른 의료인들은 주사침 손상의 객관적 징후가 없고 후속 손목수술이나 다른 상태가 원인일 수 있다고 의견을 밝혔다.",
        "On March 26, 1996, [PERSON_C], a laboratory technician employed by [COMPANY_A], drew blood from [PERSON_A]'s left arm for a complete blood count. [PERSON_A] reported excruciating pain when the needle was inserted and returned on April 1 with bruising, pain, and numbness in the left hand, receiving a diagnosis of a left-arm hematoma. Pain and numbness continued on April 8, leading to a vascular consultation and nerve testing. On August 21, 1996, [PERSON_A] underwent nerve-release surgery in the left forearm and carpal-tunnel-release surgery, but symptoms persisted, and by March 1997 the left arm was purplish and cold. In July 1997, [PERSON_A] was diagnosed with complex regional pain syndrome. A clinician for [PERSON_A] said a needle injury to a nerve could have caused the syndrome, while other clinicians said there was no objective sign of such an injury and that the later wrist surgery or another condition could have been the cause.",
    ),
    "KR_5ebeddec620aad3412": pair(
        "[PERSON_B]는 [VEHICLE_A]를 타고 [LOCATION_A]의 우측 차로 오른쪽 부분을 [LOCATION_B]에서 [LOCATION_C] 방향으로 시속 약 30km로 진행했고, [PERSON_A]는 [VEHICLE_B]를 타고 가까운 뒤쪽의 같은 차로 왼쪽 부분을 진행하였다. [PERSON_B]는 [LOCATION_D]로 빠져나가기 위해 갑자기 핸들을 왼쪽으로 틀어 도로를 가로질렀다. [PERSON_A]는 충돌을 피하려고 급정지하다 [VEHICLE_B]와 함께 도로 오른쪽으로 넘어져 척골 상단 골절 등의 상해를 입었다. [PERSON_B]는 좌회전 전 도로 왼쪽으로 미리 이동하거나 수신호로 진행방향을 알리지 않았고 가까운 뒤쪽도 확인하지 않았다.",
        "[PERSON_B] rode [VEHICLE_A] at about 30 km/h along the right side of the right lane of [LOCATION_A] from [LOCATION_B] toward [LOCATION_C], while [PERSON_A] followed closely on [VEHICLE_B] along the left side of the same lane. To exit toward [LOCATION_D], [PERSON_B] suddenly turned left across the road. [PERSON_A] braked abruptly to avoid a collision, fell with [VEHICLE_B] to the right side of the road, and sustained an upper-ulna fracture and other injuries. Before turning, [PERSON_B] did not move toward the left side of the road, signal the intended direction by hand, or check the area close behind.",
    ),
    "US_3f0101285fcc76e2b1": pair(
        "[PERSON_A]는 1947년부터 1979년까지 [COMPANY_C]에서 기계공으로 일하며 [PRODUCT_A]에 노출되었고 그 결과 흉막삼출과 폐 실질 흉터가 생겼다고 주장하였다. [COMPANY_A]는 음료 캔 제조업체로 1963년 11월 [COMPANY_B] 주식의 과반수를 취득하였다. [COMPANY_B]에는 [PRODUCT_B]를 제조·판매·설치하는 사업부가 있었지만, [COMPANY_A]는 그 사업부를 운영하지 않고 주식 취득 90일 뒤 매각하였다. [COMPANY_A]는 1996년 2월 10일 [COMPANY_B]의 주식 전부를 취득해 합병하였다. [COMPANY_A]는 자신이 [PRODUCT_A]를 제조·유통·판매한 적이 없고, 과거 단기간 [COMPANY_B]를 보유했을 때에도 [COMPANY_B]가 [PRODUCT_A]를 생산하지 않았다고 주장하였다.",
        "[PERSON_A] alleged exposure to [PRODUCT_A] while working as a machinist for [COMPANY_C] from 1947 through 1979 and resulting pleural effusion and parenchymal lung scarring. [COMPANY_A], a beverage-can manufacturer, acquired a majority of [COMPANY_B]'s stock in November 1963. [COMPANY_B] had a division that manufactured, sold, and installed [PRODUCT_B], but [COMPANY_A] did not operate that division and sold it ninety days after acquiring the stock. [COMPANY_A] acquired all of [COMPANY_B]'s stock and merged with it on February 10, 1996. [COMPANY_A] maintained that it had never manufactured, distributed, or sold [PRODUCT_A] and that [COMPANY_B] did not produce [PRODUCT_A] during the earlier short period of ownership.",
    ),
    "US_78f2e4c0127c877a51": pair(
        "다섯 살인 [PERSON_A]의 부모는 공기주입식 놀이기구가 있는 실내 놀이시설 [COMPANY_A]에서 생일파티를 열었다. 파티 전 아버지 [PERSON_B]는 [PERSON_A]를 참가자로 적은 위험고지·면제 양식에 부모 자격으로 서명하였다. 파티 중 [PERSON_A]가 미끄럼틀에서 뛰어내리다 한쪽 다리가 부러졌다. 양식에는 놀이 참여에 신체 상해 등의 위험이 수반된다고 적혀 있었고, [COMPANY_A]는 [PERSON_B]가 그 위험을 알고 동의했다고 주장하였다. [PERSON_A] 측은 어린 참가자의 안전을 확보할 책임까지 사라진 것은 아니라고 다투었다.",
        "The parents of five-year-old [PERSON_A] held a birthday party at [COMPANY_A], an indoor play facility with inflatable equipment. Before the party, [PERSON_B], the child's father, signed a risk-notice and release form as a parent, with [PERSON_A] listed as the participant. During the party, [PERSON_A] jumped from a slide and broke one leg. The form stated that participation carried risks including bodily injury, and [COMPANY_A] said [PERSON_B] knew and accepted those risks. [PERSON_A]'s side disputed that the form eliminated the facility's responsibility to protect a young participant.",
    ),
}


def sanitize(text: str, language: str) -> str:
    if language == "en":
        substitutions = (
            (r"\bThe plaintiffs were\b", "The affected people were"),
            (r"\bPlaintiffs were\b", "The affected people were"),
            (r"\bThe plaintiffs alleged that\b", "It was alleged that"),
            (r"\bPlaintiffs alleged that\b", "It was alleged that"),
            (r"\bThe plaintiff alleged that\b", "It was alleged that"),
            (r"\bThe plaintiff's side alleged that\b", "It was alleged that"),
            (r"\bThe complaint alleges that\b", "It was alleged that"),
            (r"\bThe complaint alleged that\b", "It was alleged that"),
            (r"\bThe defendants included\b", "The involved entities included"),
            (r"\bdefendants' respective\b", "the involved companies' respective"),
            (r"\bthe defendants\b", "the involved entities"),
            (r"\bthe defendant\b", "the involved entity"),
            (r"\bthe plaintiffs\b", "the affected people"),
            (r"\bthe plaintiff\b", "the affected person"),
            (r"\bdefendants\b", "involved entities"),
            (r"\bplaintiffs\b", "affected people"),
            (r"\bdefendant\b", "involved entity"),
            (r"\bplaintiff\b", "affected person"),
            (r"\bPlaintiffs\b", "The affected people"),
            (r"\bplaintiffs\b", "affected people"),
            (r"\bfiled suit\b", "sought redress"),
            (r"\ba lawsuit\b", "the dispute"),
            (r"\bthe lawsuit\b", "the dispute"),
            (r"\bthe complaint\b", "the account"),
            (r"\ban amended complaint\b", "an amended account"),
            (r"\bcomplaint\b", "report"),
            (r"\blawsuit\b", "dispute"),
            (r"\bThe petition alleged that\b", "It was alleged that"),
            (r"\bThe initial petition alleged that\b", "It was alleged that"),
            (r"\bthe petition alleged that\b", "it was alleged that"),
            (r"\bThe petition alleged\b", "It was alleged"),
            (r"\bthe petition alleged\b", "it was alleged"),
            (r"\ba claim petition seeking workers' compensation benefits\b", "a request for workplace-injury benefits"),
            (r"\bproducts liability claim\b", "assertion that the vehicle was defective"),
            (r"\bcivil claims\b", "requests for redress"),
            (r"\bgreater than any negligent conduct by Officer \[PERSON_B\]\b", "more significant than Officer [PERSON_B]'s conduct"),
            (r"\balleged malpractice\b", "allegedly inadequate professional work"),
            (r"\bnegligently failed to include\b", "failed to include"),
            (r"\bthe portion attributable to \[PERSON_B\]’s negligence\b", "the portion assigned to [PERSON_B]’s own conduct"),
            (r"\bmay invoke the defenses and limitations of liability that the carrier may assert\b", "receive the same contractual protections and payment limits as the carrier"),
            (r"\bstate government\b", "public administration"),
            (r"\bstate health department\b", "public health agency"),
            (r"\bformer city emergency managers\b", "formerly appointed managers"),
            (r"\bstate officials\b", "public officials"),
            (r"\bcity officials\b", "local public officials"),
            (r"\bstate and city involved entities\b", "public entities"),
            (r"\bstate involved entities\b", "public entities"),
            (r"\bcity involved entities\b", "local public entities"),
        )
    else:
        substitutions = (
            (r"원고 측은", "관련 당사자들은"),
            (r"원고들은", "관련 당사자들은"),
            (r"원고는", "관련 당사자는"),
            (r"피고들에는", "관련 기관들에는"),
            (r"피고에는", "관련 기관에는"),
            (r"피고들은", "관련 기관들은"),
            (r"피고는", "관련 기관은"),
            (r"피고들의", "관련 기관들의"),
            (r"피고들이", "관련 기관들이"),
            (r"피고 측", "관련 기관 측"),
            (r"원고 측", "관련 피해자 측"),
            (r"원고들의", "관련 피해자들의"),
            (r"원고들이", "관련 피해자들이"),
            (r"원고 1 외 2인은", "관련 가족 3인은"),
            (r"원고 6명은", "관련 가족 6명은"),
            (r"다른 2명은", "다른 가족 2명은"),
            (r"소송의 증거로", "원인 조사 자료로"),
            (r"소송을 제기했다", "배상을 요구했다"),
            (r"소송을 제기했다", "배상을 요구했다"),
            (r"집단소송 합의", "집단적 합의"),
            (r"소송이라는 제목", "분쟁이라는 제목"),
            (r"제기한 소송", "제기한 분쟁"),
            (r"피고 본인신문 결과 등에 따르면", "제공된 진술에 따르면"),
            (r"관련 피해자 측 관련자는 원고 1 외 15인이고", "관련 피해자 측은 16명이고"),
            (r"주 정부", "공공행정기관"),
            (r"주 보건부", "공중보건기관"),
            (r"전직 시 비상관리자", "전직 임명 관리자"),
            (r"주 공무원", "공공기관 관계자"),
            (r"시 공무원", "지역 공공기관 관계자"),
            (r"주 피고들과 시 관련 기관들", "공공기관들"),
            (r"주 관련 기관들이 시 피고들에게", "공공기관들이 지역 기관들에게"),
            (r"피고들의 행위", "관련 기관들의 행위"),
            (r"민사 청구", "피해구제 요구"),
            (r"제조물 책임 청구를 제기하고 입증할 능력", "차량 결함 주장을 조사하고 뒷받침할 능력"),
            (r"\[PERSON_B\]의 과실부분", "[PERSON_B] 본인의 행동에 배정된 부분"),
            (r"항변 및 책임제한을 원용할 수", "동일한 계약상 보호와 지급한도를 적용받을 수"),
            (r"과실로 주장된 행위", "부적절했다고 주장된 업무처리"),
            (r"과실로 포함하지 않았다고", "포함하지 않았다고"),
            (r"고소장에는", "제공된 설명에는"),
            (r"이 사건 당사자 중에는", "관련자에는"),
            (r"사건 당사자는", "관련자는"),
            (r"당사자는", "관련자는"),
        )
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE if language == "en" else 0)
    if language == "ko":
        text = text.replace(
            "관련 피해자 측 관련자는 원고 1 외 15인이고, 관련 기관 측 관련자는 [COMPANY_A] 외 1인이다.",
            "관련 피해자는 16명이고, 관련 기업은 [COMPANY_A] 외 1곳이다.",
        )
    else:
        text = text.replace(
            "The parties on the affected person side are affected person 1 and 15 others, and the parties on the involved entity side are [COMPANY_A] and one other.",
            "The affected group comprised sixteen people, and the involved businesses were [COMPANY_A] and one other entity.",
        )
    # Remove a few purely procedural identification sentences left by the v3 master.
    text = re.sub(r"(?i)(?:^|(?<=[.!?])\s+)One party to this case is .*?\.\s*", "", text)
    text = re.sub(r"(?:^|(?<=다\.))\s*이 사건의 한 당사자는 .*?다\.\s*", "", text)
    return " ".join(text.split())


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    cases = list(read_jsonl(WORK / "provisional_cases_200_v4.jsonl"))
    old = {row["case_id"]: row for row in read_jsonl(OUT / "final_fact_patterns_200_v3.jsonl")}
    case_ids = {row["case_id"] for row in cases}
    replacement_ids = {row["case_id"] for row in cases if row.get("replacement_status") == "v4_targeted_replacement"}
    if replacement_ids != set(NEW_FACTS):
        raise RuntimeError({"missing_new_facts": sorted(replacement_ids - set(NEW_FACTS)), "extra_new_facts": sorted(set(NEW_FACTS) - replacement_ids)})

    facts = []
    units = []
    for case in sorted(cases, key=lambda row: row["case_id"]):
        case_id = case["case_id"]
        if case_id in NEW_FACTS:
            ko, en = NEW_FACTS[case_id]
            status = "v4_source_reextracted_replacement"
        elif case_id in CORRECTIONS:
            ko, en = CORRECTIONS[case_id]
            status = "v4_documented_correction"
        else:
            source = old[case_id]
            ko, en = source["neutral_fact_ko"], source["neutral_fact_en"]
            status = "v4_preserved_reviewed_fact"
        ko = sanitize(ko, "ko")
        en = sanitize(en, "en")
        source_language = "ko" if case["origin_country"] == "KR" else "en"
        master = ko if source_language == "ko" else en
        row = {
            "case_id": case_id,
            "origin_country": case["origin_country"],
            "origin_state": case.get("origin_state"),
            "primary_domain": case["primary_domain"],
            "case_domain": case["primary_domain"],
            "liability_theories": case.get("liability_theories") or [],
            "source_language": source_language,
            "neutral_fact_ko": ko,
            "neutral_fact_en": en,
            "neutral_fact_source": master,
            "neutral_fact_ko_sha256": sha(ko),
            "neutral_fact_en_sha256": sha(en),
            "neutral_fact_source_sha256": sha(master),
            "translation_equivalence_status": "v4_bilingual_content_preserved_or_directly_rewritten",
            "source_grounding_status": "direct_controlling_source_repair" if status != "v4_preserved_reviewed_fact" else "preserved_v3_reviewed_master_rechecked_against_final_source",
            "text_review_provenance": status,
            "replacement_status": case.get("replacement_status", "retained"),
            "corpus_version": VERSION,
        }
        facts.append(row)
        units.append({
            "case_id": case_id,
            "fact_id": "V4_CANONICAL_NEUTRAL_FACT",
            "fact_type": "canonical_bilingual_neutral_fact",
            "text": master,
            "neutral_ko": ko,
            "neutral_en": en,
            "source_span": None,
            "source_grounding_status": row["source_grounding_status"],
            "epistemic_status": "directly_source_repaired" if status != "v4_preserved_reviewed_fact" else "preserved_and_reaudited",
            "include_in_neutral_fact": True,
            "origin_country": case["origin_country"],
            "analysis_split": None,
            "corpus_version": VERSION,
        })

    if len(facts) != 200 or {row["case_id"] for row in facts} != case_ids:
        raise RuntimeError("Final fact roster does not match final case roster")
    write_jsonl(WORK / "final_fact_patterns_200_v4.jsonl", facts)
    write_jsonl(WORK / "final_fact_units_200_v4.jsonl", units)
    print(json.dumps({"facts": len(facts), "new": len(NEW_FACTS), "corrected": len(CORRECTIONS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
