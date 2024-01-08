import os
import pandas as pd
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema import HumanMessage, AIMessage

# Azure OpenAI를 사용하는 객체 생성
chat = AzureChatOpenAI(
    deployment_name='배포이름',
    model_name='모델이름',
    temperature=0.5
)



# 기본 프롬프트 템플릿 생성
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
# 예시 데이터
examples = [
    {"input": "비효과적 호흡 양상 환자의 간호기록을 SOAPIE 양식으로 작성해줘.", "output": "환자 정보:\n 이름: 이민호\n 나이: 45세\n 성별: 남성\n\nSOAPIE 기록:\n\nS: 주관적 데이터 (Subjective):\n 환자는 최근에 기침과 호흡 곤란을 느끼며 "'가슴이 무겁고 숨쉬기가 어려워'"라고 언급함.\n 특성: 허파 소리가 어렵게 들리며   호흡 속도가 증가하고   부종에 따른 발기부전 증상이 나타남.\n\nO: 객관적 데이터 (Objective):\n 호흡 속도가 증가하고   흉통이 있음.\n 산소 포화도는 90%로 감소하였고   기침 소리와 숨소리가 청취됨.\n\nA: 평가 (Assessment):\n 환자는 현재 비효과적 호흡 양상을 나타냄.\n\nP: 계획 (Plan): \n1. 산소 요법 시작.\n2. 기침 및 호흡 훈련 시행.\n3. 폐 활성화를 위한 포지셔닝 변경.\n4. 필요 시 의사에게 의견 구함.\n\nI: 간호 중재 (Intervention):\n1. 산소 치료 시작.\n2. 호흡 훈련 및 기침 도움을 위한 간호 중재.\n3. 환자를 일정한 각도로 반전시켜 폐 활성화.\n4. 의사에게 상황 보고.\n\nE: 평가 (Evaluation):\n1. 산소 요법으로 산소 포화도가 95%로 상승함.\n2. 호흡 훈련으로 호흡 양상이 개선됨.\n3. 폐 활성화를 위한 포지셔닝이 효과적임.\n4. 의사의 조언에 따라 추가 조치가 필요함."},
    {'input': "비효과적 기도 청결 환자의 간호기록 SOAPIE 양식으로 작성해줘.", "output": "환자 정보: \n 이름: 윤승우\n 나이: 43세\n 성별: 남성\n\nSOAPIE 기록:\n\nS: 주관적 데이터 (Subjective):\n 환자는 최근에 "'목이 건조하고 기침이 나와 기도가 막힌 느낌이에요'"라고 언급함.\n\nO: 객관적 데이터 (Objective): \n입안과 목이 건조하며   기침이 동반되고 가래가 관찰됨. \n환자의 입안에서 기후제 사용 흔적이 있음.\n\nA: 평가 (Assessment): \n환자는 비효과적인 기도 청결로 인한 기침과 건조한 목 증상을 나타냄.\n\nP: 계획 (Plan):  \n1. 적절한 수분 공급   \n2. 입안 관리 교육   \n3. 의사의 처방에 따른 기침 억제제 투여.\n\nI: 간호 중재 (Intervention): \n1. 수분 공급을 증가시켜 목이 건조하지 않도록 유지함   \n2. 입안을 깨끗하게 관리하고 적절한 가습기 사용을 교육함   \n3. 의사의 처방에 따라 기침을 억제하는 약물을 투여함.\n\nE: 평가 (Evaluation): \n1. 수분 공급으로 목이 더 적당한 상태로 유지되고   \n2. 입안 관리 교육을 통해 입안의 건조함이 완화되며   \n3. 기침 억제제 투여로 기침이 감소함."},
    {'input': "가스 교환 장애 환자의 간호기록 SOAPIE 양식으로 작성해줘.", "output":"환자 정보: \n 이름: 황민서\n 나이: 30세\n 성별: 여성\n\nSOAPIE 기록:\n\nS: 주관적 데이터 (Subjective):\n 환자는 최근에 숨쉬기가 어려워지며 "'가슴이 무거워지고 숨이 차서 정말로 힘들어'"라고 언급함.\n\nO: 객관적 데이터 (Objective): \n호흡 속도가 급격히 증가하고   흉부에서 호흡음이 들림. \n산소 포화도는 88%로 낮아져 있음.\n\nA: 평가 (Assessment): \n환자는 가스교환 장애로 인한 급성 호흡곤란 상태에 있음.\n\nP: 계획 (Plan):  \n1. 산소 치료 시작   \n2. 기침을 돕기 위한 의약품 투여   \n3. 평가를 통한 추가 간호 중재 결정.\n\nI Intervention): \n1. 산소 치료를 시작하여 호흡 소요량을 충족시킴   \n2. 의사의 처방에 따라 기침을 돕기 위한 의약품 투여   \n3. 계속되는 평가를 통해 호흡 소요량 및 기타 증상을 지속적으로 모니터링.\n\nE: 평가 (Evaluation): \n1. 산소 치료로 산소 포화도가 상승함   \n2. 의약품 투여로 기침이 완화되어 호흡이 편해짐   \n3. 지속적인 평가를 통해 환자의 호흡 상태를 지속적으로 관찰하고 조절함."},
    {'input': "낙상 환자의 간호기록 SOAPIE 양식으로 작성해줘.","output":"환자 정보:\n 이름: 이철수\n 나이: 68세\n 성별: 남성\n\nSOAPIE 기록:\n\nS: 주관적 데이터 (Subjective):\n 어깨와 허리 부위 통증\n\nO: 객관적 데이터(Objective)\n 통증 위치: 어깨와 허리 부위\n 통증 정도: 5/10\n 보행 능력: 보조기를 사용하여 걷고 있음\n 환자 상태: 과거 낙상으로 균형 감각 저하\n 안전 조치: 병상 주변에 낙상 방지 매트 및 낙상 경보 장치 설치\n\nA: 간호 평가(Assessment)\n 환자의 낙상 위험 상태 확인\n 통증으로 인한 환자의 움직임 제약\n 환자의 균형 감각 저하로 인한 낙상 위험 증가\n\nP: 계획(Plan)\n1. 통증 관리:\n    통증 강도 평가 및 기록\n    의사와 협력하여 통증 관리 계획 수립 \n\n2. 환자 안전을 위한 조치:\n    보행 시 보조기 사용 권장\n    낙상 방지 매트 및 낙상 경보 장치 유지\n    환자에게 정기적인 안전 교육 제공 \n    낙상 위험 감소를 위한 환자 맞춤형 운동 프로그램 개발\n    환자의 상태에 따라 낙상 예방 계획을 조정하여 안전성 강화\n\nI: 간호 중재(Intervention)\n    간호사와 물리치료사 협력하여 균형 훈련 계획 수립\n    환자에게 일상 생활에서의 균형 향상을 위한 운동 권장\n    환자의 우려와 요구사항을 듣고 존중\n    낙상 예방을 위한 환자의 협력 유도\n\nE: 평가(Evaluation)\n 통증 관리 계획으로 통증이 완화되고   환자의 움직임이 유연해짐\n 환자가 보조기를 적극적으로 활용하여 보행 중 안정성이 향상됨\n 균형 감각 향상을 위한 간호 중재가 성공적으로 진행되어   환자의 낙상 위험 감소"},
    {'input': "불안정한 혈압 위험 환자의 간호기록 SOAPIE 양식으로 작성해줘.","output":"환자 정보: \n 이름: 김경주\n 나이: 61세\n 성별: 남성\n\nSOAPIE 기록:\n\nS: 주관적 데이터 (Subjective):\n 환자는 최근에 두통과 어지러움을 자주 느끼며 "'눈 앞이 흐릿하고 가끔씩 두근거리는 느낌이 들어'"라고 언급함.\n\nO: 객관적 데이터 (Objective): \n혈압 측정 결과 : 수축기 혈압이 150 mmHg를 넘어서고 이완기 혈압도 95 mmHg를 넘어섬. \n심박수는 정상 범위 내에 있음.\n\nA: 평가 (Assessment): \n환자는 불안정한 혈압 상태에 있으며  고혈압에 따른 두통 및 어지러움이 나타남.\n\nP: 계획 (Plan):  \n1. 혈압 감시 및 기록   \n2. 생활양식 개선 및 저염 식이 권고   \n3. 심리적 지원 및 스트레스 관리 교육.\n\nI: 간호 중재 (Intervention): \n1. 혈압을 정기적으로 감시하고 기록함   \n2. 영양사와 협력하여 생활양식 개선과 저염 식이를 권고함   \n3. 심리상담사와 협력하여 스트레스 관리 교육을 제공함.\n\nE: 평가 (Evaluation): \n1. 혈압 감시로 혈압이 안정되고 있음을 확인함   \n2. 영양 상태가 개선되고 혈압 관리에 도움이 되는 식이 습관이 형성됨   \n3. 스트레스 관리 교육을 통해 환자의 정서적 상태가 개선됨."},
    {'input': "근육 긴장 환자의 간호기록 SOAPIE 양식으로 작성해줘.","output":"환자 정보: \n 이름: 박성호\n 나이: 40세\n 성별: 남성\n\nSOAPIE 기록:\n\nS: 주관적 데이터 (Subjective):\n 환자는 "'어깨와 목이 뭉쳐서 통증이 있고   자주 두통도 느껴져'"라고 언급함.\n\nO: 객관적 데이터 (Objective): \n어깨와 목 주변 근육이 경직되어 있으며   환자는 허리를 구부리기 어려워하는 모습을 보임. \n특히 어깨 부분에서 두통이 발생하는 것으로 나타남.\n\nA: 평가 (Assessment): \n환자는 근육의 긴장으로 인한 어깨와 목 통증   그리고 두통을 경험하고 있음.\n\nP: 계획 (Plan):  \n1. 근육 이완을 위한 안정적인 자세 교육   \n2. 따뜻한 찜질 적용   \n3. 의사의 처방에 따른 근육 이완을 돕는 약물 투여.\n\nI: 간호 중재 (Intervention): \n1. 올바른 자세 유지를 교육하여 근육에 가해지는 압력을 최소화함   \n2. 따뜻한 찜질을 통해 근육을 이완시키고 혈액 순환을 촉진함   \n3. 의사의 처방에 따라 근육 이완을 돕는 약물을 투여함.\n\nE: 평가 (Evaluation): \n1. 올바른 자세 유지로 근육의 긴장이 감소하고 통증이 완화됨   \n2. 따뜻한 찜질로 근육이 이완되어 움직임이 유연해짐   \n3. 약물 투여로 근육의 긴장이 완화되고 통증이 감소함."},
    {'input': "abdomen pain NRS 7점 환자의 간호기록 SOAPIE 양식으로 작성해줘.","output":"환자 정보: \n 이름: 박지현\n 나이: 45세\n 성별: 남성\n\nSOAPIE 기록:\n\nS: 주관적 데이터 (Subjective):\n 환자는 "'복부가 심하게 아프고   통증이 8점 정도로 심해'"라고 언급함.\n\nO: 객관적 데이터 (Objective): \n복부 부위에서 환자의 표정이 불편하며   복부를 만지면 통증으로 인해 피부가 긴장되어 있음. \nNRS(Numeric Rating Scale)로 측정한 통증 점수는 8/10임.\n\nA: 평가 (Assessment): \n환자는 심한 복부 통증으로 고통을 겪고 있음.\n\nP: 계획 (Plan):  \n1. 의사와의 상담 후 통증 관리 계획 수립   \n2. 통증 관리를 위한 약물 투여   \n3. 통증의 원인을 평가하기 위한 추가 진단 검사 실시.\n\nI: 간호 중재 (Intervention): \n1. 의사의 지시에 따라 통증 관리 계획을 수행함   \n2. 통증을 완화하기 위해 의사의 처방에 따라 약물을 투여함   \n3. 추가 진단 검사를 위해 필요한 조치를 취함.\n\nE: 평가 (Evaluation): \n1. 통증 관리 계획을 통해 통증이 감소함   \n2. 약물 투여로 인해 환자의 편안함이 증가함   \n3. 추가 진단 검사 결과를 토대로 통증의 원인을 파악하고 조치를 취함."},
    {'input': "외상성 지주막하 출혈 환자의 간호기록 SOAPIE 양식으로 작성해줘.","output":"환자 정보: \n 이름: 박성준\n 나이: 22세\n 성별: 남성\n\nSOAPIE 기록:\n\nS: 주관적 데이터 (Subjective):\n 환자는 "'머리가 심하게 아프고   안구 주위에 눈물이 나오면서 시야가 흐릿해져'"라고 언급함.\n\nO: 객관적 데이터 (Objective): \n머리에 출혈 부위가 보이며   안구 주위에 혈종이 형성되어 있음. \n환자의 시야가 흐릿하고 빛에 민감한 반응이 나타남.\n\nA: 평가 (Assessment): \n외상성 지주막하 출혈로 인한 머리 통증 및 시야 장애를 경험하고 있음.\n\nP: 계획 (Plan): \n1. 의사와의 협의 후 출혈 관리 계획 수립   \n2. 안구 주위 혈종의 압력 완화를 위한 조치   \n3. 의사의 지시에 따라 머리 CT 등 추가 검사 실시.\n\nI: 간호 중재 (Intervention): \n1. 출혈 관리 계획을 수행함   \n2. 안구 주위 혈종을 피하기 위해 머리를 고정하는 조치를 시행함   \n3. 의사의 지시에 따라 머리 CT 등 추가 검사를 시행함.\n\nE: 평가 (Evaluation): \n1. 출혈 관리 계획을 통해 출혈 양상이 감소함   \n2. 안구 주위 혈종의 압력 완화로 시야 장애가 개선됨   \n3. 추가 검사 결과를 통해 출혈의 정확한 위치와 정도를 확인함."},
    {'input': "아침 식후 고혈당 환자의 간호기록 SOAPIE 양식으로 작성해줘.","output":"환자 정보: \n 이름: 이진희\n 나이: 42세\n 성별: 여성\n\nSOAPIE 기록:\n\nS: 주관적 데이터 (Subjective):\n 환자는 "'아침 식후에 혀가 마르고 입이 쓰려워요. 배가 불러도 계속 배가 고프고   속이 메스꺼워집니다'"라고 언급함.\n\nO: 객관적 데이터 (Objective): \n혀가 건조하고 입이 쓰려워 보이며   식후에도 지속적인 고통감을 나타냄. \n혈당 모니터링 결과   아침 식후 고혈당으로 200 mg/dL 상승함.\n\nA: 평가 (Assessment): \n아침 식후 고혈당으로 인한 증상이 나타나고 있으며   식후 혈당 수치가 상승함.\n\nP: 계획 (Plan):  \n1. 의사와의 협의 후 혈당 관리 계획 수립   \n2. 식사 전후 인슐린 투여 조절   \n3. 영양사 상담 및 식습관 개선을 위한 교육 실시.\n\nI: 간호 중재 (Intervention): \n1. 혈당 관리 계획을 수행함   \n2. 의사의 지시에 따라 인슐린 투여량을 조절함   \n3. 영양사의 조언을 받아 식습관을 개선하는 교육을 실시함.\n\nE: 평가 (Evaluation): \n1. 혈당 관리 계획을 통해 혈당 수치가 안정화됨   \n2. 조절된 인슐린 투여로 식후 고통감이 감소함   \n3. 영양 교육을 통해 식습관이 개선되고 증상이 완화됨."},
    {'input': "양쪽 무릎 NRS 6점 환자의 간호기록 SOAPIE 양식으로 작성해줘.","output":"환자 정보:\n 이름: 박서준\n 나이: 35세\n 성별: 여성\n\nSOAPIE 기록:\n\nS: 주관적 데이터 (Subjective):\n 환자는 "'양쪽 무릎이 아프고   통증이 6점 이상으로 느껴져 걷기가 힘들어졌어요'"라고 언급함.\n\nO: 객관적 데이터 (Objective): \n양쪽 무릎에서 통증을 호소하며 보행 시 어려움을 나타냄. \nNRS를 사용하여 양쪽 무릎 통증을 7점으로 평가함.\n\nA: 평가 (Assessment): \n양쪽 무릎 통증으로 인한 걷기의 어려움을 경험하고 있음.\n\nP: 계획 (Plan):  \n1. 의사와의 협의 후 통증 관리 계획 수립   \n2. 물리치료 및 약물 요법   \n3. 환자 교육 및 자가 관리 계획 수립.\n\nI: 간호 중재 (Intervention): \n1. 통증 관리 계획을 수행함   \n2. 물리치료 세션 및 처방된 약물 복용   \n3. 환자에게 통증 관리 및 자가 돌봄에 대한 교육을 제공함.\n\nE: 평가 (Evaluation): \n1. 통증 관리 계획을 통해 무릎 통증이 완화됨   \n2. 물리치료와 약물 복용으로 인해 걷기의 어려움이 감소함   \n3. 환자가 스스로 통증을 관리하는 능력이 향상됨."},
    {'input': "통증관리 환자의 간호기록 SOAPIE 양식으로 작성해줘.","output":"환자 정보:\n 이름: 이현우\n 나이: 60세\n 성별: 남성\n\nSOAPIE 기록:\n\nS: 주관적 데이터 (Subjective):\n 환자는 허리 통증으로 외래 진료를 받으러 왔음.\n 통증은 최근 1주일 동안 지속되고 있으며   특히 일어나거나 앉을 때 심해짐.\n\nO: 객관적 데이터 (Objective):\n 통증 부위: 허리 하부\n 통증 강도: 6/10 (휴식 중)\n 움직임 제한 및 불편함 관찰\n 허리 Xray 결과: 디스크 허즘 소견\n\nA: 평가 (Assessment):\n 만성적인 허리 통증으로 외래 진료 필요.\n 허리 Xray 결과를 토대로 디스크 허즘의 가능성이 있음을 고려.\n\nP: 계획 (Plan):\n 정확한 통증 원인 규명을 위한 추가 검사 진행 (MRI 등).\n 통증 관리를 위한 의약품 처방 및 물리치료 시작.\n 허리에 부담을 줄이기 위한 자세 교정과 운동 가이드 제공.\n\nI: 간호 중재 (Intervention):\n 정확한 통증 위치와 특성 기록.\n 의약품 투여 및 효과 모니터링.\n 물리치료사와의 협업을 통한 운동 가이드 제공.\n\nE: 평가 (Evaluation):\n 추가 검사 결과를 토대로 디스크 허즘의 정도와 위치 확인.\n 의약품 및 물리치료로 허리 통증이 완화되고 환자의 움직임이 개선됨."},
]



# FewShotChatMessagePromptTemplate 인스턴스 생성
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

# 추가 지침
instruction = """
You are a nurse working in a general hospital. Answer the question, adhering to the following guidelines:
1. Write in Korean
2. Utilize the information available in the prompt. If relevant information is not found there, then seek answers through other channels.
3. Use precise and concise language. Use endings like 음, 함, 임.
4. Fill out patient information in order
5. Necessary to include '주관적 증상(Subjective Data), 객관적 정보(Objective Data), 사정(Assessment), 계획(Planning), 중재(Interventions), 평가(Evaluation)
6. Create a fictional patient's information.
7. Be as specific as possible.
8. Record precise measurements.
9. Use bullet points for the description instead of using numeric order.
10.Fill in the patient's name, age, and gender, but no other personal information.
11. Give it a different name ex) 김현중, 구진성, 홍선우, 조아라, 최승우, 남하늘, 신영태...
"""



# 최종 프롬프트 템플릿 생성
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instruction),
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# 사용자 질문
# user_question = "지주막하 출혈(Traumatic SAH, Hemorrhage)에 대한 간호기록을 NANDA형식으로 작성해줘"
# message = HumanMessage(content=user_question)

# 체인 구성
chain = final_prompt | chat

# CSV 파일에서 데이터 불러오기
df = pd.read_csv("fix_csvfile\\SOAPIE_cleaned.csv")

# 새로운 데이터셋
new_dataset = []

# 각 행 처리
for index, row in df.iterrows():
    user_question = row['disease name'], "환자의 간호기록 SOAPIE 양식으로 작성해줘."  # CSV 파일의 'input' 열에서 질문을 가져옴
    message = HumanMessage(content=user_question)
    response = chain.invoke({"input": user_question, "output": ""})
    new_dataset.append({'input': user_question, 'output': response.content})
    print(response)

# 새로운 데이터셋을 DataFrame으로 변환
new_df = pd.DataFrame(new_dataset)

# 새로운 데이터셋을 CSV 파일로 저장
new_df.to_csv("저장경로", index=False, encoding='utf-8')
