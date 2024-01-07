import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Azure OpenAI를 사용하는 객체 생성
chat = AzureChatOpenAI(
    deployment_name='배포 이름',
    model_name='모델 이름',
)

# 한국어 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            
            """ 당신은 간호기록 작성지를 작성해주는 전문가입니다. 다음의 의료기록을 분석하고 예시를 참고하여 대표적인 병명 하나만 기입하세요.
            예시 : 비효과적 호흡 양상, 비효과적 기도청결, 가스교환 장애, 낙상의 위험, 불안정한 혈압의 위험, 근육의 긴장,  충수염으로 인한 복통,  외상성 지주막하, 출혈,  당뇨병,  무릎관절증 퇴행성 관절염,  통증,  욕창,  자발적 환기장애,  급성통증,  고체온,  분만 통증 
            """,
        ),
        ("human", "{medical_record}"),
    ]
)

# 프롬프트와 채팅 모델 결합
chain = prompt | chat  
# CSV 파일에서 데이터 불러오기
pd.read_csv("문장 수정한 파일 경로", encoding="utf-8")
# 새로운 데이터셋
new_dataset = []

# 각 행 처리
for index, row in df.iterrows():
    response = chain.invoke({"medical_record": row['input']})
    new_dataset.append({'input': row['input'], 'disease name': response})
    print(response)
    

# 새로운 데이터셋을 DataFrame으로 변환
new_df = pd.DataFrame(new_dataset)

# 새로운 데이터셋을 CSV 파일로 저장
new_df.to_csv("저장하고 싶은 경로\\이름.csv", index=False, encoding='utf-8')
