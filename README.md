# 기업연계 팀 프로젝트(디케이메디인포)
- 프로젝트 기간: 2023.11 ~ 2023.12(6주)
- 참여인원 : 4명

|역할|담당업무|
|--|--|
|팀장|- 전체 일정 관리 및 회의 리드 <br> - 기존 전처리 코드에 정규식 코드 추가 <br> - LangChainFramework를 활용한 LLM 연결 코드 구현 <br> - Streamlit을 활용한 Chatbot 사이트 구현 |
--- 
## 프로젝트 개요
전자 간호기록 데이터 기반 실습용 간호기록 문구 자동 생성 쳇봇 서비스 구현

## 프로젝트 배경
- 대형병원 병동간호사 업무 40%는 전자간호기록시스템 관련 업무
- 그 중 환자에 대한 간호기록 작성에 많은 시간 소요
- 관련해서 오버타임을 하는 경우가 많음
- 간호기록은 교대 근무에 따른 시계열 방식으로 기록됨

## 프로젝트 기술스택
![py](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)&nbsp;&nbsp;![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)&nbsp;&nbsp;![langchain](https://img.shields.io/badge/LangChain-EE4C2C?style=for-the-badge&logo=LangChain&logoColor=white)&nbsp;&nbsp;![streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)&nbsp;&nbsp;![github](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)

## 프로젝트 작업 흐름


![스크린샷 2024-02-13 193853](https://github.com/satangmu/SmartNurse_Project/assets/148983269/87160e72-975f-4416-a891-45d48f80eba2)



## 프로젝트 진행 과정

1. 데이터 병합 및 기록양식 별 분리

![스크린샷 2024-01-16 010548](https://github.com/satangmu/SmartNurse_Project/assets/148983269/f6a5d2d3-ac75-4485-8167-13c184d60221)

2. 데이터 전처리

![스크린샷 2024-01-16 013618](https://github.com/satangmu/SmartNurse_Project/assets/148983269/9ee05ea0-aac3-4293-88a3-8d2b86d99d76)

![스크린샷 2024-01-16 013826](https://github.com/satangmu/SmartNurse_Project/assets/148983269/4ce4d26b-614f-48a9-8c59-edefdff635bd)

3. 모델 파인튜닝
   
![스크린샷 2024-01-16 014245](https://github.com/satangmu/SmartNurse_Project/assets/148983269/a11eda99-ccbf-4874-aa3e-a86417c723fb)

4. 프롬프트 엔지니어링

![스크린샷 2024-01-16 014538](https://github.com/satangmu/SmartNurse_Project/assets/148983269/a57a4d03-6b87-48bd-ac54-b5a4cb59dd65)

5. 결과

![챗봇-데모영상2](https://github.com/satangmu/SmartNurse_Project/assets/148983269/3a96b23f-6261-4be1-831f-700c044ada70)

## 느낀점
### - 비용 한계
- LLM인 GPT3.5 model을 Fine Tuning 하는 방법은 많은 비용이 발생 -> RAG 하이브리드 모델을 활용하는 방법 모색
### - 새로운 기술 활용
- 데이터 전처리 하는데 시간을 많이 써서 프로젝트 기간 안에 Chatbot을 못 만들거라 판단 -> 프로젝트 기간 동안 남은 시간에 LangChain과 Streamlit을 배우면서 Chatbot을 만드는데 있어서 큰 기여를 함



  
