# v1-gcp-preview
Preview: 소비자를 위한 OCR과 리뷰 분석서비스의 GCP 배포를 위한 소스코드 및 배포 코드입니다.
## 전체 서비스 아키텍처

### 서비스에 사용되는 데이터 파이프라인
![preview_service_architecture (1)](https://github.com/user-attachments/assets/885269c7-2557-45b3-a87c-ecdda27b2cbe)

### 머신러닝 데이터 파이프라인
![image](https://github.com/user-attachments/assets/fca29d5f-92dd-46a7-ba3d-84d0aa1f42eb)




## 클라우드 아키텍처

### Cloud Functions:
- OCR과 리뷰를 위한 데이터 전처리를 담당합니다
- OCR 데이터, 리뷰 데이터를 위하여 각각 두 가지 종류가 있습니다
- 크롤러를 통하여 데이터를 받습니다 (Preview - https://github.com/ket0825/preview)
- 서버리스이며 확장성이 높습니다

### Pub/Sub:
- Cloud Function을 통하여 토픽을 게시받고, 구독하는 Cloud run에 데이터를 넘겨줍니다
- 푸시 모델입니다
- LLM 모델을 포함하는 Cloud run을 트리거합니다

### Cloud Run:
- Klue-BERT를 이용한 ABSA 모델을 CPU로 서빙합니다 (model - https://github.com/9unu/Preview_model)
- OCR 데이터와 리뷰 데이터를 위한 두 가지 종류의 이미지가 artifact registry에 저장되어 있습니다
- docker compose로 서버와 DB가 배포되어 있는 Compute Engine e2-medium 인스턴스에 데이터를 저장합니다 (Honeycomb - https://github.com/ket0825/honeycomb)  
- 서버리스이며 확장성이 높습니다.

### Cloud Build:
- Compute Engine 인스턴스를 제외한 모든 아키텍처를 배포합니다.
- 코드를 통한 인프라스트럭처 배포 서비스입니다(IaC)
- yaml과 dockerfile로 이루어져 있습니다.

-------------


# v1-gcp-preview
Contains GCP and deploy codes for Preview: Product Review and OCR analyzer for customer.

## Whole service architecture

### Data pipeline
![preview_service_architecture (1)](https://github.com/user-attachments/assets/885269c7-2557-45b3-a87c-ecdda27b2cbe)

### ML local pipeline
![image](https://github.com/user-attachments/assets/fca29d5f-92dd-46a7-ba3d-84d0aa1f42eb)




## Cloud Architecture

### Cloud Functions:
- Get data with crawler (Preview - https://github.com/ket0825/preview)
- preprocess reviews and OCR data
- serverless and scalable

### Pub/Sub:
- published by cloud functions and cloud run subscribe the topic published by cloud functions
- push model
- trigger cloud run containing LLM model

### Cloud Run:
- Klue-BERT ABSA model served by CPU (model - https://github.com/9unu/Preview_model)
- two types: for OCR data and review data
- load data at VM instances (Honeycomb - https://github.com/ket0825/honeycomb)
- serverless and scalable

### Cloud Build:
- Deploy whole architecture except VM instances
- IaC(Infrastructure as a Code)
- consists of yaml file and dockerfile

