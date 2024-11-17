# v1-gcp-preview
Preview: 소비자를 위한 OCR과 리뷰 분석서비스의 GCP 배포를 위한 소스코드 및 배포 코드입니다.
## 전체 서비스 아키텍처 (최종)
![preview drawio](https://github.com/user-attachments/assets/96c68cee-322b-4291-833e-91fc684ca5e6)

### 머신러닝 파이프라인
![image](https://github.com/user-attachments/assets/fca29d5f-92dd-46a7-ba3d-84d0aa1f42eb)

## 클라우드 아키텍처 (최종)

~~Cloud Functions:~~

~~Pub/Sub:~~

~~Cloud Run:~~

### GCP Batch:
- Job as Service 형태입니다.
- GPU 여분이 존재하는 region들을 파악하여 SPOT 형태로 모델을 근실시간 서빙이 가능합니다(model - https://github.com/9unu/Preview_model).
- 네트워크, SPOT 등 비용과 효용을 최대한 고려하였습니다.
- deploy.py로 스크립트로 배포하였습니다.

### Cloud Build:
- Compute Engine 인스턴스를 제외한 모든 아키텍처를 배포합니다.
- 코드를 통한 인프라스트럭처 배포 서비스입니다(IaC)
- yaml과 dockerfile로 이루어져 있습니다.


-------------

# v1-gcp-preview
Contains GCP and deploy codes for Preview: Product Review and OCR analyzer for customer.

## Whole service architecture (Now improving...)
![preview drawio](https://github.com/user-attachments/assets/96c68cee-322b-4291-833e-91fc684ca5e6)

### Local ML pipeline
![image](https://github.com/user-attachments/assets/fca29d5f-92dd-46a7-ba3d-84d0aa1f42eb)

## Cloud Architecture (final)

~~Cloud Functions:~~

~~Pub/Sub:~~

~~Cloud Run:~~

### GCP Batch:
- Job as Service
- Find regions which remain extra GPU and can be provisioning as SPOT type and can near real-time serving models(model - https://github.com/9unu/Preview_model).
- Considered costs and efficiency such as Network cost, SPOT provisioning, etc.
- Deploy with python script "deploy.py"

### Cloud Build:
- Deploy whole architecture except VM instances
- IaC(Infrastructure as a Code)
- consists of yaml file and dockerfile
