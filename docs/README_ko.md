# v1-gcp-preview
"Preview: N사 쇼핑 가격비교" GCP 배포를 위한 소스코드 및 배포 코드입니다.
## 전체 서비스 아키텍처 (최종)
![image](https://github.com/user-attachments/assets/2bdeee34-ff59-4368-a9d6-e4e6935f44d3)


### 머신러닝 파이프라인
![ml_pipeline drawio](https://github.com/user-attachments/assets/f9c22f23-c661-4753-9938-761c47ca5cc9)


## 클라우드 아키텍처 (최종)

~~Cloud Functions:~~

~~Pub/Sub:~~

~~Cloud Run:~~

### GCP Batch:
- Job as Service 형태입니다.
- GPU 여분이 존재하는 region들을 파악하여 SPOT 형태로 모델을 근실시간 서빙이 가능합니다(model - https://github.com/9unu/Preview_model).
- 네트워크, SPOT 등 비용과 효용에 대한 시나리오를 최대한 고려하였습니다.
- deploy.py로 스크립트로 배포하였습니다.

### Cloud Build:
- Compute Engine, Batch 인스턴스를 제외한 모든 아키텍처를 배포합니다.
- 코드를 통한 인프라스트럭처 배포 서비스입니다(IaC)
- yaml과 dockerfile로 이루어져 있습니다.


-------------