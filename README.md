# v1-gcp-preview
Contains GCP and Compile for deploy Preview: Product Review and OCR analyzer for customer.

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

