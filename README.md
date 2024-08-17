# v1-gcp-preview
Contains GCP and Compile for deploy Preview: Product Review and OCR analyzer for customer services.


## Cloud Architecture

### Cloud Functions:
- preprocess reviews and OCR data.
- serverless and scalable.

### Pub/Sub:
- published by cloud functions and cloud run subscribe the topic published by cloud functions.
- push model.
- trigger cloud run containing LLM model.

### Cloud Run:
- Klue-BERT ABSA model served by CPU.
- two types: for OCR data and review data.
- load data at VM instances (Honeycomb)
- serverless and scalable

### Cloud Build:
- Deploy whole architecture except VM instances.
- IaC(Infrastructure as a Code)
- consists of yaml file and dockerfile.
