# v1-gcp-preview
Contains GCP and deploy codes for Preview: Product Review and OCR analyzer for customer.

*Read this in other languages: [한국어](docs/README_ko.md)*

## Whole service architecture
![image](https://github.com/user-attachments/assets/2bdeee34-ff59-4368-a9d6-e4e6935f44d3)

### Slurm Cluster ML pipeline
![ml_pipeline drawio](https://github.com/user-attachments/assets/f9c22f23-c661-4753-9938-761c47ca5cc9)

## Cloud Architecture (final)

~~Cloud Functions:~~

~~Pub/Sub:~~

~~Cloud Run:~~

### GCP Batch:
- Job as Service
- Find regions which remain extra GPU and can be provisioning as SPOT type and serving models with batch(model - https://github.com/9unu/Preview_model).
- Considered costs and efficiency such as Network cost, SPOT provisioning, etc.
- Deploy with python script "deploy.py"

### Cloud Build:
- Deploy whole architecture except VM instances
- IaC(Infrastructure as a Code)
- consists of yaml file and dockerfile
