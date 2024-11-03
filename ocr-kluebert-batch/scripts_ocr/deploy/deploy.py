from google.cloud import batch_v1
from google.cloud.batch_v1.types import TaskGroup
from google.cloud.batch_v1.types import TaskGroup
from dotenv import load_dotenv
import os

REPOSITORY = 'ocr-kluebert-batch'
IMAGE_NAME = 'kluebert-ocr'
TAG = 'v1'

def create_gpu_job(project_id, location):
    client = batch_v1.BatchServiceClient
    load_dotenv()
    
    env_vars = {
        "DB_HOST": os.environ["DB_HOST"],
        "DB_USER": os.environ["DB_USER"],
        "DB_PASSWORD": os.environ["DB_PASSWORD"],
        "DB_PORT": os.environ["DB_PORT"],
        "DB_NAME": os.environ["DB_NAME"],
        "BATCH_SIZE": os.environ["BATCH_SIZE"],
        "TOPIC_TYPE": os.environ["TOPIC_TYPE"],
        "CATEGORIES": os.environ["CATEGORIES"],
    }    
      
    images_uri = f'{location}-docker.pkg.dev/{project_id}/{REPOSITORY}/{IMAGE_NAME}:{TAG}'    
    
    #TODO: GPU 드라이버는 cloud-init를 통해 설치하도록 설정 필요
    job = {
        "task_groups":[
            {
                "task_spec":{
                    "runnables": [{
                        "container": {
                            "image_uri": images_uri,
                            "environment": {
                                "variables": env_vars
                            }
                        }
                    }],                                
                },
                "task_count": 1,
                "parallelism": 1            
            },            
        ],
        "allocation_policy": {
            "instances": [                
                {
                    "installGpuDrivers": INSTALL_GPU_DRIVERS,
                    "policy": {
                        "machine_type": "n1-standard-4",
                        "disk": {
                            "type": "pd-ssd",
                            "size_gb": 50
                        },
                        "provisioning_model": "SPOT",                        
                        "accelerators": [
                            {
                                "type": "nvidia-tesla-t4",
                                "count": 1
                            }
                        ]
                    }
                }
            ],
            "location": {
                "allowedLocations": [
                    location
                ]
            }        
        },
    }    
        
    parent = f"projects/{project_id}/locations/{location}"
    response = client.create_job(parent=parent, job=job)
    print(f"Created job: {response.name}")
    return response

