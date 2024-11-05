from google.cloud import batch_v1
from dotenv import load_dotenv
import os

load_dotenv()
REPOSITORY = os.environ['REPOSITORY']
IMAGE_NAME = os.environ['IMAGE_NAME']
TAG = os.environ['TAG']
VPC_NAME = os.environ['VPC_NAME']
PROJECT_ID = os.environ['PROJECT_ID']
ZONES = os.environ['ZONES']

from google.cloud import compute_v1
from copy import deepcopy

def get_instance_template(project_id: str, template_name: str) -> compute_v1.InstanceTemplate:
    """기존 instance template을 가져옵니다."""
    client = compute_v1.InstanceTemplatesClient()
    return client.get(project=project_id, instance_template=template_name)

def create_modified_template(
    project_id: str, 
    source_template_name: str,
    new_template_name: str,
    new_subnet: str,
    new_region: str
) -> compute_v1.Operation:
    """
    기존 template을 복제하고 subnet과 region을 변경합니다.
    
    Args:
        project_id: GCP 프로젝트 ID
        source_template_name: 복제할 원본 템플릿 이름
        new_template_name: 새로 생성할 템플릿 이름
        new_subnet: 새로운 서브넷 (형식: projects/PROJECT/regions/REGION/subnetworks/SUBNET)
        new_region: 새로운 리전 (예: us-central1)
    """
    client = compute_v1.InstanceTemplatesClient()
    
    # 원본 템플릿 가져오기
    source_template = get_instance_template(project_id, source_template_name)
    
    # 새로운 템플릿 객체 생성
    new_template = compute_v1.InstanceTemplate()    
    
    # 원본 템플릿의 속성을 복사
    template_copy = deepcopy(source_template)
    
    # name 필드 제거 (API에서 자동 거부됨)
    if hasattr(template_copy, 'name'):
        delattr(template_copy, 'name')
    
    # 기타 자동 생성 필드 제거
    if hasattr(template_copy, 'id'):
        delattr(template_copy, 'id')
    if hasattr(template_copy, 'creation_timestamp'):
        delattr(template_copy, 'creation_timestamp')
    if hasattr(template_copy, 'self_link'):
        delattr(template_copy, 'self_link')
    
    # 새로운 템플릿에 복사한 속성 할당
    new_template = template_copy
    
    # 템플릿 이름 변경
    new_template.name = new_template_name
    new_template.description = f"Cloned from {source_template_name} with modified subnet and region"
    # 네트워크 인터페이스 region 변경        
    new_template.region = new_region        
        
    
    # 네트워크 인터페이스 region 및 subnet 변경: 
    if new_template.properties.network_interfaces:                
        new_template.properties.network_interfaces[0].subnetwork = new_subnet
        
    # MARK: region 관련 설정 변경 (예: disk source image): 미사용...    
    if new_template.properties.disks:
        for disk in new_template.properties.disks:
            if disk.source and 'regions' in disk.source:
                disk.source = disk.source.replace(
                    disk.source.split('/regions/')[1].split('/')[0],
                    new_region
                )
    
    # 새로운 템플릿 생성 요청
    operation = client.insert(
        project=project_id,
        instance_template_resource=new_template
    )
    
    return operation

def wait_for_operation(operation: compute_v1.Operation, project_id: str):
    """작업 완료를 기다립니다."""
    client = compute_v1.GlobalOperationsClient()
    return client.wait(project=project_id, operation=operation.name)

def clone_template_with_new_network(
    project_id,
    source_template, 
    new_region,
    ) -> str:
        
    new_template = f"temp-{new_region}-{source_template}"    
    new_subnet = f"projects/{project_id}/regions/{new_region}/subnetworks/default" # subnet name is default
    
    operation = create_modified_template(
        project_id=PROJECT_ID,
        source_template_name=source_template,
        new_template_name=new_template,
        new_subnet=new_subnet,
        new_region=new_region
    )
    
    # 작업 완료 대기
    wait_for_operation(operation, project_id)
    print(f"Successfully created new template: {new_template}")
    return new_template
    

def delete_instance_template(project_id: str, template_name: str) -> compute_v1.Operation:
    """인스턴스 템플릿을 삭제합니다."""
    client = compute_v1.InstanceTemplatesClient()    
    return client.delete(project=project_id, instance_template=template_name)

# MARK: 반드시 JOB 완료 이후에 삭제해야 함.
def delete_template(project_id, template_name):
    try:
        operation = delete_instance_template(project_id, template_name)
        wait_for_operation(operation, project_id)
        print(f"Successfully deleted template: {template_name}")
    except Exception as e:
        print(f"Error deleting template: {str(e)}")

def create_gpu_job(project_id, zone, new_template):
    client = batch_v1.BatchServiceClient()        
    region = "-".join(zone.split("-")[:2])
    
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
      
    images_uri = f'asia-northeast3-docker.pkg.dev/{project_id}/{REPOSITORY}/{IMAGE_NAME}:{TAG}'    

    task_group = batch_v1.TaskGroup()
    task_spec = batch_v1.TaskSpec()
    runnables = batch_v1.Runnable()
    runnables.environment.variables = env_vars
    runnables.container = batch_v1.Runnable.Container()
    runnables.container.image_uri = images_uri
    task_spec.runnables = [runnables]
    
    allocation_policy = batch_v1.AllocationPolicy()    
    instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()    
    instances.instance_template = new_template
    
    # MARK: InstanceTemplate 미사용으로 주석 처리    
    # policy = batch_v1.AllocationPolicy.InstancePolicy()
    # policy.machine_type = 'n1-standard-4'
    # policy.boot_disk.type_ = 'pd-ssd'
    # policy.boot_disk.size_gb = 50
    # policy.provisioning_model = 'SPOT'    
    
    # accelerators = batch_v1.AllocationPolicy.Accelerator()
    # accelerators.type_ = 'nvidia-tesla-t4'
    # accelerators.count = 1
    # policy.accelerators = [accelerators]    
    
    location = batch_v1.AllocationPolicy.LocationPolicy()
    location.allowed_locations = [zone]
    
    allocation_policy.instances = [instances]
    allocation_policy.location = location
    
    logs_policy = batch_v1.LogsPolicy()
    logs_policy.destination = 'CLOUD_LOGGING'
    
    network_policy = batch_v1.AllocationPolicy.NetworkPolicy()
    
    network_interface = batch_v1.AllocationPolicy.NetworkInterface()    
    network_interface.network = f'projects/{project_id}/global/networks/default'    
    network_interface.subnetwork = f'projects/{project_id}/regions/{region}/subnetworks/default'
        
    network_policy.network_interfaces = [network_interface]
    
    allocation_policy.network = network_policy
    
    job = {
        "task_groups":[
            {
                "task_spec":{
                    "runnables": [{
                        "environment": {
                                "variables": env_vars
                            },
                        "container": {
                            "image_uri": images_uri,                            
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
                    "instance_template": new_template,
                    
                    # "policy": {
                    #     "machine_type": "n1-standard-4",
                    #     "boot_disk": {
                    #         "type_": "pd-ssd",
                    #         "size_gb": 50,
                    #         # 기본으로 컨테이너 작업 시Batch-cos-stable-offical 사용
                    #         # 'image': 'projects/cos-cloud/global/images/family/cos-stable' 
                    #     },
                    #     # 기본 metadata 사용 사례가 없음...
                    #     # "metadata": metadata,
                    #     "provisioning_model": "SPOT",                        
                    #     "accelerators": [
                    #         {
                    #             "type_": "nvidia-tesla-t4",
                    #             "count": 1
                    #         }
                    #     ]
                    # }
                }
            ],
            "location": {
                "allowed_locations": [
                    f"zones/{zone}"
                ]
            },
            "network": {
                "network_interfaces": [
                    {
                        "network": f'projects/{project_id}/global/networks/{VPC_NAME}',
                        "subnetwork": f'projects/{project_id}/regions/{region}/subnetworks/default'
                    }
                ]
            },
        },
        "logs_policy": {
            "destination": "CLOUD_LOGGING",
        }
    }            
    
    # job = batch_v1.Job()
    # job.task_groups = [task_group]
    # job.allocation_policy = allocation_policy
    # job.logs_policy = logs_policy
    
    # create_request = batch_v1.CreateJobRequest()
    # create_request.parent = f'projects/{project_id}/locations/{zone}'
    # create_request.job = job            
    
    
    # 여기서 metadata 등록 가능...
    # response = client.create_job(request=create_request, metadata=metadata)    
    parent = f'projects/{project_id}/locations/{region}'
    
    response = client.create_job(job=job, parent=parent)
    print(f"Created job: {response.name}")
    return response

if __name__ == "__main__":
    for zone in ZONES.split(","):
        region = "-".join(zone.split("-")[:2])
        new_template = clone_template_with_new_network(PROJECT_ID, "batch-kluebert-ocr-template", region)
        create_gpu_job(PROJECT_ID, zone, new_template)
        # delete_template(PROJECT_ID, new_template)