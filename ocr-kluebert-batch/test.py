import sys

from src_ocr.deploy.env_config import EnvConfig

env_config = EnvConfig()   
print(env_config.__dict__)

print(sys.path)