import os
from pydantic_settings import BaseSettings

class EnvConfig(BaseSettings):    
    # These are the settings that are loaded from the environment   
    # If we change the model by category, change the label_info_file and out_model_path   
    eval_batch_size: int = os.getenv("EVAL_BATCH_SIZE", 1)        
    init_model_path: str = os.getenv("INIT_MODEL_PATH", "klue/bert-base")    
    max_length: int = os.getenv("MAX_LENGTH", 512)
    need_birnn: int = os.getenv("NEED_BIRNN", 0)    
    aspect_drop_ratio: float = os.getenv("ASPECT_DROP_RATIO", 0.3)        
    aspect_in_feature: int = os.getenv("ASPECT_IN_FEATURE", 768)    
    base_path: str = os.getenv("BASE_PATH", "ocr_tagging")        
    label_info_file: str = os.getenv("LABEL_INFO_FILE", "meta.bin")
    out_model_path: str = os.getenv("OUT_MODEL_PATH", "pytorch_model.bin")    
    post_server: str = os.getenv("POST_SERVER", "http://localhost:5000/api/product/detail/one")
    
    # DB config
    host: str = os.getenv("DB_HOST")
    user: str = os.getenv("DB_USER")
    password: str = os.getenv("DB_PASSWORD")
    name: str = os.getenv("DB_NAME")
    port: int = os.getenv("PORT")
    
    # Batch config
    batch_size: int = os.getenv("BATCH_SIZE", 1)
    topic_type: str = os.getenv("TOPIC_TYPE", "klue-bert-v1")
    
    # extra_battery,keyboard,monitor.
    # IN 처리를 위해 '로 감싸줌.
    categories: str = "'"+ \
        "','".join(os.getenv("CATEGORIES", 'extra_battery').split(","))  \
            + "'"
    
    
        
    