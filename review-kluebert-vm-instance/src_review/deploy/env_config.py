import os
from pydantic_settings import BaseSettings

class EnvConfig(BaseSettings):    
    # These are the settings that are loaded from the environment   
    # If we change the model by category, change the label_info_file and out_model_path   
    eval_batch_size: int = os.getenv("EVAL_BATCH_SIZE", 16)        
    init_model_path: str = os.getenv("INIT_MODEL_PATH", "klue/bert-base")    
    max_length: int = os.getenv("MAX_LENGTH", 512)
    need_birnn: int = os.getenv("NEED_BIRNN", 0)    
    sentiment_drop_ratio: float = os.getenv("SENTIMENT_DROP_RATIO", 0.3)
    aspect_drop_ratio: float = os.getenv("ASPECT_DROP_RATIO", 0.3)        
    sentiment_in_feature: int = os.getenv("SENTIMENT_IN_FEATURE", 768)
    aspect_in_feature: int = os.getenv("ASPECT_IN_FEATURE", 768)        
    base_path: str = os.getenv("BASE_PATH", "review_tagging")        
    label_info_file: str = os.getenv("LABEL_INFO_FILE", "meta.bin")
    out_model_path: str = os.getenv("OUT_MODEL_PATH", "pytorch_model.bin")    
    post_server: str = os.getenv("POST_SERVER", "http://localhost:5000/api/review")
        
    