from typing import List, Dict
import argparse
import os
from google.cloud import storage
from google.auth import default
import joblib
import transformers
from contextlib import asynccontextmanager
import requests
import json

from modeling.model import ABSAModel
from utils.model_utils import device_setting, load_model
from deploy.stream_log import StreamLog
from deploy.pydantic_models import PubsubRequest, KlueBertReviewRequest
from deploy.env_config import EnvConfig
from modeling.trainer import inference_fn

from fastapi import FastAPI, HTTPException, APIRouter, Request
from fastapi.responses import RedirectResponse
import uvicorn


model = None
log = StreamLog().set_log(level="DEBUG") # only stream.
tokenizer = None
device = None
enc_sentiment, enc_aspect, enc_aspect2, enc_sentiment_score, enc_aspect_score = None, None, None, None, None

def load_bert_model(config: argparse.Namespace):
    global log
    log.info(f"Loading BERT model. Config: {config.__dict__}")
    
    local_model_path = './tmp/pytorch_model.bin'
    local_metadata_path = './tmp/meta.bin'  
    
    if os.path.exists(local_model_path) and os.path.exists(local_metadata_path):
        log.info("Model and metadata already exist. Skip downloading.")        
    
    else:
        bucket_name = config.base_path
        model_file = config.out_model_path
        metadata_file = config.label_info_file
        storage_client = storage.Client()
        
        _, project_id = default()
        print(f"Current Project ID: {project_id}")
        
        # # 버킷 목록 조회
        buckets = storage_client.list_buckets()

        print("Buckets:")
        for bucket in buckets:
            print(f" - {bucket.name}")
        
        #  # 특정 버킷의 객체(파일) 목록 조회    
        bucket = storage_client.get_bucket(bucket_name)
        blobs = bucket.list_blobs()
        
        print(f"\nObjects in bucket '{bucket_name}':")
        for blob in blobs:
            print(f" - {blob.name}")
        
        bucket = storage_client.bucket(bucket_name)
        model_blob = bucket.blob(model_file)
        metadata_blob = bucket.blob(metadata_file)                
        
        os.makedirs('./tmp', exist_ok=True)
        
        print(f"Downloading model to {local_model_path}")     
        model_blob.download_to_filename(local_model_path)        
        log.info(f"Model downloaded to {local_model_path}")
        metadata_blob.download_to_filename(local_metadata_path)
        log.info(f"Metadata (label) downloaded to {local_metadata_path}")    
    
    try:        
        log.info("Loading metadata...")
        metadata = joblib.load(local_metadata_path)
    except Exception as e:    
        log.error(f"Error loading metadata: {e}")
        raise ValueError(f"Error loading metadata: {e}")
        
    log.info("metadata loaded!")
    global enc_sentiment, enc_aspect, enc_aspect2, enc_sentiment_score, enc_aspect_score
    enc_sentiment, enc_aspect, enc_aspect2 = metadata["enc_sentiment"], metadata["enc_aspect"], metadata["enc_aspect2"]
    enc_sentiment_score, enc_aspect_score = metadata["enc_sentiment_score"], metadata["enc_aspect_score"]
    
    num_sentiment = len(list(metadata["enc_sentiment"].classes_))
    num_aspect, num_aspect2 = len(list(metadata["enc_aspect"].classes_)), len(list(metadata["enc_aspect2"].classes_))
    num_sentiment_score = len(list(metadata["enc_sentiment_score"].classes_)) ### score_num 추가
    num_aspect_score = len(list(metadata["enc_aspect_score"].classes_)) ### score_num 추가
    
    global device
    device = device_setting(log)
    global model
    model = ABSAModel(config=config, num_sentiment=num_sentiment, num_aspect=num_aspect, num_aspect2=num_aspect2,
                            num_sentiment_score=num_sentiment_score, num_aspect_score=num_aspect_score,
                            need_birnn=bool(config.need_birnn))
    model = load_model(model=model, state_dict_path=local_model_path, device=device)
    global tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(config.init_model_path, do_lower_case=False)

def inference(
    config, review_data:List[Dict] # TODO: List[KlueBertReviewRequest] -> content_list: List[str]
              ) -> List[Dict]: # List[KlueBertReviewResponse]
    global model
    global tokenizer
    global log
    global device
    global enc_sentiment, enc_aspect, enc_aspect2, enc_sentiment_score, enc_aspect_score   
    
    log.info(f"matchNvMid: {review_data[0]['matchNvMid']}")          
    result = inference_fn(config, review_data, tokenizer, model, enc_sentiment, enc_aspect, enc_aspect2, enc_sentiment_score, enc_aspect_score, device, log)
    
    log.info(f"Result length: {len(result)}")
    log.info(f"first result our topics: {result[0]['our_topics']}")    
    return result



router = APIRouter()

@router.get("/review/health")
async def root():
    return {"message": "klue-bert-review-tagging"}

@router.post("/review/predict")
# def predict(packet: PubsubRequest): 
def predict(packet: KlueBertReviewRequest):  # local test 용
    config = app.state.config                 
    packet_dict = packet.model_dump() # local test 용
    # packet_dict = packet.message.decoded_data # data만 뽑아서 decode하고 dict로 변환        
    if isinstance(packet_dict, list):
        print(f"list case: {packet_dict}")
    elif isinstance(packet_dict, dict):
        print(f"dict case: packet_dict: {packet_dict.keys()}")
    else:
        print(f"unknown case: {packet_dict}")        
    
    
    # review_data = packet_dict['reviews']
    inference_result = inference(config, packet_dict['reviews']) # content_list
    
    for review, result_obj in zip(packet_dict['reviews'], inference_result):
        review['our_topics'] = result_obj['our_topics']
        
    # [review.__setitem__('our_topics', our_topics) for review, our_topics in zip(data, inference_result)]            
    try:
        res = requests.post(config.post_server, json=packet_dict, timeout=10)  # 서버 터지네...
        if res.status_code == 200:
            try:
                return res.json()
            except requests.exceptions.JSONDecodeError:     # application/json이 아닌 경우
                return {
                "status": "SUCCESS" if "[SUCCESS]" in res.text else "UNKNOWN",
                "message": res.text
            }
        else:
            raise HTTPException(status_code=res.status_code, detail=res.text)
            # return {"error": f"HTTP {res.status_code}", "response_text": res.text}
    except requests.exceptions.RequestException as e:
        log.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")


def create_app(config: EnvConfig):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = config
        load_bert_model(config)
        yield
        print("Cleaning up")        

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    
    return app

config = EnvConfig()
app = create_app(config)
    
if __name__ == "__main__":        
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)