from typing import List, Dict
import argparse
import os
from google.cloud import storage
from google.auth import default
import joblib
import transformers
from fastapi import FastAPI, HTTPException, APIRouter, Request
from fastapi.responses import RedirectResponse
import uvicorn
from contextlib import asynccontextmanager
import requests
import json

from modeling.model import ABSAModel
from utils.model_utils import device_setting, load_model
from deploy.stream_log import StreamLog
from deploy.pydantic_models import PubsubRequest, KlueBertReviewRequest
from modeling.trainer import inference_fn



model = None
log = StreamLog().set_log(level="DEBUG") # only stream.
tokenizer = None
device = None
enc_sentiment, enc_aspect, enc_aspect2, enc_sentiment_score, enc_aspect_score = None, None, None, None, None

def load_bert_model(config: argparse.Namespace):
    global log
    log.info(f"Loading BERT model. Config: {config.__dict__}")
    
    # bucket_name = config.base_path
    # model_file = config.out_model_path
    # metadata_file = config.label_info_file
    # storage_client = storage.Client()
    
    # _, project_id = default()
    # print(f"Current Project ID: {project_id}")
    
    # # 버킷 목록 조회
    # buckets = storage_client.list_buckets()

    # print("Buckets:")
    # for bucket in buckets:
    #     print(f" - {bucket.name}")
    
    #  # 특정 버킷의 객체(파일) 목록 조회    
    # bucket = storage_client.get_bucket(bucket_name)
    # blobs = bucket.list_blobs()
    
    # print(f"\nObjects in bucket '{bucket_name}':")
    # for blob in blobs:
    #     print(f" - {blob.name}")
    
    # bucket = storage_client.bucket(bucket_name)
    # model_blob = bucket.blob(model_file)
    # metadata_blob = bucket.blob(metadata_file)        
    
    local_model_path = './tmp/pytorch_model.bin'
    metadata_path = './tmp/meta.bin'    
    
    # print(f"Downloading model to {local_model_path}")     
    # model_blob.download_to_filename(local_model_path)        
    # log.info(f"Model downloaded to {local_model_path}")
    # metadata_blob.download_to_filename(metadata_path)
    # log.info(f"Metadata (label) downloaded to {metadata_path}")    
    
    try:        
        log.info("Loading metadata...")
        metadata = joblib.load(metadata_path)
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

@router.get("/")
async def root():
    return {"message": "klue-bert-review-tagging"}

@router.post("/predict")
def predict(packet: PubsubRequest): 
# def predict(packet: KlueBertReviewRequest):  # local test 용
    config = app.state.config                 
    packet_dict = packet.message.decoded_data # data만 뽑아서 decode하고 dict로 변환        
    if isinstance(packet_dict, list):
        print(f"list case: {packet_dict}")
    elif isinstance(packet_dict, dict):
        print(f"dict case: packet_dict: {packet_dict.keys()}")
    else:
        print(f"unknown case: {packet_dict}")        
    
    # packet_dict = packet.model_dump() # local test 용
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


@router.get("/health")
def health():    
    return {"message": "klue-bert-review-tagging"}


def create_app(config: argparse.Namespace):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = config
        load_bert_model(config)
        yield
        print("Cleaning up")        

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    
    return app


    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    
    eval_batch_size = 1 if not os.environ.get("EVAL_BATCH_SIZE") else int(os.environ.get("EVAL_BATCH_SIZE"))
    init_model_path = "klue/bert-base" if not os.environ.get("INIT_MODEL_PATH") else os.environ.get("INIT_MODEL_PATH")
    max_length = 512 if not os.environ.get("MAX_LENGTH") else int(os.environ.get("MAX_LENGTH"))
    need_birnn = 0 if not os.environ.get("NEED_BIRNN") else int(os.environ.get("NEED_BIRNN"))
    sentiment_drop_ratio = 0.3 if not os.environ.get("SENTIMENT_DROP_RATIO") else float(os.environ.get("SENTIMENT_DROP_RATIO"))
    aspect_drop_ratio = 0.3 if not os.environ.get("ASPECT_DROP_RATIO") else float(os.environ.get("ASPECT_DROP_RATIO"))
    sentiment_in_feature = 768 if not os.environ.get("SENTIMENT_IN_FEATURE") else int(os.environ.get("SENTIMENT_IN_FEATURE"))
    aspect_in_feature = 768 if not os.environ.get("ASPECT_IN_FEATURE") else int(os.environ.get("ASPECT_IN_FEATURE"))
    base_path = "./tmp/model" if not os.environ.get("BASE_PATH") else os.environ.get("BASE_PATH")
    label_info_file = "meta.bin" if not os.environ.get("LABEL_INFO_FILE") else os.environ.get("LABEL_INFO_FILE")
    out_model_path = "pytorch_model.bin" if not os.environ.get("OUT_MODEL_PATH") else os.environ.get("OUT_MODEL_PATH")
    
    if not os.environ.get("POST_SERVER"):
        post_server = "http://localhost:5000/api/review"
        # raise ValueError("POST_SERVER is required.")                
    else:
        post_server = os.environ.get("POST_SERVER")
        
    parser.add_argument("--eval_batch_size", type=int, default=eval_batch_size, help="한 batch에 속할 테스트 데이터 샘플의 size")
    parser.add_argument("--init_model_path", type=str, default=init_model_path, help="사용된 BERT의 종류")
    parser.add_argument("--max_length", type=int, default=max_length, help="토큰화된 문장의 최대 길이(bert는 기본 512)")
    parser.add_argument("--need_birnn", type=int, default=need_birnn, help="model에 Birnn Layer를 추가했는지 여부 (True: 1/False: 0)")
    parser.add_argument("--sentiment_drop_ratio", type=float, default=sentiment_drop_ratio,
                        help="Sentiment 속성의 과적합 방지를 위해 dropout을 수행한 비율")
    parser.add_argument("--aspect_drop_ratio", type=float, default=aspect_drop_ratio,
                        help="Aspect Category 속성의 과적합 방지를 위해 dropout을 수행한 비율")
    parser.add_argument("--sentiment_in_feature", type=int, default=sentiment_in_feature,
                        help="각 Sentiment input sample의 size")
    parser.add_argument("--aspect_in_feature", type=int, default=aspect_in_feature,
                        help="각 Aspect Category input sample의 size")
    parser.add_argument("--base_path", type=str, default=base_path, help="평가를 수행할 Model과 Encoder가 저장된 bucket 경로")
    ### 경로 수정
    parser.add_argument("--label_info_file", type=str, help="사용할 Encoder 파일명", default=label_info_file)
    parser.add_argument("--out_model_path", type=str, help="평가할 model의 파일명", default=out_model_path)
    parser.add_argument("--post_server", type=str, help="평가 결과를 전송할 서버 주소", default=post_server)
    
    config = parser.parse_args()
    
    app = create_app(config)
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)