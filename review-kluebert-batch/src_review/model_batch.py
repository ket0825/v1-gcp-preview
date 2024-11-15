# TODO: Connection timeout 시에 server side cursor가 다시 last_reid 부터 시작하도록 한다 -> 더욱 안전한 방법 생각 필요

import os
from typing import List, Dict
import traceback
import time

import pymysql
from google.cloud import storage
from google.auth import default
import joblib
import pymysql.cursors
import transformers

from modeling.model import ABSAModel
from utils.model_utils import device_setting, load_model
from utils.set_logger import Log
from deploy.pydantic_models import InputReview
from deploy.env_config import EnvConfig
from modeling.trainer import inference_fn, preprocess_fn_deploy
from data_manager.dataset.topic_name_to_code import topic_name_to_code

# test 과정
# 1. local 실행
# 2. batch로 실행.

model = None
log = Log().set_log(log_path="./logs", filename='inference.log', level="DEBUG", ) # only stream.
tokenizer = None
device = None
enc_sentiment, enc_aspect, enc_aspect2, enc_sentiment_score, enc_aspect_score = None, None, None, None, None
config = EnvConfig()


def load_bert_model(config: EnvConfig):
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
    
    log.info(f"match_nv_mid: {review_data[0]['match_nv_mid']}")
    log.info(f"reid first : {review_data[0]['reid']}")
    result = inference_fn(config, review_data, tokenizer, model, enc_sentiment, enc_aspect, enc_aspect2, enc_sentiment_score, enc_aspect_score, device, log)
    
    log.info(f"Result length: {len(result)}")
    log.info(f"first result our topics: {result[0]['our_topics']}")    
    return result


def predict(
        packet_list: List[InputReview]
    ) -> Dict:  
    """    
    class InputReview(BaseModel):
        id:int
        type:str
        prid:str
        caid:str
        reid:str
        content:str    
        our_topics_yn:str
        n_review_id:str
        quality_score:float
        buy_option:str
        star_score:int    
        
        topic_count:int
        topic_yn:str
        topics:Dict    
        
        user_id:str
        aida_modify_time:datetime
        mall_id:str
        mall_seq:str
        mall_name:str
        match_nv_mid:str
        nv_mid:str
        image_urls:Dict
        update_time:datetime
        topic_type:str 
    """                    
    
    packet_dict_list = [packet.model_dump() for packet in packet_list]
    if isinstance(packet_dict_list, list):
        log.info(f"list case: {len(packet_dict_list)}")
    elif isinstance(packet_dict_list, dict): # this case.
        log.info(f"dict case: packet_dict: {packet_dict_list.keys()}")
    else:
        log.info(f"unknown case: {packet_dict_list}")                    
            
    packet_dict_list = inference(config, packet_dict_list) # content_list
            
    return packet_dict_list

def fetch_review_result(
        cursor: pymysql.cursors.DictCursor,
        packet_list: List[Dict]
    ) -> None:
    try:    
        batch_len = len(packet_list)
        reid_in_placeholder = ",".join(["%s"]*batch_len)
        
        sql = f"""
            DELETE FROM cosmos.topic
            WHERE reid IN ({reid_in_placeholder})
            AND type = 'RT0';            
            """                
        
        # batch 단위로 삭제.
        cursor.execute(sql, [packet['reid'] for packet in packet_list])        
            
        valid_topics = []
        for packet in packet_list:
            for topic in packet["our_topics"]:
                topic['prid'] = packet['prid']
                topic_name = topic.get('topic')            
                topic['topic_name'] = topic_name
                topic_code = topic_name_to_code.get(topic_name)
                
                if topic_code is None:
                    log.info(f"[ERROR] Invalid topic: {topic_name}, text: {topic['text']}", f"Invalid topic: {topic_name}, text: {topic['text']}")
                    continue            
                            
                valid_topics.append({
                    'prid': packet['prid'],
                    'type': 'RT0',
                    'topic_code': topic_code,
                    'topic_name': topic_name,
                    'topic_score': topic['topic_score'],
                    'positive_yn': topic['positive_yn'],
                    'sentiment_scale': topic['sentiment_scale'],
                    'text': topic['text'],
                    'start_pos': topic.get('start_pos', 0),
                    'end_pos': topic.get('end_pos', 0),
                })                
        
        if valid_topics:
            valid_topics_len = len(valid_topics)
            log.info(f"valid_topics_len: {valid_topics_len}")
            # column 길이만큼: 10 (prid, type, topic_code, topic_name, topic_score, positive_yn, sentiment_scale, text, start_pos, end_pos) 
            valid_topics_values_placeholder = f'({",".join(["%s"]*10)})'
            # batch로 insert 하기 위한 placeholder
            valid_topics_bulk_placeholer = ",".join([valid_topics_values_placeholder]*valid_topics_len)
            
            sql = f"""
                INSERT INTO cosmos.topic
                (prid, type, topic_code, topic_name, topic_score, positive_yn, sentiment_scale, text, start_pos, end_pos)
                VALUES {valid_topics_bulk_placeholer};
            """ 
            
            flatten_valid_topics = [value for topic in valid_topics for value in topic.values()]
            cursor.execute(sql, flatten_valid_topics)           
                
        ## our_topics_yn = 'Y'로 업데이트.
        # 만약 추론 결과 없어도 업데이트(이미 처리한 것이기 때문)
        sql = f"""
            UPDATE cosmos.review
            SET our_topics_yn = 'Y'
            WHERE reid IN ({reid_in_placeholder});
            """            
        cursor.execute(sql, [packet['reid'] for packet in packet_list])
            
        cursor.connection.commit()            
    except pymysql.err.DataError: 
        log.error(f"[DataError] {traceback.format_exc()}")
        cursor.connection.rollback()
    except:
        log.error(f"{traceback.format_exc()}")
        
        
def main():
    """
    ### 1. DB 연결.
    
    ### 2. topic_type이 다르고, categories 내에 포함된 review를 가져온다.    
    
    ### 3. batch_size 만큼 가져와서 predict.
    
    ### 4. predict 결과를 fetch_review_result로 저장.
    
    ## 설정해야 할 환경변수:    
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
    
    ## DB config
    host: str = os.getenv("DB_HOST")
    user: str = os.getenv("DB_USER", "root")
    password: str = os.getenv("DB_PASSWORD", "1234")
    port: int = os.getenv("PORT", 3306)
    
    ## Batch config
    batch_size: int = os.getenv("BATCH_SIZE", 16)
    topic_type: int = os.getenv("TOPIC_TYPE")
    categories: str = os.getenv("CATEGORIES")  
    
    """
    log.info(f"Batch start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}",)
    
    is_done = False
    last_reid = None
    is_good_conn = True
    while not is_done:    
        # reconnect        
        conn = pymysql.connect(
            host=config.host,
            user=config.user,    
            password=config.password,
            database=config.name,
            port=config.port,
            charset='utf8mb4',
            collation='utf8mb4_general_ci',
            read_timeout=300,
            # cursorclass=pymysql.cursors.DictCursor,
        )
        # 단일 연결에 대한 단일 활성 결과셋이기에, SSDictCursor를 일반 cursor와 같이 사용한다면 connection을 하나 더 생성해야 한다.
        conn_ss = pymysql.connect(
            host=config.host,
            user=config.user,    
            password=config.password,
            database=config.name,
            port=config.port,
            charset='utf8mb4',
            collation='utf8mb4_general_ci',
            read_timeout=300,
            # cursorclass=pymysql.cursors.SSDictCursor
        )
        
        batch_size = config.batch_size
        topic_type = config.topic_type
        categories = config.categories            
        # In Pydantic, string values enclosed in single quotes are treated as a single string.
        categories = "'" \
            + "','".join(categories.split(","))  \
            + "'" 
        try:
            with conn_ss.cursor(cursor=pymysql.cursors.DictCursor) as ss_cursor, \
                conn.cursor(cursor=pymysql.cursors.DictCursor) as sub_cursor:
                sql = f"""
                    SELECT caid FROM cosmos.category
                    WHERE s_category IN ({categories});
                    """   
                sub_cursor.execute(sql)
                
                caids = sub_cursor.fetchall()
                caids = [caid['caid'] for caid in caids]
                caids_text = "'" + \
                    "','".join([caid for caid in caids]) \
                        + "'"
                        
                
                # 해당하는 카테고리와 topic_type이 다른 리뷰를 가져온다.
                # topic_type이 모두 같다면 실행되지 않는다. (후에 caid indexing 필요할 수도. 아니 무조건 추가해야 함.)
                sql = f"""
                    SELECT * FROM cosmos.review
                    WHERE                   
                    caid IN ({caids_text})                    
                    AND (topic_type NOT LIKE '{topic_type}' OR isnull(topic_type))                    
                    ;
                    """    
                t1 = time.time()
                ss_cursor.execute(sql) 
                log.info(f"Fetch time: {time.time()-t1}")
                
                while True:
                    batch = ss_cursor.fetchmany(batch_size)
                    
                    # connection timeout 시에 server side cursor가 다시 last_reid부터 시작하도록 한다.
                    if not is_good_conn and last_reid:
                        reid_list = [row['reid'] for row in batch]
                        if last_reid not in reid_list:
                            continue
                        else:
                            last_reid = None
                                                                        
                    if not batch:
                        is_done = True
                        break
                    # preprocess.                
                    batch = preprocess_fn_deploy(batch)
                    # 여기서는 batch 단위로 predict.
                    packet_list = predict([InputReview(**row) for row in batch]) # validation
                    fetch_review_result(sub_cursor, packet_list)
                    last_reid = batch[-1]['reid']
                    is_good_conn = True
                
            log.info(f"Batch done: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}",)
        
        except pymysql.err.OperationalError as e:
            if e.args[0] == 2013:
                log.error(f"[OperationalError] Lost Connection {traceback.format_exc()}")
                is_good_conn = False
            else:
                log.error(f"[OperationalError] {traceback.format_exc()}")                                                    
        except Exception as e:        
            is_done = True
            log.error(f"{traceback.format_exc()}")
                    
        finally:
            # ss_cursor.fetchall() # ss_cursor close.
            conn.close()
            conn_ss.close()
        

if __name__ == '__main__':    
    load_bert_model(config)    
    main()
    
    
    