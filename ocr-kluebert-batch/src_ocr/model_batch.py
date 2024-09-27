# TODO: 1. model inference 시에 필요한 column이 있어야 함. (ex sent_list_processed)
# TODO: 1.1 일단 Camel case를 snake case로 변경 필요. (ocr은 snake case로 되어 있음.)
# TODO: 1.2 그러기 위하여 전처리 로직 추가 필요. (o)

import os
import json
import traceback

import pymysql

from typing import List, Dict, Union

from google.cloud import storage
from google.auth import default
import joblib
import transformers

from modeling.model import ABSAModel
from utils.model_utils import device_setting, load_model
from utils.set_logger import Log
from deploy.pydantic_models import InputOCR
from deploy.env_config import EnvConfig
from modeling.trainer import inference_fn, preprocess_seller_spec

from data_manager.dataset.topic_name_to_code import topic_name_to_code


# test 과정
# 1. local 실행
# 2. batch로 실행.

model = None
log = Log().set_log(level="DEBUG", log_path="./logs/", filename="inference.log") # only stream.
tokenizer = None
device = None
enc_aspect, enc_aspect2,= None, None
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
        log.info(f"Current Project ID: {project_id}")
        
        # # 버킷 목록 조회
        buckets = storage_client.list_buckets()

        log.info("Buckets:")
        for bucket in buckets:
            log.info(f" - {bucket.name}")
        
        #  # 특정 버킷의 객체(파일) 목록 조회    
        bucket = storage_client.get_bucket(bucket_name)
        blobs = bucket.list_blobs()
        
        log.info(f"\nObjects in bucket '{bucket_name}':")
        for blob in blobs:
            log.info(f" - {blob.name}")
        
        bucket = storage_client.bucket(bucket_name)
        model_blob = bucket.blob(model_file)
        metadata_blob = bucket.blob(metadata_file)                
        
        os.makedirs('./tmp', exist_ok=True)
        
        log.info(f"Downloading model to {local_model_path}")     
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
    
    global enc_aspect, enc_aspect2
    enc_aspect, enc_aspect2 = metadata["enc_aspect"], metadata["enc_aspect2"]    
        
    num_aspect, num_aspect2 = len(list(metadata["enc_aspect"].classes_)), len(list(metadata["enc_aspect2"].classes_))    
    
    global device
    device = device_setting(log)
    global model
    
    model = ABSAModel(
        config=config,
        num_aspect=num_aspect,
        num_aspect2=num_aspect2,
        need_birnn=bool(config.need_birnn)
        )
    model = load_model(model=model, state_dict_path=local_model_path, device=device)
    global tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(config.init_model_path, do_lower_case=False)

def inference(
    config, 
    seller_spec:List[Dict]
              ) -> List[Dict]: # + our_topics
    global model
    global tokenizer
    global log
    global device
    global enc_aspect, enc_aspect2
    
    log.info(f"seller_spec: {seller_spec[0]['img_str']}")  
    # 임시조치: seller_spec의 img_str이 공백인 경우가 있음. 이 경우는 제외함.
    seller_spec = [spec for spec in seller_spec if spec['img_str'] != ""]
            
    result = inference_fn(config, seller_spec, tokenizer, model, enc_aspect, enc_aspect2, device, log)
    
    # clean result
    result = [
            {
                'img_str': spec['img_str'],
                'our_topics': spec['our_topics'],
                'bbox_text': spec['bbox_text'],
            }
            for spec in result
        ]
    
    log.info(f"Img length: {len(result)}")
    log.info(f"Total our topics length: {sum(len(img['our_topics']) for img in result)}")    
    return result


# def predict(packet: PubsubRequest): 
def predict(
        packet: InputOCR
    ) -> Dict:  # local test 및 외부 로드밸런서용.
    """    
    class InputOCR(BaseModel):    
        type: str
        prid: str
        caid: str         
        grade: float
        name: str
        lowest_price:int
        review_count:int
        url: str
        brand: Union[str, None]
        maker: Union[str, None]
        naver_spec: Union[Dict, List, None]
        seller_spec: List[Dict]
        detail_image_urls: Union[Dict, List, None]
        topic_type: Union[str, None]
        update_time: datetime
    """                    
    
    packet_dict = packet.model_dump()
    if isinstance(packet_dict, list):
        log.info(f"list case: {packet_dict}")
    elif isinstance(packet_dict, dict): # this case.
        log.info(f"dict case: packet_dict: {packet_dict.keys()}")
    else:
        log.info(f"unknown case: {packet_dict}")        
    try:        
        packet_dict['seller_spec'] = inference(config, packet_dict['seller_spec']) # content_list    
        
                
        return packet_dict
    except Exception as e:        
        log.error(f"[ERROR]: {e}, at {packet_dict['prid']} \n {traceback.format_exc()}")

def fetch_ocr_result(
        cursor,
        packet_dict: Dict
    ) -> None:
    try:    
        # TODO: 이 부분은 추후에 수정 필요. 현재는 type OT0로 하드코딩.                    
        sql = f"""
            DELETE FROM cosmos.topic
            WHERE prid = '{packet_dict['prid']}'
            AND type = 'OT0';
            """    
        
        cursor.execute(sql)
        
        our_topics = []
        for img_spec in packet_dict['seller_spec']:
            if (img_spec["img_str"] != "" 
                and img_spec.get("our_topics")
                ):            
                our_topics.append(img_spec["our_topics"])
        
        if our_topics:
            valid_topics = []
            for idx, image_topics in enumerate(our_topics):                    
                for topic in image_topics:
                    topic['prid'] = packet_dict['prid']
                    topic_name = topic.get('topic')            
                    topic['topic_name'] = topic_name
                    topic_code = topic_name_to_code.get(topic_name)
                
                    if topic_code is None:
                        log.info(f"[ERROR] Invalid topic: {topic_name}, text: {topic['text']}", f"Invalid topic: {topic_name}, text: {topic['text']}")
                        continue            
                            
                    valid_topics.append({
                        'prid': packet_dict['prid'],
                        'type': 'OT0',
                        'topic_code': topic_code,
                        'topic_name': topic_name,
                        'image_number': idx,
                        'text': topic['text'],
                        'start_pos': topic.get('start_pos', 0),
                        'end_pos': topic.get('end_pos', 0),
                        'bbox': topic.get('bbox', '')
                    })
                
            if valid_topics:
                
                sql = f"""
                    INSERT INTO cosmos.topic
                    (prid, type, topic_code, topic_name, image_number, text, start_pos, end_pos, bbox)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                """
                
                cursor.executemany(sql, [
                    (
                        topic['prid'],
                        topic['type'],
                        topic['topic_code'],
                        topic['topic_name'],
                        topic['image_number'],
                        topic['text'],
                        topic['start_pos'],
                        topic['end_pos'],
                        json.dumps(topic['bbox'], ensure_ascii=False)
                    )
                    for topic in valid_topics
                ])
        
        cursor.connection.commit() 
    except Exception as e:
        log.error(f"[ERROR] fetching predict OCR topic: {e} \n at {packet_dict['prid']} \n{traceback.format_exc()}")
        cursor.connection.rollback()                   

def sanatize_text(text:str, is_seller_spec=False) -> Union[Dict, List, None]:
    try:
        json_type = json.loads(text)         
    # text가 None인 경우.
    except TypeError:        
        return None
    
    if not json_type:
        return None
    
    # 구 ocr 버전은 지원 안함.
    if (is_seller_spec 
        and isinstance(json_type[0], list)
        ):
        return None

    return json_type
               

def main():    
    conn = pymysql.connect(
        host=config.host,
        user=config.user,    
        password=config.password,
        database=config.name,
        port=config.port,
        charset='utf8mb4',
        collation='utf8mb4_general_ci',
        cursorclass=pymysql.cursors.DictCursor,     
    )
    
    batch_size = config.batch_size
    topic_type = config.topic_type
    categories = config.categories

    try:
        with conn.cursor() as cursor, conn.cursor() as sub_cursor:
            sql = f"""
                SELECT caid FROM cosmos.category
                WHERE s_category IN ({categories});
                """   
            cursor.execute(sql)
            
            caids = cursor.fetchall()
            caids_text = "'" + \
                "','".join([caid['caid'] for caid in caids]) \
                    + "'"
            #TODO: cursor timeout 시에 재연결 로직 추가 필요.
            sql = f"""
                SELECT * FROM cosmos.product
                WHERE caid IN ({caids_text}) 
                AND (topic_type NOT LIKE '{topic_type}' OR isnull(topic_type));
                """    
            cursor.execute(sql)
            iter_count = 0
            while True:
                # conn.ping(reconnect=True)
                batch = cursor.fetchmany(batch_size)
                iter_count += 1 
                
                if not batch:
                    break                                                
                
                if not iter_count > 4: # To Find error
                    continue
                
                for row in batch:        
                        
                    row['seller_spec'] = sanatize_text(row['seller_spec'], is_seller_spec=True)
                    row['naver_spec'] = sanatize_text(row['naver_spec'])                    
                    row['detail_image_urls'] = sanatize_text(row['detail_image_urls'])
                    
                    
                    if row['seller_spec'] is None:
                        continue                
                    
                    row['seller_spec'] = preprocess_seller_spec(row['seller_spec']) # preprocess_function
                    
                    packet_dict = predict(InputOCR(**row)) # validation                    
                    fetch_ocr_result(sub_cursor, packet_dict)                    
    
    except Exception as e:                
        log.error(f"[ERROR] {e} at batch {batch[0]['prid']} \n {traceback.format_exc()}")
                    
    finally:
        conn.close()
        

if __name__ == '__main__':    
    load_bert_model(config)    
    main()