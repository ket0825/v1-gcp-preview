import json
import os
import functions_framework
from google.cloud import pubsub_v1
from concurrent import futures
from typing import Callable
from flask import jsonify

from utils import preprocess_seller_spec

publisher = pubsub_v1.PublisherClient()
project_id = os.environ["GCP_PROJECT"]
publish_futures = []

def get_callback(
    publish_future: pubsub_v1.publisher.futures.Future, data: str
) -> Callable[[pubsub_v1.publisher.futures.Future], None]:
    def callback(publish_future: pubsub_v1.publisher.futures.Future) -> None:
        try:
            # Wait 60 seconds for the publish call to succeed.
            publish_future.result(timeout=60)
        except futures.TimeoutError:
            print(f"Publishing {data} timed out.")
                
    return callback

@functions_framework.http
def ocr_kluebert_preprocessing(request):
    
    if request.method != "POST":
        return jsonify({"status": 405, "message": f"{request.method} Method not allowed"}), 405
    
    print("OCR kluebert-preprocessing")            
    request_json = request.get_json(silent=True)    

    if (
        "prid" not in request_json
        or "caid" not in request_json
        or "grade" not in request_json
        or "name" not in request_json
        or "lowest_price" not in request_json
        or "review_count" not in request_json
        or "url" not in request_json
        or "brand" not in request_json
        or "maker" not in request_json
        or "naver_spec" not in request_json
        or "seller_spec" not in request_json
        or "detail_image_urls" not in request_json
        ):
        raise ValueError(f"JSON is invalid. Keys: {request_json.keys()}")       

    # seller spec preprocessing.
    request_json['seller_spec'] = preprocess_seller_spec(request_json['seller_spec'])    
    
    topic_name = os.environ["TOPIC_NAME"]
    print(f"Publishing message to {topic_name}")    
    topic_path = publisher.topic_path(project_id, topic_name)
    print(f"Publishing path: {topic_path}")
    
    ocr_length = len(request_json['seller_spec'])    
    print(f"ocr length: {ocr_length}")
    message_data = json.dumps(request_json, ensure_ascii=False).encode("utf-8-sig")
    publish_future = publisher.publish(topic_path, data=message_data)            
    publish_future.add_done_callback(get_callback(publish_future, message_data))
    publish_futures.append(publish_future)    
        
    futures.wait(publish_futures, return_when=futures.ALL_COMPLETED)            
    print("[INFO] All messages published.")
    
    return jsonify({"message": f"OCR length: {ocr_length}", "status":200}), 200
    