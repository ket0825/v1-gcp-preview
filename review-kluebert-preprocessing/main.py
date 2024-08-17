import json
import os
import functions_framework
from google.cloud import pubsub_v1
from concurrent import futures
from typing import Callable
from flask import jsonify

from utils import split_content_into_sentences

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
def review_kluebert_preprocessing(request):
    
    if request.method != "POST":
        return jsonify({"status": 405, "message": f"{request.method} Method not allowed"}), 405
    
    print("kluebert-preprocessing")            
    request_json = request.get_json(silent=True)    

    if (
        "type" not in request_json 
        or "category" not in request_json 
        or "prid" not in request_json 
        or "match_nv_mid" not in request_json 
        or "reviews" not in request_json
        ):
        raise ValueError(f"JSON is invalid. Keys: {request_json.keys()}")       

    for review in request_json['reviews']:
        review['content'] = split_content_into_sentences(review['content'])    
    
    topic_name = os.environ["TOPIC_NAME"]
    print(f"Publishing message to {topic_name}")    
    topic_path = publisher.topic_path(project_id, topic_name)
    print(f"Publishing path: {topic_path}")
    
    max_review_length = int(os.environ.get("MAX_REVIEW_LENGTH"))
    review_length = len(request_json['reviews'])    
    review_count = 0
    while review_length > review_count:        
        request_json_chunk = {
            "type": request_json['type'],
            "category": request_json['category'],
            "match_nv_mid": request_json['match_nv_mid'],
            "reviews": request_json['reviews'][review_count:review_count+max_review_length]
        }                
        print(f"review length: {len(request_json_chunk['reviews'])}")
        message_data = json.dumps(request_json_chunk, ensure_ascii=False).encode("utf-8-sig")            
        publish_future = publisher.publish(topic_path, data=message_data)            
        publish_future.add_done_callback(get_callback(publish_future, message_data))
        publish_futures.append(publish_future)
        review_count += max_review_length
        
    futures.wait(publish_futures, return_when=futures.ALL_COMPLETED)            
    print("[INFO] All messages published.")
    
    return jsonify({"message": f"Review length: {review_length}", "status":200}), 200
    