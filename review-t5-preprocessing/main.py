import json
import os
import functions_framework
from google.cloud import pubsub_v1
from flask import jsonify

from utils import split_content_into_sentences

publisher = pubsub_v1.PublisherClient()
project_id = os.environ["GCP_PROJECT"]


@functions_framework.http
def review_model_t5_preprocessing(request):
    print("t5-preprocessing")
    request_json = request.get_json(silent=True)    

    if ("type" not in request_json 
        or "category" not in request_json 
        or "prid" not in request_json 
        or "match_nv_mid" not in request_json 
        or "reviews" not in request_json
        ):
        raise ValueError(f"JSON is invalid. Keys: {request_json.keys()}")       

    for review in request_json['reviews']:
        review['content'] = split_content_into_sentences(review['content'])  # 이후 \n으로 구분된 문장으로 변경
    

    topic_name = os.environ["TOPIC_REVIEW_T5"]
    message_data = json.dumps(request_json, ensure_ascii=False).encode("utf-8-sig")
    topic_path = publisher.topic_path(project_id, topic_name)
    print(f"Publishing message to {topic_name}")
    future = publisher.publish(topic_path, data=message_data)
    future.result()


    # 여기서 publish 해야 함. 일단 이것부터 테스트.
    return {"status": 200}
    