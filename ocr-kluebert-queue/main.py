import json
import os
import functions_framework
from cloudevents.http import CloudEvent
from flask import jsonify
from urllib import request
import base64

endpoint_url = os.environ['ENDPOINT_URL']


@functions_framework.cloud_event
def ocr_kluebert_queue(cloud_event: CloudEvent):
    expected_type = "google.cloud.pubsub.topic.v1.messagePublished"
    received_type = cloud_event['type']
    if received_type != expected_type:
        raise ValueError(
            f"Expected event type {expected_type} but received {received_type}"
        )        
    
    print("OCR kluebert queue")            
    
    data = cloud_event.data["message"]["data"]
    data = base64.b64decode(data).decode('utf-8')
    data = json.loads(data)        
    

    if (
        "prid" not in data
        or "caid" not in data
        or "grade" not in data
        or "name" not in data
        or "lowest_price" not in data
        or "review_count" not in data
        or "url" not in data
        or "brand" not in data
        or "maker" not in data
        or "naver_spec" not in data
        or "seller_spec" not in data
        or "detail_image_urls" not in data
        ):
        raise ValueError(f"Data is invalid. Keys: {data.keys()}")           
    
    message_data = json.dumps(data).encode('utf-8')
    req = request.Request(endpoint_url, data=message_data, method="POST")
    req.add_header('Content-Type', 'application/json')
    
    with request.urlopen(req) as response:
        res_data = response.read().decode('utf-8')
        print(f"Response: {res_data}")
        return jsonify({"message": f"{res_data}", "status":200}), 200
    