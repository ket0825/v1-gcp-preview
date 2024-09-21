import json
import os
import urllib.request
import functions_framework
from google.cloud import pubsub_v1
from concurrent import futures
from typing import Callable
from flask import jsonify
import urllib

import base64

from utils import preprocess_fn_deploy


project_id = os.environ["GCP_PROJECT"]
OCR_LB_ENDPOINT = os.environ["OCR_LB_ENDPOINT"]


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
        
    request_json['reviews'] = preprocess_fn_deploy(request_json['reviews'])
        
    response_list = []
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
        
        req = urllib.request.Request(OCR_LB_ENDPOINT, data=message_data, method="POST")
        req.add_header('Content-Type', 'application/json')
        try:
            with urllib.request.urlopen(req) as response:            
                res_data = response.read().decode('utf-8')
                print(f"Response: {res_data}")
                response_list.append(res_data)
        except Exception as e:
            print(f"Error in OCR Cloud Functions: {e}")                            
            response_list.append(f"Error in OCR Cloud Functions: {e}")
        
        review_count += max_review_length                
    
    return jsonify(
        {
            "message": f"Review length: {review_length}", 
            "response": "\n".join(response_list),
            "status":200
        }), 200
    