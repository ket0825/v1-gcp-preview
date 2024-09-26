import json
import os
import functions_framework
import urllib.request
from flask import jsonify

from utils import preprocess_seller_spec

project_id = os.environ["GCP_PROJECT"]
OCR_LB_ENDPOINT = os.environ["OCR_LB_ENDPOINT"]


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
    
    ocr_length = len(request_json['seller_spec'])    
    print(f"ocr length: {ocr_length}")
    message_data = json.dumps(request_json, ensure_ascii=False).encode("utf-8-sig")        
    req = urllib.request.Request(OCR_LB_ENDPOINT, data=message_data, method="POST")
    req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req) as response:            
            res_data = response.read().decode('utf-8')
            print(f"Response: {res_data}")            
    except Exception as e:
        print(f"Error in OCR Cloud Functions: {e}")                                        
    
    return jsonify({"message": f"OCR length: {ocr_length}", "status":200}), 200
    