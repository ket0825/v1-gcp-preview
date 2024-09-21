from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Optional, Any
import json
import base64

class InputOCR(BaseModel):    
    prid: str
    caid: str         
    grade: float
    name: str
    lowest_price:int
    review_count:int
    url: str
    brand: str
    maker: str
    naver_spec: Dict
    seller_spec: List[Dict]
    detail_image_urls: List[str]    

class Message(BaseModel):
    model_config = ConfigDict(extra='ignore')    
    
    data: str
    attributes: Optional[Dict[str, str]] = None
    message_id: str
    publish_time: str
    ordering_key: Optional[str] = None        

    @property
    def decoded_data(self) -> Dict[str, Any]:
        return json.loads(base64.b64decode(self.data).decode('utf-8-sig'))
    
class PubsubRequest(BaseModel):
    message: Message
    subscription: str  

    
