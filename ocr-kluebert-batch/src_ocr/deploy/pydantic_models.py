from pydantic import BaseModel, ConfigDict
from datetime import datetime

from typing import List, Dict, Optional, Any, Union
import json
import base64

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

    
