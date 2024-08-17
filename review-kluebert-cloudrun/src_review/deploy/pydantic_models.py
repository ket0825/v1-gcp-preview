from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Optional, Any
import json
import base64

class InputReviews(BaseModel):
    matchNvMid: str
    id: str
    aidaModifyTime: str
    content: str
    mallId: str
    mallSeq: str
    nvMid: str
    qualityScore: float
    starScore: int
    topicCount: int
    topicYn: str
    topics: List[Dict]
    userId: str
    mallName: str    
    
class KlueBertReviewRequest(BaseModel):
    type:str
    category:str
    match_nv_mid:str
    reviews:List[InputReviews]    
    

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
    
class OutputReviews(InputReviews):
    our_topics: List[Dict]
        
class KlueBertReviewResponse(BaseModel):  
    type:str
    category:str
    match_nv_mid:str
    reviews:List[OutputReviews]
 
    
