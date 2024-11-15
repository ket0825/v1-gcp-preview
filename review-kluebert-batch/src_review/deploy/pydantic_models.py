from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Optional, Any, Union
import json
import base64
from datetime import datetime

class InputReview(BaseModel):
    id:int
    type:str
    prid:str
    caid:str
    reid:str
    content:str    
    our_topics_yn:str
    n_review_id:str
    quality_score:float
    buy_option:Union[str, None]
    star_score:int    
    
    topic_count:int
    topic_yn:str
    topics:Union[Dict, List, None]    
    
    user_id:str
    aida_modify_time:datetime
    mall_id:str
    mall_seq:str
    mall_name:str
    match_nv_mid:str
    nv_mid:str
    image_urls:Union[Dict, List, None]
    
    update_time:datetime
    topic_type:Union[str, None]

    # 임시 column
    sent_list_preprocessed: List
    
        
# class KlueBertReviewRequest(BaseModel):
#     type:str
#     category:str
#     prid: str
#     match_nv_mid:str
#     reviews:List[InputReview]
    

# class Message(BaseModel):
#     model_config = ConfigDict(extra='ignore')    
    
#     data: str
#     attributes: Optional[Dict[str, str]] = None
#     message_id: str
#     publish_time: str
#     ordering_key: Optional[str] = None        

#     @property
#     def decoded_data(self) -> Dict[str, Any]:
#         return json.loads(base64.b64decode(self.data).decode('utf-8-sig'))
    

# class PubsubRequest(BaseModel):
#     message: Message
#     subscription: str  
    
# class OutputReviews(InputReviews):
#     our_topics: List[Dict]
        
# class KlueBertReviewResponse(BaseModel):  
#     type:str
#     category:str
#     match_nv_mid:str
#     reviews:List[OutputReviews]
 
    
