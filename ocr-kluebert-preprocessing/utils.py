import re
from kss import split_sentences

pattern1 = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]+") # 한글 자모음만 반복되면 삭제
pattern2 = re.compile(r":\)|[\@\#\$\^\*\(\)\[\]\{\}\<\>\/\"\'\=\+\\\|\_(:\));]+") # ~, !, %, &, -, ,, ., :, ?는 제거 X /// 특수문자 제거
pattern3 = re.compile(r"([^\d])\1{2,}") # 숫자를 제외한 동일한 문자 3개 이상이면 삭제
emoticon_pattern = re.compile(r'[:;]-?[()D\/]')
pattern4 = re.compile( # 이모티콘 삭제
    "["                               
    "\U0001F600-\U0001F64F"  # 감정 관련 이모티콘
    "\U0001F300-\U0001F5FF"  # 기호 및 픽토그램
    "\U0001F680-\U0001F6FF"  # 교통 및 지도 기호
    "\U0001F1E0-\U0001F1FF"  # 국기
    # "\U00002702-\U000027B0"  # 기타 기호
    # "\U000024C2-\U0001F251"  # 추가 기호 및 픽토그램      # 이거 2줄까지 하면 한글이 사라짐
    "]+", flags=re.UNICODE)

def regexp(sentences):
    replaced_str = ' '
    for i in range(len(sentences)):
        sent = sentences[i]
        new_sent = pattern1.sub(replaced_str, sent)
        new_sent = pattern2.sub(replaced_str, new_sent)
        new_sent = pattern3.sub(replaced_str, new_sent)        
        new_sent = emoticon_pattern.sub(replaced_str, new_sent)
        new_sent = pattern4.sub(replaced_str, new_sent)
        new_sent = new_sent.replace('!.', '!').replace('?.', '?')

        sentences[i] = new_sent.strip()

    return sentences


def preprocess_img_spec(img_str:str):   
    kss_sent_list = split_sentences(img_str, backend='mecab')    
    text = '\n'.join(kss_sent_list)
    sentences = text.split('\n')
    sentences = [[sent] for sent in sentences]
    sentences_period_added = [[sent[0].strip().replace('\n', " ") + '.'] for sent in sentences if sent[0].strip()]
    sentences_regexp = [regexp(sent_list) for sent_list in sentences_period_added]
    preprocessed_img_str = ' '.join([''.join(row) for row in sentences_regexp])    
    return text, preprocessed_img_str, sentences_regexp

def preprocess_seller_spec(seller_spec:list):
    """

    Args:
        seller_spec (list): 
        [
            {
            "img_str": "",
            "bbox_text": []
            },
        ]

    Returns:
        seller_spec (list): 
        [
            {
            "img_str": "",
            "img_str_preprocessed": "",
            "img_sent_list_preprocessed": [[문장...]]
            "bbox_text": []
            },
        ]
    """    
    for img_attr in seller_spec:
        img_attr['img_str'], img_attr['img_str_preprocessed'], img_attr['img_sent_list_preprocessed'] \
            = preprocess_img_spec(img_attr['img_str'])    
    return seller_spec