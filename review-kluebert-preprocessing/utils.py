import re
from kss import split_sentences

more_than_one_space = re.compile(r'\s{2,}')
pattern1 = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]+") # 한글 자모음만 반복되면 삭제
pattern2 = re.compile(r":\)|[\@\#\$\^\*\(\)\[\]\{\}\<\>\/\"\'\=\+\\\|\_(:\));]+") # ~, !, %, &, -, ,, ., ;(얘는 제거함), :, ?는 제거 X /// 특수문자 제거
pattern3 = re.compile(r"([^\d])\1{2,}") # 숫자를 제외한 동일한 문자 3개 이상이면 삭제
emoticon_pattern = re.compile(r'[:;]-?[()D\/]')
pattern4 = re.compile( # 이모티콘 삭제
    "["                               
    "\U0001F600-\U0001F64F"  # 감정 관련 이모티콘
    "\U0001F300-\U0001F5FF"  # 기호 및 픽토그램
    "\U0001F680-\U0001F6FF"  # 교통 및 지도 기호
    "\U0001F1E0-\U0001F1FF"  # 국기
    "]+", flags=re.UNICODE)
pattern5 = re.compile(r"([~,!%&-.]){2,}") # 특수문자는 동일한 문자 2개 이상이면 삭제

def regexp_text(text):
    replaced_str = ' '    
    new_text = pattern1.sub(replaced_str, text)
    new_text = pattern2.sub(replaced_str, new_text)
    new_text = pattern3.sub(replaced_str, new_text)        
    new_text = emoticon_pattern.sub(replaced_str, new_text)
    new_text = pattern4.sub(replaced_str, new_text)
    new_text = pattern5.sub(replaced_str, new_text)
    new_text = new_text.replace('  ', ' ').strip()
    return new_text.replace('\n', ' ').strip()

def split_content_into_sentences(content):
    sentences = split_sentences(content, backend='mecab')    
    return ' '.join([sent.strip() + "." for sent in sentences if sent.strip()])