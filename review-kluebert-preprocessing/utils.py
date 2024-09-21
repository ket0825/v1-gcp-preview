import re
from kss import split_sentences
from typing import List, Dict
import pandas as pd

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
whitespace_pattern = re.compile(r'\s+') # 빈칸 여러개 무조건 1개로 고정시키기 위한 pattern

def normalize_whitespace(content):
    # 정규 표현식을 사용하여 연속된 공백을 단일 공백으로 변환
    normalized_content = whitespace_pattern.sub(' ', content)
    return normalized_content

def replace_newline(text):
    return text.replace('\n', ' ')

def preprocess_content(content):
    sentences = split_sentences(content)
    sentences = [[sent] for sent in sentences]
    sentences_regexp = [regexp(sent_list) for sent_list in sentences]
    sentences_period_added = [replace_newline(sent[0].strip()) + '.' for sent in sentences_regexp if sent[0].strip()]
    sentences_period_added = [[sent] for sent in sentences_period_added]
    preprocessed_content = ' '.join([''.join(row) for row in sentences_period_added])
    return preprocessed_content, sentences_period_added


def regexp(sentences):
    replaced_str = ' '
    for i in range(len(sentences)):
        sent = sentences[i]
        new_sent = pattern1.sub(replaced_str, sent)
        new_sent = pattern2.sub(replaced_str, new_sent)
        new_sent = pattern3.sub(replaced_str, new_sent)        
        new_sent = emoticon_pattern.sub(replaced_str, new_sent)
        new_sent = pattern4.sub(replaced_str, new_sent)       
        sentences[i] = new_sent

    return sentences


def preprocess_fn_deploy(
        review_list:List[Dict]
    ) -> List[Dict]:
    
    """preprocess data['review']

    Args:
        config (_type_): Config
        data (List[Dict]): data['review']
    """

    
    df = pd.DataFrame(review_list)    
    df.loc[:, "content"] = df["content"].fillna(method="ffill")

    df["temp"] = df['content'].apply(preprocess_content)
    df['content'] = df['temp'].apply(lambda x: x[0])
    df['sent_list_preprocessed'] = df['temp'].apply(lambda x: x[1])
    df.drop(['temp'], axis=1, inplace=True)

    df = df[df['content'] != ''] # 빈 텍스트 삭제(의미 없으니까)
    df.reset_index(drop=True, inplace=True)

    df['content'] = df['Acontent'].apply(normalize_whitespace) # spacing 문제 해결 위해서 공백은 무조건 1칸으로 고정

    # our_topics 있는 경우에 같이 저장.
    try:
        df = df[['id', 'content', 'sent_list_preprocessed', 'aidaModifyTime', 'mallId', 'mallSeq', 
            'matchNvMid', 'nvMid', 'qualityScore', 'starScore', 'topicCount', 'topicYn', 'topics' ,
            'userId', 'mallName', 'our_topics']]
        
    except KeyError as e:
        df = df[['id', 'content', 'sent_list_preprocessed', 'aidaModifyTime', 'mallId', 'mallSeq', 
            'matchNvMid', 'nvMid', 'qualityScore', 'starScore', 'topicCount', 'topicYn', 'topics' ,
            'userId', 'mallName']]
    
    df["mallSeq"] = df["mallSeq"].astype(str)
    df["matchNvMid"] = df["matchNvMid"].astype(str)
    df["nvMid"] = df["nvMid"].astype(str)
    
    preprocessed_review_list = df.to_dict(orient='records')        
    print('finish')
    return preprocessed_review_list