from pathlib import Path
import os
import joblib
from sklearn import preprocessing
from sklearn.utils import column_or_1d
from collections import OrderedDict
import pandas as pd
import transformers
import torch
from torch.utils.data import IterableDataset
from utils.file_io import get_file_list, read_csv
from data_manager.parsers.label_unification.label_map import label_list, label_changing_rule
import math


# Encoder - Sentiment 및 Aspect 속성을 Encoding할 Label Encoder class
class MyLabelEncoder(preprocessing.LabelEncoder):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self


class Encoder:
    def __init__(self, config, fp=None, extension="csv"):

        self.fp = fp
        self.file_list = get_file_list(fp, extension)
        self.meta_data_fp = os.path.join(config.base_path + config.label_info_file)

        # Sentiment 속성을 위한 Encoder 및 Aspect Category를 위한 Encoder
        self.enc_sentiment, self.enc_aspect, self.enc_aspect2 = None, None, None
        self.enc_sentiment_score, self.enc_aspect_score=None, None ##### 스코어 두개 추가
        
        self.sentiment_labels = ['PAD', 'O']
        
        self.aspect_labels, self.aspect2_labels = ['PAD', 'O'], ['PAD', 'O']
        self.sentiment_score_labels=['PAD', 'O'] #### 스코어 레이블  두개 추가
        self.aspect_score_labels=['PAD', 'O'] #### 스코어 레이블  두개 추가

    # 저장된 Encoder 유무를 확인
    # 있으면 load, 없으면 생성
    def check_encoder_fp(self):
        meta_data_f = Path(self.meta_data_fp)
        if meta_data_f.exists():
            meta_data = joblib.load(self.meta_data_fp)
            self.enc_sentiment = meta_data["enc_sentiment"]
            self.enc_aspect, self.enc_aspect2 = meta_data["enc_aspect"], meta_data["enc_aspect2"]  
            self.enc_sentiment_score, self.enc_aspect_score = meta_data["enc_sentiment_score"], meta_data["enc_aspect_score"]#### 스코어 두개 추가
        else:
            meta_data = self.set_encoder()
            joblib.dump(meta_data, self.meta_data_fp)

    def get_encoder(self):
        if (self.enc_sentiment is None 
            or self.enc_aspect is None 
            or self.enc_aspect2 is None 
            or self.enc_sentiment_score is None 
            or self.enc_aspect_score is None
            ):
            self.check_encoder_fp()
        return self.enc_aspect, self.enc_aspect2, self.enc_sentiment,  self.enc_aspect_score, self.enc_sentiment_score #### 스코어 두개 추가

    def set_encoder(self):
        if len(self.file_list) == 0:
            print(f"파일 경로 {self.fp}에 Encoding할 데이터가 존재하지 않습니다.")
            raise FileExistsError()

        for now_fp in self.file_list:
            df = read_csv(now_fp)
            self.aspect_labels.extend(list(df["Aspect"].unique()))
            self.sentiment_labels.extend(list(df["Sentiment"].unique()))
    
            self.aspect_score_labels.extend(list(df["Aspect_Score"].unique())) # 스코어 두개 레이블 리스트
            self.sentiment_score_labels.extend(list(df["Sentiment_Score"].unique())) # 스코어 두개 레이블 리스트
    

        # Encoder 선언 및 fitting
        self.enc_sentiment, self.enc_aspect, self.enc_aspect2 = MyLabelEncoder(), MyLabelEncoder(), MyLabelEncoder()
        self.enc_aspect_score, self.enc_sentiment_score = MyLabelEncoder(), MyLabelEncoder() #### 스코어 두개 추가
        
        self.aspect_labels = list(OrderedDict.fromkeys(self.aspect_labels))
        self.sentiment_labels = list(OrderedDict.fromkeys(self.sentiment_labels))
        self.sentiment_score_labels = list(OrderedDict.fromkeys(self.sentiment_score_labels))#### 스코어 두개 추가
        self.aspect_score_labels = list(OrderedDict.fromkeys(self.aspect_score_labels))#### 스코어 두개 추가
        self.aspect2_labels.extend([label for label in label_list])

        self.enc_aspect = self.enc_aspect.fit(self.aspect_labels)
        self.enc_aspect2 = self.enc_aspect2.fit(self.aspect2_labels)
        self.enc_sentiment = self.enc_sentiment.fit(self.sentiment_labels)
        self.enc_sentiment_score = self.enc_sentiment_score.fit(self.sentiment_score_labels)#### 스코어 두개 추가
        self.enc_aspect_score = self.enc_aspect_score.fit(self.aspect_score_labels)#### 스코어 두개 추가
        
        return {"enc_aspect": self.enc_aspect, "enc_aspect2": self.enc_aspect2, "enc_sentiment": self.enc_sentiment
                ,"enc_sentiment_score":self.enc_sentiment_score, "enc_aspect_score":self.enc_aspect_score}#### 스코어 두개 추가


class ABSADataset(IterableDataset):
    #### 스코어 두개 추가
    def __init__(self, config, fp, enc_aspect, enc_aspect2, enc_sentiment, enc_aspect_score, enc_sentiment_score, batch_size, data_len=0, extension="csv"):
        self.data_len = data_len
        self.file_list = get_file_list(fp, extension)
        self.batch_size = batch_size
        self.max_len = config.max_length
        self.config = config

        # Encoder Setting
        self.enc_aspect = enc_aspect
        self.enc_aspect2 = enc_aspect2
        self.enc_sentiment = enc_sentiment
        self.enc_sentiment_score=enc_sentiment_score #### 스코어 두개 추가
        self.enc_aspect_score=enc_aspect_score #### 스코어 두개 추가
        

        # for embedding
        self.tokenizer = transformers.BertTokenizer.from_pretrained(config.init_model_path, do_lower_case=False)
        self.CLS_IDS = self.tokenizer.encode('[CLS]', add_special_tokens=False)  # [2]
        self.PAD_IDS = self.tokenizer.encode('[PAD]', add_special_tokens=False)  # [0]
        self.SEP_IDS = self.tokenizer.encode('[SEP]', add_special_tokens=False)  # [3]
        self.PADDING_TAG_IDS = [0]
        self.s_len = 0

    def __iter__(self):
        # read data
        for now_fp in self.file_list:
            df = read_csv(now_fp)
            df.loc[:, "Review #"] = df["Review #"].fillna(method="ffill")
            df["Aspect2"] = df["Aspect"]
            df = df.replace({"Aspect2": label_changing_rule})

            df.loc[:, "Aspect"] = self.enc_aspect.transform(df[["Aspect"]])
            df.loc[:, "Aspect2"] = self.enc_aspect2.transform(df[["Aspect2"]])
            df.loc[:, "Sentiment"] = self.enc_sentiment.transform(df[["Sentiment"]])
            df.loc[:, "Sentiment_Score"] = self.enc_sentiment_score.transform(df[["Sentiment_Score"]])#### 스코어 두개 추가
            df.loc[:, "Aspect_Score"] = self.enc_aspect_score.transform(df[["Aspect_Score"]])#### 스코어 두개 추가
            

            sentences = df.groupby("Review #")["Word"].apply(list).values
            aspects = df.groupby("Review #")["Aspect"].apply(list).values
            aspects2 = df.groupby("Review #")["Aspect2"].apply(list).values
            sentiments = df.groupby("Review #")["Sentiment"].apply(list).values 
            sentiment_scores=df.groupby("Review #")["Sentiment_Score"].apply(list).values #### 스코어 두개 추가
            aspect_scores=df.groupby("Review #")["Aspect_Score"].apply(list).values #### 스코어 두개 추가

            for i in range(len(sentences)):
                self.s_len += 1
                yield self.parsing_data(sentences[i], aspects[i], aspects2[i], sentiments[i], sentiment_scores[i], aspect_scores[i]) #### 스코어 두개 추가

    def __len__(self):
        if self.data_len == 0:
            self.data_len = self.get_length()
        return self.data_len

    # data length를 계산
    def get_length(self):
        if self.data_len > 0:
            return self.data_len
        else:
            for now_fp in self.file_list:
                df = read_csv(now_fp)
                sentences = df.groupby("Review #")["Word"].apply(list).values
                self.data_len += len(sentences)
            self.data_len = math.ceil(self.data_len / self.batch_size)
            return self.data_len
    
    def parsing_data(self, text, aspect, aspect2, sentiment, sentiment_score, aspect_score):
        ids = []
        target_aspect = [] # target Aspect Category tensor ids 저장 리스트
        target_aspect2 = []  # target 대분류 Aspect Category tensor ids 저장 리스트 (대분류 기준 성능 측정을 위함)
        target_sentiment = [] # target Sentiment tensor ids 저장 리스트
        target_sentiment_score=[] #### 스코어 두개 추가
        target_aspect_score=[] #### 스코어 두개 추가
        
        for i, s in enumerate(text):
            inputs = self.tokenizer.encode(s, add_special_tokens=False)
            input_len = len(inputs)
            ids.extend(inputs)
            target_aspect.extend([aspect[i]] * input_len)
            target_aspect2.extend([aspect2[i]] * input_len)
            target_sentiment.extend([sentiment[i]] * input_len)
            target_sentiment_score.extend([sentiment_score[i]] * input_len) #### 스코어 두개 추가
            target_aspect_score.extend([aspect_score[i]] * input_len) #### 스코어 두개 추가
            

        # BERT가 처리할 수 있는 길이 (max_length)에 맞추어 slicing
        ids = ids[:self.max_len - 2]
        target_aspect = target_aspect[:self.max_len - 2]
        target_aspect2 = target_aspect2[:self.max_len - 2]
        target_sentiment = target_sentiment[:self.max_len - 2]
        target_sentiment_score = target_sentiment_score[:self.max_len - 2] #### 스코어 두개 추가
        target_aspect_score = target_aspect_score[:self.max_len - 2] #### 스코어 두개 추가
        

        # SPECIAL TOKEN 추가 및 PADDING 수행
        ids = self.CLS_IDS + ids + self.SEP_IDS
        target_aspect = self.PADDING_TAG_IDS + target_aspect + self.PADDING_TAG_IDS  # CLS, SEP 태그 0
        target_aspect2 = self.PADDING_TAG_IDS + target_aspect2 + self.PADDING_TAG_IDS
        target_sentiment = self.PADDING_TAG_IDS + target_sentiment + self.PADDING_TAG_IDS
        target_sentiment_score = self.PADDING_TAG_IDS + target_sentiment_score + self.PADDING_TAG_IDS#### 스코어 두개 추가
        target_aspect_score = self.PADDING_TAG_IDS + target_aspect_score + self.PADDING_TAG_IDS#### 스코어 두개 추가
        
        

        mask = [1] * len(ids)
        token_type_ids = self.PAD_IDS * len(ids)
        padding_len = self.max_len - len(ids)
        ids = ids + (self.PAD_IDS * padding_len)
        mask = mask + ([0] * padding_len)

        token_type_ids = token_type_ids + (self.PAD_IDS * padding_len)
        target_aspect = target_aspect + (self.PADDING_TAG_IDS * padding_len)
        target_aspect2 = target_aspect2 + (self.PADDING_TAG_IDS * padding_len)
        target_sentiment = target_sentiment + (self.PADDING_TAG_IDS * padding_len)#### 스코어 두개 추가
        target_sentiment_score = target_sentiment_score + (self.PADDING_TAG_IDS * padding_len)#### 스코어 두개 추가
        target_aspect_score = target_aspect_score + (self.PADDING_TAG_IDS * padding_len)
        

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_aspect": torch.tensor(target_aspect, dtype=torch.long),
            "target_aspect2": torch.tensor(target_aspect2, dtype=torch.long),
            "target_sentiment": torch.tensor(target_sentiment, dtype=torch.long),#### 스코어 두개 추가
            "target_sentiment_score": torch.tensor(target_sentiment_score, dtype=torch.long),#### 스코어 두개 추가
            "target_aspect_score": torch.tensor(target_aspect_score, dtype=torch.long)
            }