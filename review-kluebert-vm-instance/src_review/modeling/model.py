import torch.nn as nn
from torchcrf import CRF
import transformers

class ABSAModel(nn.Module):
    def __init__(self, config, num_sentiment, num_aspect, num_aspect2, num_sentiment_score, num_aspect_score, need_birnn=False, rnn_dim=128):
        super(ABSAModel, self).__init__()
        # Sentiment와 Aspect Category의 class 개수
        self.num_sentiment = num_sentiment
        self.num_aspect = num_aspect
        self.num_aspect2 = num_aspect2
        self.num_sentiment_score = num_sentiment_score  # 스코어 추가
        self.num_aspect_score = num_aspect_score  # 스코어 추가
        self.need_birnn = need_birnn

        # 사전 학습된 BERT를 load (최종 모델은 klue-bert 사용)
        self.bert = transformers.BertModel.from_pretrained(config.init_model_path)

        # Dropout layer
        self.sentiment_drop = nn.Dropout(config.sentiment_drop_ratio)
        self.aspect_drop = nn.Dropout(config.aspect_drop_ratio)
        self.aspect2_drop = nn.Dropout(config.aspect_drop_ratio)
        self.sentiment_score_drop = nn.Dropout(config.aspect_drop_ratio)  # 스코어 추가
        self.aspect_score_drop = nn.Dropout(config.aspect_drop_ratio)  # 스코어 추가

        # Sentiment 및 Aspect Category layer 차원 설정
        sentiment_in_feature = config.sentiment_in_feature
        aspect_in_feature = config.aspect_in_feature

        # birnn layer 추가
        if need_birnn:
            self.sentiment_birnn = nn.LSTM(sentiment_in_feature, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            self.aspect_birnn = nn.LSTM(aspect_in_feature, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            self.aspect2_birnn = nn.LSTM(aspect_in_feature, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            self.aspect_score_birnn = nn.LSTM(aspect_in_feature, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            self.sentiment_score_birnn = nn.LSTM(sentiment_in_feature, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            sentiment_in_feature = rnn_dim * 2
            aspect_in_feature = rnn_dim * 2

        # Sentiment와 Aspect Category의 Linear Layer 구성
        self.hidden2sent_tag = nn.Linear(sentiment_in_feature, self.num_sentiment)
        self.hidden2asp_tag = nn.Linear(aspect_in_feature, self.num_aspect)
        self.hidden2asp2_tag = nn.Linear(aspect_in_feature, self.num_aspect2)
        self.hidden2asp_score_tag = nn.Linear(aspect_in_feature, self.num_aspect_score)
        self.hidden2sent_score_tag = nn.Linear(sentiment_in_feature, self.num_sentiment_score)

        # Sentiment와 Aspect Category의 CRF Layer 구성
        self.sent_crf = CRF(self.num_sentiment, batch_first=True)
        self.asp_crf = CRF(self.num_aspect, batch_first=True)
        self.asp2_crf = CRF(self.num_aspect2, batch_first=True)
        self.asp_score_crf = CRF(self.num_aspect_score, batch_first=True)
        self.sent_score_crf = CRF(self.num_sentiment_score, batch_first=True)

    def forward(self, ids, mask=None, token_type_ids=None, target_aspect=None, target_aspect2=None, target_sentiment=None, target_aspect_score=None, target_sentiment_score=None):
        # 사전학습된 bert에 input을 feed
        model_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)[0]

        # BI-RNN layer
        if self.need_birnn:
            sentiment_output, _ = self.sentiment_birnn(model_output)
            aspect_output, _ = self.aspect_birnn(model_output)
            aspect2_output, _ = self.aspect2_birnn(model_output)
            aspect_score_output, _ = self.aspect_score_birnn(model_output)
            sentiment_score_output, _ = self.sentiment_score_birnn(model_output)
        else:
            sentiment_output = model_output
            sentiment_score_output = model_output
            aspect_score_output = model_output
            aspect_output = model_output
            aspect2_output = model_output

            
        # 과적합 방지를 위해 Sentiment와 Aspect Category Dropout 수행
        sentiment_output = self.sentiment_drop(sentiment_output)
        aspect_output = self.aspect_drop(aspect_output)
        aspect2_output = self.aspect2_drop(aspect2_output)
        aspect_score_output = self.aspect_score_drop(aspect_score_output)
        sentiment_score_output = self.sentiment_score_drop(sentiment_score_output)

        # Linear Layer feeding
        sentiment_emmisions = self.hidden2sent_tag(sentiment_output)
        aspect_emmisions = self.hidden2asp_tag(aspect_output)
        aspect2_emmisions = self.hidden2asp2_tag(aspect2_output)
        aspect_score_emmisions = self.hidden2asp_score_tag(aspect_score_output)
        sentiment_score_emmisions = self.hidden2sent_score_tag(sentiment_score_output)

        # CRF Layer Decoding
        sentiment = self.sent_crf.decode(sentiment_emmisions)
        aspect = self.asp_crf.decode(aspect_emmisions)
        aspect2 = self.asp2_crf.decode(aspect2_emmisions)
        aspect_score = self.asp_score_crf.decode(aspect_score_emmisions)
        sentiment_score = self.sent_score_crf.decode(sentiment_score_emmisions)

        # loss 계산
        if target_aspect is not None and target_aspect2 is not None and target_sentiment is not None and target_aspect_score is not None and target_sentiment_score is not None:
            sentiment_loss = -1 * self.sent_crf(sentiment_emmisions, target_sentiment, mask=mask.byte())
            aspect_loss = -1 * self.asp_crf(aspect_emmisions, target_aspect, mask=mask.byte())
            aspect2_loss = -1 * self.asp2_crf(aspect2_emmisions, target_aspect2, mask=mask.byte())
            aspect_score_loss = -1 * self.asp_score_crf(aspect_score_emmisions, target_aspect_score, mask=mask.byte())
            sentiment_score_loss = -1 * self.sent_score_crf(sentiment_score_emmisions, target_sentiment_score, mask=mask.byte())

            return sentiment_loss, aspect_loss, aspect2_loss, aspect_score_loss, sentiment_score_loss, sentiment, aspect, aspect2, sentiment_score, aspect_score
        else:
            return sentiment, aspect, aspect2, sentiment_score, aspect_score
