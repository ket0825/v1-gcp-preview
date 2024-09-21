from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from data_manager.parsers.label_unification.label_map import label_changing_rule
import torch
import time
import datetime
import transformers
import json
import os
from utils.file_io import get_file_list, read_json, read_csv

import re
from collections import Counter
import kss

global unk_pattern
unk_pattern = re.compile(r'\[UNK\]')

# input tensor의 구조 변경을 위한 함수
def parsing_batch(data, device):
    d = {}
    for k in data[0].keys():
        d[k] = list(d[k] for d in data)
    for k in d.keys():
        d[k] = torch.stack(d[k]).to(device)
    return d

# 모델 학습
def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    sentiment_loss_total = 0
    aspect_loss_total = 0
    aspect2_loss_total = 0
    sentiment_score_loss_total = 0
    aspect_score_loss_total = 0
    loader_len = len(data_loader)
    for data in tqdm(data_loader, total=loader_len):
        data = parsing_batch(data, device)
        optimizer.zero_grad()  # backward를 위한 gradients 초기화
        sentiment_loss, aspect_loss, aspect2_loss, aspect_score_loss, sentiment_score_loss, sentiment, aspect, aspect2, sentiment_score, aspect_score = model(**data)

        # 각각의 손실에 대해 역전파
        sentiment_loss.backward(retain_graph=True)
        aspect_loss.backward(retain_graph=True)
        aspect2_loss.backward(retain_graph=True)
        aspect_score_loss.backward(retain_graph=True)
        sentiment_score_loss.backward()

        optimizer.step()
        scheduler.step()

        final_loss += (sentiment_loss.item() + aspect_loss.item() + aspect2_loss.item() + aspect_score_loss.item() + sentiment_score_loss.item()) / 5
        sentiment_loss_total += sentiment_loss.item()
        aspect_loss_total += aspect_loss.item()
        aspect2_loss_total += aspect2_loss.item()
        sentiment_score_loss_total += sentiment_score_loss.item()
        aspect_score_loss_total += aspect_score_loss.item()
    return final_loss / loader_len, sentiment_loss_total / loader_len, aspect_loss_total / loader_len, aspect2_loss_total / loader_len, sentiment_score_loss_total / loader_len, aspect_score_loss_total / loader_len

def eval_fn(data_loader, model, enc_sentiment, enc_aspect, enc_aspect2, enc_sentiment_score, enc_aspect_score, device, log, f1_mode='micro', flag='valid'):
    print("eval_start")
    model.eval()
    final_loss = 0
    sentiment_loss_total = 0
    aspect_loss_total = 0
    aspect2_loss_total = 0
    sentiment_score_loss_total = 0
    aspect_score_loss_total = 0
    nb_eval_steps = 0
    # 성능 측정 변수 선언
    sentiment_accuracy, aspect_accuracy, aspect2_accuracy, sentiment_score_accuracy, aspect_score_accuracy = 0, 0, 0, 0, 0
    sentiment_f1score, aspect_f1score, aspect2_f1score, sentiment_score_f1score, aspect_score_f1score = 0, 0, 0, 0, 0
    sentiment_preds, sentiment_labels = [], []
    aspect_preds, aspect_labels = [], []
    aspect2_preds, aspect2_labels = [], []
    sentiment_score_preds, sentiment_score_labels = [], []
    aspect_score_preds, aspect_score_labels = [], []

    loader_len = len(data_loader)

    eval_start_time = time.time()  # evaluation을 시작한 시간을 저장 (소요 시간 측정을 위함)
    for data in tqdm(data_loader, total=loader_len):
        data = parsing_batch(data, device)
        sentiment_loss, aspect_loss, aspect2_loss, aspect_score_loss, sentiment_score_loss, predict_sentiment, predict_aspect, predict_aspect2, predict_sentiment_score, predict_aspect_score = model(**data)

        sentiment_label = data['target_sentiment'].cpu().numpy().reshape(-1)
        aspect_label = data['target_aspect'].cpu().numpy().reshape(-1)
        aspect2_label = data['target_aspect2'].cpu().numpy().reshape(-1)
        sentiment_score_label = data['target_sentiment_score'].cpu().numpy().reshape(-1)
        aspect_score_label = data['target_aspect_score'].cpu().numpy().reshape(-1)

        sentiment_pred = np.array(predict_sentiment).reshape(-1)
        aspect_pred = np.array(predict_aspect).reshape(-1)
        aspect2_pred = np.array(predict_aspect2).reshape(-1)
        sentiment_score_pred = np.array(predict_sentiment_score).reshape(-1)
        aspect_score_pred = np.array(predict_aspect_score).reshape(-1)

        # remove padding indices
        pad_label_indices = np.where(sentiment_label == 0)  # pad 레이블
        sentiment_label = np.delete(sentiment_label, pad_label_indices)
        sentiment_pred = np.delete(sentiment_pred, pad_label_indices)

        pad_label_indices = np.where(aspect_label == 0)  # pad 레이블
        aspect_label = np.delete(aspect_label, pad_label_indices)
        aspect_pred = np.delete(aspect_pred, pad_label_indices)

        pad_label_indices = np.where(aspect2_label == 0)  # pad 레이블
        aspect2_label = np.delete(aspect2_label, pad_label_indices)
        aspect2_pred = np.delete(aspect2_pred, pad_label_indices)

        pad_label_indices = np.where(sentiment_score_label == 0)  # pad 레이블
        sentiment_score_label = np.delete(sentiment_score_label, pad_label_indices)
        sentiment_score_pred = np.delete(sentiment_score_pred, pad_label_indices)

        pad_label_indices = np.where(aspect_score_label == 0)  # pad 레이블
        aspect_score_label = np.delete(aspect_score_label, pad_label_indices)
        aspect_score_pred = np.delete(aspect_score_pred, pad_label_indices)

        # Accuracy 및 F1-score 계산
        sentiment_accuracy += accuracy_score(sentiment_label, sentiment_pred)
        aspect_accuracy += accuracy_score(aspect_label, aspect_pred)
        aspect2_accuracy += accuracy_score(aspect2_label, aspect2_pred)
        sentiment_score_accuracy += accuracy_score(sentiment_score_label, sentiment_score_pred)
        aspect_score_accuracy += accuracy_score(aspect_score_label, aspect_score_pred)

        sentiment_f1score += f1_score(sentiment_label, sentiment_pred, average=f1_mode)
        aspect_f1score += f1_score(aspect_label, aspect_pred, average=f1_mode)
        aspect2_f1score += f1_score(aspect2_label, aspect2_pred, average=f1_mode)
        sentiment_score_f1score += f1_score(sentiment_score_label, sentiment_score_pred, average=f1_mode)
        aspect_score_f1score += f1_score(aspect_score_label, aspect_score_pred, average=f1_mode)

        # target label과 모델의 예측 결과를 저장 => classification report 계산 위함
        sentiment_labels.extend(sentiment_label)
        sentiment_preds.extend(sentiment_pred)
        aspect_labels.extend(aspect_label)
        aspect_preds.extend(aspect_pred)
        aspect2_labels.extend(aspect2_label)
        aspect2_preds.extend(aspect2_pred)
        sentiment_score_labels.extend(sentiment_score_label)
        sentiment_score_preds.extend(sentiment_score_pred)
        aspect_score_labels.extend(aspect_score_label)
        aspect_score_preds.extend(aspect_score_pred)

        final_loss += (sentiment_loss.item() + aspect_loss.item() + aspect2_loss.item() + aspect_score_loss.item() + sentiment_score_loss.item()) / 5
        sentiment_loss_total += sentiment_loss.item()
        aspect_loss_total += aspect_loss.item()
        aspect2_loss_total += aspect2_loss.item()
        sentiment_score_loss_total += sentiment_score_loss.item()
        aspect_score_loss_total += aspect_score_loss.item()
        nb_eval_steps += 1

    # encoding 된 Sentiment와 Aspect Category를 Decoding (원 형태로 복원)
    sentiment_pred_names = enc_sentiment.inverse_transform(sentiment_preds)
    sentiment_label_names = enc_sentiment.inverse_transform(sentiment_labels)
    aspect_pred_names = enc_aspect.inverse_transform(aspect_preds)
    aspect_label_names = enc_aspect.inverse_transform(aspect_labels)
    aspect2_pred_names = enc_aspect2.inverse_transform(aspect2_preds)
    aspect2_label_names = enc_aspect2.inverse_transform(aspect2_labels)
    sentiment_score_pred_names = enc_sentiment_score.inverse_transform(sentiment_score_preds)
    sentiment_score_label_names = enc_sentiment_score.inverse_transform(sentiment_score_labels)
    aspect_score_pred_names = enc_aspect_score.inverse_transform(aspect_score_preds)
    aspect_score_label_names = enc_aspect_score.inverse_transform(aspect_score_labels)

    # [Sentiment에 대한 성능 계산]
    sentiment_accuracy = round(sentiment_accuracy / nb_eval_steps, 2)
    sentiment_f1score = round(sentiment_f1score / nb_eval_steps, 2)

    # 각 감정 속성 별 성능 계산
    sentiment_report = classification_report(sentiment_label_names, sentiment_pred_names, digits=4)

    # [Aspect Category에 대한 성능 계산]
    aspect_accuracy = round(aspect_accuracy / nb_eval_steps, 2)
    aspect_f1score = round(aspect_f1score / nb_eval_steps, 2)

    # 각 aspect category에 대한 성능을 계산 (대분류 속성 기준)
    aspect_report = classification_report(aspect_label_names, aspect_pred_names, digits=4)

    # [Aspect2 Category에 대한 성능 계산]
    aspect2_accuracy = round(aspect2_accuracy / nb_eval_steps, 2)
    aspect2_f1score = round(aspect2_f1score / nb_eval_steps, 2)

    # 각 aspect2 category에 대한 성능을 계산
    aspect2_report = classification_report(aspect2_label_names, aspect2_pred_names, digits=4)

    # [Sentiment Score에 대한 성능 계산]
    sentiment_score_accuracy = round(sentiment_score_accuracy / nb_eval_steps, 2)
    sentiment_score_f1score = round(sentiment_score_f1score / nb_eval_steps, 2)

    # 각 sentiment score에 대한 성능을 계산
    sentiment_score_report = classification_report(sentiment_score_label_names, sentiment_score_pred_names, digits=4)

    # [Aspect Score에 대한 성능 계산]
    aspect_score_accuracy = round(aspect_score_accuracy / nb_eval_steps, 2)
    aspect_score_f1score = round(aspect_score_f1score / nb_eval_steps, 2)

    # 각 aspect score에 대한 성능을 계산
    aspect_score_report = classification_report(aspect_score_label_names, aspect_score_pred_names, digits=4)

    eval_loss = final_loss / loader_len  # model의 loss
    eval_end_time = time.time() - eval_start_time  # 모든 데이터에 대한 평가 소요 시간
    eval_sample_per_sec = str(datetime.timedelta(seconds=(eval_end_time / loader_len)))  # 한 샘플에 대한 평가 소요 시간
    eval_times = str(datetime.timedelta(seconds=eval_end_time))  # 시:분:초 형식으로 변환

    # validation 과정일 때는, 각 sample에 대한 개별 결과값은 출력하지 않음
    if flag == 'eval':
        for i in range(len(aspect_label_names)):
            if aspect_label_names[i] != aspect_pred_names[i]:
                asp_result = "X"
            else:
                asp_result = "O"
            if sentiment_label_names[i] != sentiment_pred_names[i]:
                pol_result = "X"
            else:
                pol_result = "O"

            log.info(f"[{i} >> Sentiment : {pol_result} | Aspect : {asp_result}] "
                     f"predicted sentiment label: {sentiment_pred_names[i]}, gold sentiment label: {sentiment_label_names[i]} | "
                     f"predicted aspect label: {aspect_pred_names[i]}, gold aspect label: {aspect_label_names[i]} | ")

    log.info("*****" + "eval metrics" + "*****")
    log.info(f"eval_loss: {eval_loss}")
    log.info(f"eval_runtime: {eval_times}")
    log.info(f"eval_samples: {loader_len}")
    log.info(f"eval_samples_per_second: {eval_sample_per_sec}")
    log.info(f"Sentiment Accuracy: {sentiment_accuracy}")
    log.info(f"Sentiment f1score {f1_mode} : {sentiment_f1score}")
    log.info(f"Aspect Accuracy: {aspect_accuracy}")
    log.info(f"Aspect f1score {f1_mode} : {aspect_f1score}")
    log.info(f"Aspect2 Accuracy: {aspect2_accuracy}")
    log.info(f"Aspect2 f1score {f1_mode} : {aspect2_f1score}")
    log.info(f"Sentiment Score Accuracy: {sentiment_score_accuracy}")
    log.info(f"Sentiment Score f1score {f1_mode} : {sentiment_score_f1score}")
    log.info(f"Aspect Score Accuracy: {aspect_score_accuracy}")
    log.info(f"Aspect Score f1score {f1_mode} : {aspect_score_f1score}")
    log.info(f"Sentiment Accuracy Report:")
    log.info(sentiment_report)
    log.info(f"Aspect Accuracy Report:")
    log.info(aspect_report)
    log.info(f"Aspect2 Accuracy Report:")
    log.info(aspect2_report)
    log.info(f"Sentiment Score Accuracy Report:")
    log.info(sentiment_score_report)
    log.info(f"Aspect Score Accuracy Report:")
    log.info(aspect_score_report)

    return eval_loss, sentiment_loss_total / nb_eval_steps, aspect_loss_total / nb_eval_steps, aspect2_loss_total / nb_eval_steps, sentiment_score_loss_total / nb_eval_steps, aspect_score_loss_total / nb_eval_steps





# ##### 전체적으로 수정 필요
# def tag_fn(config, tokenizer, data_loader, model, enc_sentiment, enc_aspect, enc_aspect2, enc_sentiment_score, enc_aspect_score, device, log):
#     print("tagging_start")
#     model.eval()
#     nb_eval_steps = 0

#     # 예측값 변수 선언
#     sentiment_preds = []
#     aspect_preds = []
#     aspect2_preds = []
#     sentiment_score_preds = []
#     aspect_score_preds = []
#     ids_inputs = []



#     # loader_len = len(data_loader)
#     ##### 배치 사이즈 4로 했을 때 전체 사이즈가 1700대인데 loader_len이 110대가 나옴
#     loader_len = data_loader.dataset.get_length()

#     tagging_start_time = time.time()  # evaluation을 시작한 시간을 저장 (소요 시간 측정을 위함)
#     for data in tqdm(data_loader, total=loader_len):
#         data = parsing_batch(data, device)
#         predict_sentiment, predict_aspect, predict_aspect2, predict_sentiment_score, predict_aspect_score = model(**data)

#         sentiment_pred = np.array(predict_sentiment).reshape(-1)
#         aspect_pred = np.array(predict_aspect).reshape(-1)
#         aspect2_pred = np.array(predict_aspect2).reshape(-1)
#         sentiment_score_pred = np.array(predict_sentiment_score).reshape(-1)
#         aspect_score_pred = np.array(predict_aspect_score).reshape(-1)
        
#         ids_input = data['ids'].numpy().reshape(-1)

#         # remove padding indices
        

#         # 모델의 예측 결과를 저장
        
#         sentiment_preds.extend(sentiment_pred)
#         aspect_preds.extend(aspect_pred)
#         aspect2_preds.extend(aspect2_pred)
#         sentiment_score_preds.extend(sentiment_score_pred)
#         aspect_score_preds.extend(aspect_score_pred)

#         ids_inputs.extend(ids_input)

        

#     # encoding 된 Sentiment와 Aspect Category를 Decoding (원 형태로 복원)
#     sentiment_pred_names = enc_sentiment.inverse_transform(sentiment_preds)
#     aspect_pred_names = enc_aspect.inverse_transform(aspect_preds)
#     aspect2_pred_names = enc_aspect2.inverse_transform(aspect2_preds)
#     sentiment_score_pred_names = enc_sentiment_score.inverse_transform(sentiment_score_preds)
#     aspect_score_pred_names = enc_aspect_score.inverse_transform(aspect_score_preds)

    
#     ids_input_names = tokenizer.decode(ids_inputs)



#     tagging_end_time = time.time() - tagging_start_time  # 모든 데이터에 대한 평가 소요 시간
#     tagging_sample_per_sec = str(datetime.timedelta(seconds=(tagging_end_time / loader_len)))  # 한 샘플에 대한 평가 소요 시간
#     tagging_times = str(datetime.timedelta(seconds=tagging_end_time))  # 시:분:초 형식으로 변환

 

#     return ids_input_names, sentiment_pred_names, aspect_pred_names, aspect2_pred_names, sentiment_score_pred_names, aspect_score_pred_names




def parsing_data(tokenizer, text, words_in_sent): # text는 리뷰 하나임
    ids = []
    words_list = [] # 단어 단위로 묶어서
    max_len = 512
    CLS_IDS = tokenizer.encode('[CLS]', add_special_tokens=False)  # [2]
    PAD_IDS = tokenizer.encode('[PAD]', add_special_tokens=False)  # [0]
    SEP_IDS = tokenizer.encode('[SEP]', add_special_tokens=False)  # [3]
    PADDING_TAG_IDS = [0]
    
    for i, s in enumerate(text):
        inputs = tokenizer.encode(s, add_special_tokens=False)
        input_len = len(inputs) 
        ids.extend(inputs)
        words_list.append(inputs) # 단어 단위로 저장

    
    # BERT가 처리할 수 있는 길이 (max_length)에 맞추어 slicing
    ids = ids[:max_len - 2]

    # words_list도 ids의 요소와 일치하게 slicing
    flattened_ids_list = [item for sublist in words_list for item in sublist]
    sliced_flattened_ids_list = flattened_ids_list[:max_len - 2]

    # sliced_flattened_ids_list를 다시 원래 구조로 되돌리기
    sliced_words_list = []
    current_length = 0

    # for sublist in words_list:
    #     if current_length + len(sublist) > max_len - 2:
    #         if (max_len - 2 - current_length == 0):
    #             pass
    #         else:
    #             sliced_words_list.append(sublist[:max_len - 2 - current_length])
    #         break
    #     else:
    #         sliced_words_list.append(sublist)
    #         current_length += len(sublist)

    updated_words_in_sent = []  # 업데이트된 words_in_sent 리스트
    last_idx = len(words_list) # 포함된 단어의 총 개수
    for i, sublist in enumerate(words_list):
        if current_length + len(sublist) > max_len - 2:
            if (max_len - 2 - current_length == 0): # 해당 단어는 토큰이 하나도 포함이 안되는거지
                last_idx = i
                pass
            else:
                sliced_words_list.append(sublist[:max_len - 2 - current_length])
                last_idx = (i+1)
                # updated_words_in_sent.append(min(words_in_sent[i], max_len - 2 - current_length))
            break
        else:
            sliced_words_list.append(sublist)
            # updated_words_in_sent.append(words_in_sent[i])
            current_length += len(sublist)
    
    sum = 0
    for i in range(len(words_in_sent)):
        sum += words_in_sent[i]
        if (last_idx <= sum):
            updated_words_in_sent.append(last_idx - sum + words_in_sent[i])
        else:
            updated_words_in_sent.append(words_in_sent[i])
    

    # SPECIAL TOKEN 추가 및 PADDING 수행
    ids = CLS_IDS + ids + SEP_IDS
    
    

    mask = [1] * len(ids)
    token_type_ids = PAD_IDS * len(ids)
    padding_len = max_len - len(ids)
    ids = ids + (PAD_IDS * padding_len)
    mask = mask + ([0] * padding_len)

    token_type_ids = token_type_ids + (PAD_IDS * padding_len)
    
    ids = [ids]
    mask = [mask]
    token_type_ids = [token_type_ids]
    

    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long)
        }, sliced_words_list, updated_words_in_sent



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

whitespace_pattern = re.compile(r'\s+') # 빈칸 여러개 무조건 1개로 고정시키기 위한 pattern

special_char_pattern = re.compile(r'\s+([~!%&-,.:?…])') # 특수문자 띄어쓰기 문제 해결하기 위한 코드(인코딩 후 디코딩 과정에서 띄어쓰기 추가되는 듯)

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


def replace_newline(text):
    return text.replace('\n', ' ')

def preprocess_content(content):
    # sentences = kss.split_sentences(content)
    # # sentences = [[sent] for sent in sentences]
    # sentences = regexp(sentences)
    # sentences_period_added = [replace_newline(sent.strip()) + '.' for sent in sentences if sent.strip()]
    # preprocessed_content = ' '.join(sentences_period_added)
    sentences = kss.split_sentences(content)
    sentences = [[sent] for sent in sentences]
    sentences_regexp = [regexp(sent_list) for sent_list in sentences]
    sentences_period_added = [replace_newline(sent[0].strip()) + '.' for sent in sentences_regexp if sent[0].strip()]
    sentences_period_added = [[sent] for sent in sentences_period_added]
    no_words_in_sentence_list = [] # sentence 단위를 확인하기 위해 sentence 별 단어의 개수 확인
    for row in sentences_period_added:
        for element in row:
            no_words_in_sentence_list.append(len(element.split()))
    preprocessed_content = ' '.join([''.join(row) for row in sentences_period_added])
    return preprocessed_content, no_words_in_sentence_list

def normalize_whitespace(content):
    # 정규 표현식을 사용하여 연속된 공백을 단일 공백으로 변환
    normalized_content = whitespace_pattern.sub(' ', content)
    return normalized_content


def remove_bio_prefix_for_list(tag_list): # BI 태그 제거
    for i in range(len(tag_list)):
        if tag_list[i].startswith('B-') or tag_list[i].startswith('I-'):
            tag_list[i] = tag_list[i][2:]
    return tag_list

def remove_bio_prefix(tag): # BI 태그 제거
    if tag.startswith('B-') or tag.startswith('I-'):
        tag = tag[2:]
    return tag

def majority_vote_for_valid_eval(labels):
    return Counter(labels).most_common(1)[0][0]

    # # 다수결을 위해 Counter 사용
    # sentiment_counts = Counter(group['Sentiment'])
    # aspect_counts = Counter(group['Aspect'])
    # sentiment_score_counts = Counter(group['Sentiment_Score'])
    # aspect_score_counts = Counter(group['Aspect_Score'])
    
    # # 가장 많이 나온 값을 찾아 반환
    # majority_sentiment = sentiment_counts.most_common(1)[0][0]
    # majority_aspect = aspect_counts.most_common(1)[0][0]
    # majority_sentiment_score = sentiment_score_counts.most_common(1)[0][0]
    # majority_aspect_score = aspect_score_counts.most_common(1)[0][0]
    
    # return pd.Series({
    #     'sentiment_majority': majority_sentiment,
    #     'aspect_majority': majority_aspect,
    #     'sentiment_score_majority': majority_sentiment_score,
    #     'aspect_score_majority': majority_aspect_score
    # })



def majority_vote(lists):
    all_elements = [element for sublist in lists for element in sublist]
    count = Counter(all_elements)
    return count.most_common(1)[0][0]
    

def create_ourt_topics_dict(row): # 각 문장 별로 dict 만드는 함수
    if (row['aspect'] == 'O' or row['aspect'] == '[PAD]'):
        return None
    else:
        if (row['sentiment'] == 'O' or row['sentiment'] == '[PAD]'):
            return None
        else: # 여기서 score 조정해도 되고
            return {
            "text": row['sentence'],
            "topic": row['aspect'],
            "start_pos": 0,  # Start position placeholder
            "end_pos": 0,  # End position placeholder
            "positive_yn": 'Y' if row['sentiment'] == '긍정' else 'N',
            "sentiment_scale": 0 if row['sentiment_score'] == 'O' or row['sentiment_score'] == '[PAD]' else int(row['sentiment_score']),
            "topic_score": 0 if row['aspect_score'] == 'O' or row['aspect_score'] == '[PAD]' else int(row['aspect_score'])
            }


def replace_all_unk_to_original(content, text):
    # [UNK]를 찾아서 위치와 해당 단어를 저장할 리스트 초기화
    content_spacingx = content.replace(' ','')
    text_spacingx = text.replace(' ','')
    
    unk_positions = []    
    
    global unk_pattern
    # [UNK]의 위치를 모두 찾기
    for match in unk_pattern.finditer(text_spacingx):
        unk_positions.append((match.start(), match.end()))

    list_to_slicing = [0] # 매칭 구간 나누기 위한 리스트

    for start, end in unk_positions: 
        list_to_slicing.append(start)
        list_to_slicing.append(end)
        
    if (list_to_slicing[-1] != len(text_spacingx)):
        list_to_slicing.append(len(text_spacingx))

    str_to_compile = ''

    if (len(list_to_slicing) % 2 == 0):
        for i in range(len(list_to_slicing) // 2 - 1):
            before_unk = text_spacingx[list_to_slicing[2*i]:list_to_slicing[2*i + 1]]
            before_match = re.escape(before_unk)
            str_to_compile += before_match
            str_to_compile += '(.*?)'
        after_unk = text_spacingx[list_to_slicing[-2]: list_to_slicing[-1]]
        after_match = re.escape(after_unk)
        str_to_compile += after_match
    else:
        for i in range(len(list_to_slicing) // 2):
            before_unk = text_spacingx[list_to_slicing[2*i]:list_to_slicing[2*i + 1]]
            before_match = re.escape(before_unk)
            str_to_compile += before_match
            str_to_compile += '(.*?)'

    pattern = re.compile(str_to_compile)
    match = pattern.search(content_spacingx)

    for i in range(len(unk_positions)):        
        text = unk_pattern.sub(match.group(i+1), text, count=1)
    
    return text


def remove_space_before_special_char(text):
    # 정규 표현식을 컴파일하여 패턴 생성
    
    # 컴파일된 패턴을 사용하여 특수 문자 앞의 공백을 제거
    normalized_text = special_char_pattern.sub(r'\1', text)
    
    return normalized_text





#### 전체적으로 수정 필요
def tag_fn(config, tokenizer, model, enc_sentiment, enc_aspect, enc_aspect2, enc_sentiment_score, enc_aspect_score, device, log):
    print("tagging_start")
    model.eval()


    file_list = get_file_list(config.tagging_fp, 'json')

    tagging_start_time = time.time()  # 태깅을 시작한 시간을 저장 (소요 시간 측정을 위함)

    for file in file_list:
        if os.path.exists(file.replace("data_untagged_json","tagged_results_json").replace('.json', '_역변환.json')):
            print(file.replace("data_untagged_json","tagged_results_json").replace('.json', '_역변환.json')+" exists --> continue")
            continue

        # 예측값 변수 선언 --> 파일 별로 선언해줘야 할듯
        sentiment_preds = []
        aspect_preds = []
        aspect2_preds = []
        sentiment_score_preds = []
        aspect_score_preds = []

        ids_inputs = []
        words_list_for_file = [] # 각 리뷰에서 단어 별로 인코딩된 값 저장한 2차원 리스트 (리뷰, 단어)
        words_in_sent_for_file = [] # 하나의 파일에 대해서 각 리뷰의 문장 별 단어 개수

        df = read_json(file)
        df.loc[:, "content"] = df["content"].fillna(method="ffill")
        df["temp"] = df['content'].apply(preprocess_content)

        df['content'] = df['temp'].apply(lambda x: x[0])
        df['# of words in each sentence'] = df['temp'].apply(lambda x: x[1])

        df = df[df['content'] != ''] # 빈 텍스트 삭제(의미 없으니까)
        df = df.reset_index(drop=True)
        
        df['content'] = df['content'].apply(normalize_whitespace) # spacing 문제 해결 위해서 공백은 무조건 1칸으로 고정
        # df['# of words in each sentence'] = df['temp'].apply(lambda x: x[1])

        # sentences = df.groupby("Review #")["Word"].apply(list).values
        sentences = [text.split() for text in df["content"]]
        # blank_count = 0
        # for i in range(len(sentences)):
        #     for j in range(len(sentences[i])):
        #         if(sentences[i][j].replace(' ','') == ''):
        #             blank_count += 1

        sentences = np.array(sentences)
        words_in_each_sentence = df["# of words in each sentence"].tolist() # 한 리뷰에 대해서 각 문장이 가지는 단어의 개수를 모은 2차원 리스트

        print("Tagging "+ file)
        for i in tqdm(range(len(sentences))):
            data, words_list, words_in_sent= parsing_data(tokenizer, sentences[i], words_in_each_sentence[i]) 
            # ids_list는 단어 단위로 묶은 것-->ids_list의 len이 단어 개수임 / words_in_sent는 리뷰 하나에 대한 문장이 가지는 단어의 개수(slicing)
            words_list_for_file.append(words_list)
            words_in_sent_for_file.append(words_in_sent)
            predict_sentiment, predict_aspect, predict_aspect2, predict_sentiment_score, predict_aspect_score = model(**data)

            sentiment_pred = np.array(predict_sentiment).reshape(-1)
            aspect_pred = np.array(predict_aspect).reshape(-1)
            aspect2_pred = np.array(predict_aspect2).reshape(-1)
            sentiment_score_pred = np.array(predict_sentiment_score).reshape(-1)
            aspect_score_pred = np.array(predict_aspect_score).reshape(-1)
            
            ids_input = data['ids'].numpy().reshape(-1)


    #     # remove padding indices
            indices_to_remove = np.where((ids_input == 2) | (ids_input == 3) | (ids_input == 0))
            sentiment_pred = np.delete(sentiment_pred, indices_to_remove)
            aspect_pred = np.delete(aspect_pred, indices_to_remove)
            aspect2_pred = np.delete(aspect2_pred, indices_to_remove)
            sentiment_score_pred = np.delete(sentiment_score_pred, indices_to_remove)
            aspect_score_pred = np.delete(aspect_score_pred, indices_to_remove)
            ids_input = np.delete(ids_input, indices_to_remove)



        # 모델의 예측 결과를 저장
        
            sentiment_preds.extend(sentiment_pred)
            aspect_preds.extend(aspect_pred)
            aspect2_preds.extend(aspect2_pred)
            sentiment_score_preds.extend(sentiment_score_pred)
            aspect_score_preds.extend(aspect_score_pred)

            ids_inputs.extend(ids_input)

        

        # encoding 된 Sentiment와 Aspect Category를 Decoding (원 형태로 복원)
        sentiment_pred_names = enc_sentiment.inverse_transform(sentiment_preds)
        aspect_pred_names = enc_aspect.inverse_transform(aspect_preds)
        aspect2_pred_names = enc_aspect2.inverse_transform(aspect2_preds)
        sentiment_score_pred_names = enc_sentiment_score.inverse_transform(sentiment_score_preds)
        aspect_score_pred_names = enc_aspect_score.inverse_transform(aspect_score_preds)

    
        ids_input_names = tokenizer.decode(ids_inputs)

        words_list_names = []

        final_sentiment_pred_names = []
        final_aspect_pred_names = []
        final_aspect2_pred_names = []
        final_sentiemnt_score_pred_names = []
        final_aspect_score_pred_names = []

        start_idx = 0
        end_idx = 0
        

        for i in range(len(words_list_for_file)): # 리뷰 차원 늘려라 --> 해결
            # if (len(ids_list_for_file[i]) != 0): # slicing 하는 과정에서 길이가 길어서 짤리면 []가 들어가는 경우가 있어서 처리
            words_list_names_for_content = []
            final_sentiment_pred_names_for_content = []
            final_aspect_pred_names_for_content = []
            final_aspect2_pred_names_for_content = []
            final_sentiemnt_score_pred_names_for_content = []
            final_aspect_score_pred_names_for_content = []
            for j in range(len(words_list_for_file[i])):
                end_idx += len(words_list_for_file[i][j])
                words_list_names_for_content.append(tokenizer.decode(words_list_for_file[i][j]))
                final_sentiment_pred_names_for_content.append(sentiment_pred_names[start_idx:end_idx])
                final_aspect_pred_names_for_content.append(aspect_pred_names[start_idx:end_idx])
                final_aspect2_pred_names_for_content.append(aspect2_pred_names[start_idx:end_idx])
                final_sentiemnt_score_pred_names_for_content.append(sentiment_score_pred_names[start_idx:end_idx])
                final_aspect_score_pred_names_for_content.append(aspect_score_pred_names[start_idx:end_idx])
                start_idx = end_idx

            words_list_names.append(words_list_names_for_content) # content 별 단어를 모은 2차원 리스트(리뷰, 단어)
            final_sentiment_pred_names.append(final_sentiment_pred_names_for_content) # content 별 단어에 대한 토큰별 예측값 리스트를 모은 2차원 리스트
            final_aspect_pred_names.append(final_aspect_pred_names_for_content)
            final_aspect2_pred_names.append(final_aspect2_pred_names_for_content)
            final_sentiemnt_score_pred_names.append(final_sentiemnt_score_pred_names_for_content)
            final_aspect_score_pred_names.append(final_aspect_score_pred_names_for_content)


        # print("len(words_list_for_file):", len(words_list_for_file)) # 한 파일에 있는 content 개수 의미
        # print("len(words_list_names):", len(words_list_names)) 
        # print("len(ids_inputs):", len(ids_inputs))
        # print("len(sentiment_pred_names):", len(sentiment_pred_names))
        # print("len(aspect_pred_names):", len(aspect_pred_names))
        # print("len(aspect2_pred_names):", len(aspect2_pred_names))
        # print("len(sentiment_score_pred_names):", len(sentiment_score_pred_names))
        # print("len(aspect_score_pred_names):", len(aspect_score_pred_names))

        new_data = []
        sentence_count_list = []
        sentence_counter = 0
        for i in range(len(words_in_sent_for_file)):
            for j in words_in_sent_for_file[i]:
                sentence_counter += 1
                for k in range(j):
                    sentence_count_list.append('sentence '+ str(sentence_counter))


        sentence_count_list_idx = 0
        for i in range(len(words_list_names)):
            for j in range(len(words_list_names[i])):
                row = ['review '+str(i+1), sentence_count_list[sentence_count_list_idx], words_list_names[i][j], final_sentiment_pred_names[i][j], final_aspect_pred_names[i][j],
                       final_aspect2_pred_names[i][j], final_sentiemnt_score_pred_names[i][j], final_aspect_score_pred_names[i][j]]
                new_data.append(row)
                sentence_count_list_idx += 1

        # 단어 단위로 정리한 df
        new_df = pd.DataFrame(new_data, columns=["review #", "sentence #", "word", "sentiment", "aspect", "aspect2", "sentiment_score", "aspect_score"])
        
        
        columns_to_process = ['sentiment', 'aspect', 'aspect2', 'sentiment_score', 'aspect_score']
        for col in columns_to_process:
            new_df[col] = new_df[col].apply(remove_bio_prefix_for_list)
        

        # new_df = new_df.join(new_df.groupby('sentence #').apply(majority_vote), on='sentence #')
        # new_df = new_df.drop(['sentiment', 'aspect', 'aspect2', 'sentiment_score', 'aspect_score'], axis=1)
        # new_df = new_df.rename(columns={'sentiment_majority': 'sentiment', 'aspect_majority': 'aspect',
        #                                 'aspect2_majority' : 'aspect2', 'sentiment_score_majority': 'sentiment_score',
        #                                 'aspect_score_majority': 'aspect_score'})

        # sentence #을 기준으로 word들을 합쳐서 하나의 문장을 만듭니다
        # 문장 단위로 정리한 df
        new_df_grouped_by_sentence = new_df.groupby('sentence #').agg({
            'review #': 'first',
            'word': ' '.join,
            'sentiment': lambda x: majority_vote(x),  # 다수결 방식으로 sentiment 결정
            'aspect': lambda x: majority_vote(x),     # 다수결 방식으로 aspect 결정
            'aspect2': lambda x: majority_vote(x),    # 다수결 방식으로 aspect2 결정
            'sentiment_score': lambda x: majority_vote(x),  # 다수결 방식으로 sentiment_score 결정
            'aspect_score': lambda x: majority_vote(x)      # 다수결 방식으로 aspect_score 결정
        }).reset_index()

        new_df_grouped_by_sentence = new_df_grouped_by_sentence.rename(columns={'word': 'sentence'})
        new_df_grouped_by_sentence['sentence_num'] = new_df_grouped_by_sentence['sentence #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
        new_df_grouped_by_sentence = new_df_grouped_by_sentence.sort_values(by='sentence_num').reset_index(drop=True)
        new_df_grouped_by_sentence = new_df_grouped_by_sentence.drop(['sentence_num'], axis=1)
        new_df_grouped_by_sentence = new_df_grouped_by_sentence[['review #', 'sentence #', 'sentence', 'sentiment', 'aspect', 'aspect2', 'sentiment_score', 'aspect_score']]
        
        new_df_grouped_by_sentence['review_num'] = new_df_grouped_by_sentence['review #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))

        
        # aspect에 대해서 O나 [PAD]가 아니면 our_topics에다가 집어넣어(review # 기준으로)
        # 집어넣기만 하면

        new_df_grouped_by_sentence["our_topics_dict"] = new_df_grouped_by_sentence.apply(create_ourt_topics_dict, axis=1)

        # 뽑힌 거 확인하려면 아래 주석 해제
        # print(new_df_grouped_by_sentence)
        # new_df_grouped_by_sentence.to_csv(file.replace("data_untagged_json","tagged_results_json").replace(".json", ".csv"), encoding='utf-8-sig')
        

        # 결과를 담을 리스트 초기화
        our_topics_list = []
        
        current_review = None
        current_list = []

        for index, row in new_df_grouped_by_sentence.iterrows():
            if current_review is None:
                current_review = row['review #']
            
            if row['review #'] == current_review:
                if row['our_topics_dict'] is not None:
                    current_list.append(row['our_topics_dict'])
            else:
                our_topics_list.append(current_list)
                current_review = row['review #']
                current_list = []
                if row['our_topics_dict'] is not None:
                    current_list.append(row['our_topics_dict'])
        # 마지막 review에 대한 처리
        # if current_list:
        our_topics_list.append(current_list)
        


        df = df.drop(['temp', '# of words in each sentence'], axis=1)
        # print(df)

        # df = df[df['content'] != ''] # 빈 텍스트 삭제

        # 처음 불러와서 전처리한 df에 our_topics 열 값 초기화
        df["our_topics"] = None

        # our_topics_list를 처음 불러온 df에다가 추가
        for i, lst in enumerate(our_topics_list):
            df.at[i, 'our_topics'] = lst

        df["mallSeq"] = df["mallSeq"].astype(str)
        df["matchNvMid"] = df["matchNvMid"].astype(str)
        df["nvMid"] = df["nvMid"].astype(str)



        # print(df)
        df.to_json(file.replace("data_untagged_json","tagged_results_json").replace('.json', '_역변환.json') ,force_ascii=False, orient='records')
        
        # utf-8-sig로 인코딩 방식 설정하기 위한 코드(더 좋은 게 있을 듯 무조건)
        with open(file.replace("data_untagged_json","tagged_results_json").replace('.json', '_역변환.json'), 'r', encoding='cp949') as f:
            json_data = f.read()
        
        json_data = json.loads(json_data) # json 읽은 거 list로 변환

        # find로 df["our_topics"]에서 찾고 못 찾으면 그냥 -1 입력(df["our_topics"]는 dict in list임)
        # 'our_topics' 안에 있는 'start_pos' 값을 'content'와 비교하여 변경
        for item in json_data:
            content = item['content']
            for topic in item['our_topics']:
                text = topic['text']
                if ('[UNK]' in topic['text']): # [UNK]가 있으면 replace하는 부분
                    topic['text'] = replace_all_unk_to_original(content, text)
                start_pos = content.find(topic['text'])
                topic['start_pos'] = start_pos
                if (topic['start_pos'] == -1):
                    topic['end_pos'] = -1
                else:
                    topic['end_pos'] = start_pos + len(topic['text'])
                
                if (topic['topic_score'] == 0):
                    if (len(topic['text']) <= 5):
                        topic['topic_score'] = 1
                    elif (len(topic['text']) <= 10):
                        topic['topic_score'] = 2
                    elif (len(topic['text']) <= 15):
                        topic['topic_score'] = 3
                    elif (len(topic['text']) <= 20):
                        topic['topic_score'] = 4
                    else:
                        topic['topic_score'] = 5

                if (topic['start_pos'] == -1): # 이제 남은 건 특수문자 띄어쓰기 문제(제발)
                    text = topic['text']
                    topic['text'] = remove_space_before_special_char(text)
                    start_pos = content.find(topic['text'])
                    topic['start_pos'] = start_pos
                    if (topic['start_pos'] != -1):
                        topic['end_pos'] = start_pos + len(topic['text'])




        json_data = json.dumps(json_data, ensure_ascii=False) # 다시 json 형식으로 맞게끔 변환

        with open(file.replace("data_untagged_json","tagged_results_json").replace('.json', '_역변환.json'), 'w', encoding='utf-8-sig') as f:
            f.write(json_data)

        print(file.replace("data_untagged_json","tagged_results_json").replace('.json', '_역변환.json')+" tagging finished")

    tagging_end_time = time.time() - tagging_start_time  # 모든 데이터에 대한 태깅 소요 시간
    # tagging_sample_per_sec = str(datetime.timedelta(seconds=(tagging_end_time / loader_len)))  # 한 샘플에 대한 태깅 소요 시간
    tagging_times = str(datetime.timedelta(seconds=tagging_end_time))  # 시:분:초 형식으로 변환
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("현재 시간:", current_time)


def inference_fn(config, data:list, tokenizer, model, enc_sentiment, enc_aspect, enc_aspect2, enc_sentiment_score, enc_aspect_score, device, log):
    log.info("inference start")
    model.eval()    

    # tagging_start_time = time.time()  # 태깅을 시작한 시간을 저장 (소요 시간 측정을 위함)
        
    # 예측값 변수 선언 --> 파일 별로 선언해줘야 할듯
    sentiment_preds = []
    aspect_preds = []
    aspect2_preds = []
    sentiment_score_preds = []
    aspect_score_preds = []

    ids_inputs = []
    words_list_for_file = [] # 각 리뷰에서 단어 별로 인코딩된 값 저장한 2차원 리스트 (리뷰, 단어)
    words_in_sent_for_file = [] # 하나의 파일에 대해서 각 리뷰의 문장 별 단어 개수
    df = pd.DataFrame(data)
    # df = read_json(data)
    df.loc[:, "content"] = df["content"].fillna(method="ffill")
    
    # 아래 부분 제거 필요. 전처리 로직.
    df["temp"] = df['content'].apply(preprocess_content)

    df['content'] = df['temp'].apply(lambda x: x[0])
    df['# of words in each sentence'] = df['temp'].apply(lambda x: x[1])

    df = df[df['content'] != ''] # 빈 텍스트 삭제(의미 없으니까)
    df = df.reset_index(drop=True)
    
    df['content'] = df['content'].apply(normalize_whitespace) # spacing 문제 해결 위해서 공백은 무조건 1칸으로 고정
    # df['# of words in each sentence'] = df['temp'].apply(lambda x: x[1])

    # sentences = df.groupby("Review #")["Word"].apply(list).values
    sentences = [text.split() for text in df["content"]]
    # blank_count = 0
    # for i in range(len(sentences)):
    #     for j in range(len(sentences[i])):
    #         if(sentences[i][j].replace(' ','') == ''):
    #             blank_count += 1

    # sentences = np.array(sentences)
    words_in_each_sentence = df["# of words in each sentence"].tolist() # 한 리뷰에 대해서 각 문장이 가지는 단어의 개수를 모은 2차원 리스트

    log.info("Tagging start")
    for i in tqdm(range(len(sentences)), disable=True):
        data, words_list, words_in_sent= parsing_data(tokenizer, sentences[i], words_in_each_sentence[i]) 
        # ids_list는 단어 단위로 묶은 것-->ids_list의 len이 단어 개수임 / words_in_sent는 리뷰 하나에 대한 문장이 가지는 단어의 개수(slicing)
        words_list_for_file.append(words_list)
        words_in_sent_for_file.append(words_in_sent)
        data = parsing_batch(data, device)
        predict_sentiment, predict_aspect, predict_aspect2, predict_sentiment_score, predict_aspect_score = model(**data)

        sentiment_pred = np.array(predict_sentiment).reshape(-1)
        aspect_pred = np.array(predict_aspect).reshape(-1)
        aspect2_pred = np.array(predict_aspect2).reshape(-1)
        sentiment_score_pred = np.array(predict_sentiment_score).reshape(-1)
        aspect_score_pred = np.array(predict_aspect_score).reshape(-1)
        
        ids_input = data['ids'].numpy().reshape(-1)


#     # remove padding indices
        indices_to_remove = np.where((ids_input == 2) | (ids_input == 3) | (ids_input == 0))
        sentiment_pred = np.delete(sentiment_pred, indices_to_remove)
        aspect_pred = np.delete(aspect_pred, indices_to_remove)
        aspect2_pred = np.delete(aspect2_pred, indices_to_remove)
        sentiment_score_pred = np.delete(sentiment_score_pred, indices_to_remove)
        aspect_score_pred = np.delete(aspect_score_pred, indices_to_remove)
        ids_input = np.delete(ids_input, indices_to_remove)



    # 모델의 예측 결과를 저장    
        sentiment_preds.extend(sentiment_pred)
        aspect_preds.extend(aspect_pred)
        aspect2_preds.extend(aspect2_pred)
        sentiment_score_preds.extend(sentiment_score_pred)
        aspect_score_preds.extend(aspect_score_pred)
        ids_inputs.extend(ids_input)

    

    # encoding 된 Sentiment와 Aspect Category를 Decoding (원 형태로 복원)
    sentiment_pred_names = enc_sentiment.inverse_transform(sentiment_preds)
    aspect_pred_names = enc_aspect.inverse_transform(aspect_preds)
    aspect2_pred_names = enc_aspect2.inverse_transform(aspect2_preds)
    sentiment_score_pred_names = enc_sentiment_score.inverse_transform(sentiment_score_preds)
    aspect_score_pred_names = enc_aspect_score.inverse_transform(aspect_score_preds)


    ids_input_names = tokenizer.decode(ids_inputs)

    words_list_names = []

    final_sentiment_pred_names = []
    final_aspect_pred_names = []
    final_aspect2_pred_names = []
    final_sentiemnt_score_pred_names = []
    final_aspect_score_pred_names = []

    start_idx = 0
    end_idx = 0
    

    for i in range(len(words_list_for_file)): # 리뷰 차원 늘려라 --> 해결
        # if (len(ids_list_for_file[i]) != 0): # slicing 하는 과정에서 길이가 길어서 짤리면 []가 들어가는 경우가 있어서 처리
        words_list_names_for_content = []
        final_sentiment_pred_names_for_content = []
        final_aspect_pred_names_for_content = []
        final_aspect2_pred_names_for_content = []
        final_sentiemnt_score_pred_names_for_content = []
        final_aspect_score_pred_names_for_content = []
        for j in range(len(words_list_for_file[i])):
            end_idx += len(words_list_for_file[i][j])
            words_list_names_for_content.append(tokenizer.decode(words_list_for_file[i][j]))
            final_sentiment_pred_names_for_content.append(sentiment_pred_names[start_idx:end_idx])
            final_aspect_pred_names_for_content.append(aspect_pred_names[start_idx:end_idx])
            final_aspect2_pred_names_for_content.append(aspect2_pred_names[start_idx:end_idx])
            final_sentiemnt_score_pred_names_for_content.append(sentiment_score_pred_names[start_idx:end_idx])
            final_aspect_score_pred_names_for_content.append(aspect_score_pred_names[start_idx:end_idx])
            start_idx = end_idx

        words_list_names.append(words_list_names_for_content) # content 별 단어를 모은 2차원 리스트(리뷰, 단어)
        final_sentiment_pred_names.append(final_sentiment_pred_names_for_content) # content 별 단어에 대한 토큰별 예측값 리스트를 모은 2차원 리스트
        final_aspect_pred_names.append(final_aspect_pred_names_for_content)
        final_aspect2_pred_names.append(final_aspect2_pred_names_for_content)
        final_sentiemnt_score_pred_names.append(final_sentiemnt_score_pred_names_for_content)
        final_aspect_score_pred_names.append(final_aspect_score_pred_names_for_content)


    # print("len(words_list_for_file):", len(words_list_for_file)) # 한 파일에 있는 content 개수 의미
    # print("len(words_list_names):", len(words_list_names)) 
    # print("len(ids_inputs):", len(ids_inputs))
    # print("len(sentiment_pred_names):", len(sentiment_pred_names))
    # print("len(aspect_pred_names):", len(aspect_pred_names))
    # print("len(aspect2_pred_names):", len(aspect2_pred_names))
    # print("len(sentiment_score_pred_names):", len(sentiment_score_pred_names))
    # print("len(aspect_score_pred_names):", len(aspect_score_pred_names))

    new_data = []
    sentence_count_list = []
    sentence_counter = 0
    for i in range(len(words_in_sent_for_file)):
        for j in words_in_sent_for_file[i]:
            sentence_counter += 1
            for k in range(j):
                sentence_count_list.append('sentence '+ str(sentence_counter))


    sentence_count_list_idx = 0
    for i in range(len(words_list_names)):
        for j in range(len(words_list_names[i])):
            row = ['review '+str(i+1), sentence_count_list[sentence_count_list_idx], words_list_names[i][j], final_sentiment_pred_names[i][j], final_aspect_pred_names[i][j],
                    final_aspect2_pred_names[i][j], final_sentiemnt_score_pred_names[i][j], final_aspect_score_pred_names[i][j]]
            new_data.append(row)
            sentence_count_list_idx += 1

    # 단어 단위로 정리한 df
    new_df = pd.DataFrame(new_data, columns=["review #", "sentence #", "word", "sentiment", "aspect", "aspect2", "sentiment_score", "aspect_score"])
    
    
    columns_to_process = ['sentiment', 'aspect', 'aspect2', 'sentiment_score', 'aspect_score']
    for col in columns_to_process:
        new_df[col] = new_df[col].apply(remove_bio_prefix_for_list)
    

    # new_df = new_df.join(new_df.groupby('sentence #').apply(majority_vote), on='sentence #')
    # new_df = new_df.drop(['sentiment', 'aspect', 'aspect2', 'sentiment_score', 'aspect_score'], axis=1)
    # new_df = new_df.rename(columns={'sentiment_majority': 'sentiment', 'aspect_majority': 'aspect',
    #                                 'aspect2_majority' : 'aspect2', 'sentiment_score_majority': 'sentiment_score',
    #                                 'aspect_score_majority': 'aspect_score'})

    # sentence #을 기준으로 word들을 합쳐서 하나의 문장을 만듭니다
    # 문장 단위로 정리한 df
    new_df_grouped_by_sentence = new_df.groupby('sentence #').agg({
        'review #': 'first',
        'word': ' '.join,
        'sentiment': lambda x: majority_vote(x),  # 다수결 방식으로 sentiment 결정
        'aspect': lambda x: majority_vote(x),     # 다수결 방식으로 aspect 결정
        'aspect2': lambda x: majority_vote(x),    # 다수결 방식으로 aspect2 결정
        'sentiment_score': lambda x: majority_vote(x),  # 다수결 방식으로 sentiment_score 결정
        'aspect_score': lambda x: majority_vote(x)      # 다수결 방식으로 aspect_score 결정
    }).reset_index()

    new_df_grouped_by_sentence = new_df_grouped_by_sentence.rename(columns={'word': 'sentence'})
    new_df_grouped_by_sentence['sentence_num'] = new_df_grouped_by_sentence['sentence #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
    new_df_grouped_by_sentence = new_df_grouped_by_sentence.sort_values(by='sentence_num').reset_index(drop=True)
    new_df_grouped_by_sentence = new_df_grouped_by_sentence.drop(['sentence_num'], axis=1)
    new_df_grouped_by_sentence = new_df_grouped_by_sentence[['review #', 'sentence #', 'sentence', 'sentiment', 'aspect', 'aspect2', 'sentiment_score', 'aspect_score']]
    
    new_df_grouped_by_sentence['review_num'] = new_df_grouped_by_sentence['review #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))

    
    # aspect에 대해서 O나 [PAD]가 아니면 our_topics에다가 집어넣어(review # 기준으로)
    # 집어넣기만 하면

    new_df_grouped_by_sentence["our_topics_dict"] = new_df_grouped_by_sentence.apply(create_ourt_topics_dict, axis=1)

    # 뽑힌 거 확인하려면 아래 주석 해제
    # print(new_df_grouped_by_sentence)
    # new_df_grouped_by_sentence.to_csv(file.replace("data_untagged_json","tagged_results_json").replace(".json", ".csv"), encoding='utf-8-sig')
    

    # 결과를 담을 리스트 초기화
    our_topics_list = []
    
    current_review = None
    current_list = []

    for index, row in new_df_grouped_by_sentence.iterrows():
        if current_review is None:
            current_review = row['review #']
        
        if row['review #'] == current_review:
            if row['our_topics_dict'] is not None:
                current_list.append(row['our_topics_dict'])
        else:
            our_topics_list.append(current_list)
            current_review = row['review #']
            current_list = []
            if row['our_topics_dict'] is not None:
                current_list.append(row['our_topics_dict'])
    # 마지막 review에 대한 처리
    # if current_list:
    our_topics_list.append(current_list)    

    df = df.drop(['temp', '# of words in each sentence'], axis=1)
    # print(df)

    # df = df[df['content'] != ''] # 빈 텍스트 삭제

    # 처음 불러와서 전처리한 df에 our_topics 열 값 초기화
    df["our_topics"] = None

    # our_topics_list를 처음 불러온 df에다가 추가
    for i, lst in enumerate(our_topics_list):
        df.at[i, 'our_topics'] = lst

    df["mallSeq"] = df["mallSeq"].astype(str)
    df["matchNvMid"] = df["matchNvMid"].astype(str)
    df["nvMid"] = df["nvMid"].astype(str)
        
    dict_data = df.to_dict(orient='records')

    # find로 df["our_topics"]에서 찾고 못 찾으면 그냥 -1 입력(df["our_topics"]는 dict in list임)
    # 'our_topics' 안에 있는 'start_pos' 값을 'content'와 비교하여 변경
    for item in dict_data:
        content = item['content']
        for topic in item['our_topics']:
            text = topic['text']
            if ('[UNK]' in topic['text']): # [UNK]가 있으면 replace하는 부분
                topic['text'] = replace_all_unk_to_original(content, text)
            start_pos = content.find(topic['text'])
            topic['start_pos'] = start_pos
            if (topic['start_pos'] == -1):
                topic['end_pos'] = -1
            else:
                topic['end_pos'] = start_pos + len(topic['text'])
            
            if (topic['topic_score'] == 0):
                if (len(topic['text']) <= 5):
                    topic['topic_score'] = 1
                elif (len(topic['text']) <= 10):
                    topic['topic_score'] = 2
                elif (len(topic['text']) <= 15):
                    topic['topic_score'] = 3
                elif (len(topic['text']) <= 20):
                    topic['topic_score'] = 4
                else:
                    topic['topic_score'] = 5

            if (topic['start_pos'] == -1): # 이제 남은 건 특수문자 띄어쓰기 문제(제발)
                text = topic['text']
                topic['text'] = remove_space_before_special_char(text)
                start_pos = content.find(topic['text'])
                topic['start_pos'] = start_pos
                if (topic['start_pos'] != -1):
                    topic['end_pos'] = start_pos + len(topic['text'])
    

    # tagging_end_time = time.time() - tagging_start_time  # 모든 데이터에 대한 태깅 소요 시간
    # tagging_sample_per_sec = str(datetime.timedelta(seconds=(tagging_end_time / loader_len)))  # 한 샘플에 대한 태깅 소요 시간
    # tagging_times = str(datetime.timedelta(seconds=tagging_end_time))  # 시:분:초 형식으로 변환
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # log.info(f"현재 시간: {current_time}")
    
    return dict_data









def tag_valid_fn(config, tokenizer, model, enc_sentiment, enc_aspect, enc_aspect2, enc_sentiment_score, enc_aspect_score, device, log):
    print("valid_tagging_start")
    model.eval()


    file_list = get_file_list(config.valid_fp, 'csv')

    tagging_start_time = time.time()  # 태깅을 시작한 시간을 저장 (소요 시간 측정을 위함)

    for file in file_list:
        if os.path.exists(file.replace("data_valid","tagged_results_valid").replace('.csv', '_역변환.csv')):
            print(file.replace("data_valid","tagged_results_valid").replace('.csv', '_역변환.csv')+" exists --> continue")
            continue

        # 예측값 변수 선언 --> 파일 별로 선언해줘야 할듯
        sentiment_preds = []
        aspect_preds = []
        aspect2_preds = []
        sentiment_score_preds = []
        aspect_score_preds = []

        ids_inputs = []
        words_list_for_file = [] # 각 리뷰에서 단어 별로 인코딩된 값 저장한 2차원 리스트 (리뷰, 단어)
        words_in_sent_for_file = [] # 하나의 파일에 대해서 각 리뷰의 문장 별 단어 개수

        df_og = read_csv(file)

        df_og.loc[:, "Review #"] = df_og["Review #"].fillna(method="ffill")
        df_og['Word Count'] = df_og.groupby(['Review #', 'Sentence #'])['Sentence #'].transform('size')
        df = df_og.groupby(['Review #', 'Sentence #']).agg({
            'Word': lambda x: ' '.join(x),  # sentence로 나중에 column 이름 변경
            'Word Count': 'first'
                      }).reset_index()
        
        df.rename(columns={'Word': 'Sentence'}, inplace=True)

        df['Review_Num'] = df['Review #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
        df['Sentence_Num'] = df['Sentence #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))

        df = df.sort_values(by=['Review_Num', 'Sentence_Num'], ascending=[True, True])

        df = df.drop(['Review_Num', 'Sentence_Num'], axis=1)

        df = df.groupby(['Review #']).agg({
            'Sentence': lambda x: ' '.join(x), # sentence로 나중에 column 이름 변경
            'Word Count': list
                      }).reset_index()
        
        df.rename(columns={'Sentence': 'content', 'Word Count': '# of words in each sentence'}, inplace=True)
        df['Review_Num'] = df['Review #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
        df = df.sort_values(by=['Review_Num'], ascending=[True])
        df = df.drop(['Review_Num'], axis=1)



        df = df[df['content'] != ''] # 빈 텍스트 삭제(의미 없으니까)
        df = df.reset_index(drop=True)

        sentences = [text.split() for text in df["content"]]
        

        sentences = np.array(sentences)
        words_in_each_sentence = df["# of words in each sentence"].tolist() # 한 리뷰에 대해서 각 문장이 가지는 단어의 개수를 모은 2차원 리스트


        print("Tagging "+ file)
        for i in tqdm(range(len(sentences))):
            data, words_list, words_in_sent= parsing_data(tokenizer, sentences[i], words_in_each_sentence[i]) 
            # ids_list는 단어 단위로 묶은 것-->ids_list의 len이 단어 개수임 / words_in_sent는 리뷰 하나에 대한 문장이 가지는 단어의 개수(slicing)
            words_list_for_file.append(words_list)
            words_in_sent_for_file.append(words_in_sent)
            predict_sentiment, predict_aspect, predict_aspect2, predict_sentiment_score, predict_aspect_score = model(**data)

            sentiment_pred = np.array(predict_sentiment).reshape(-1)
            aspect_pred = np.array(predict_aspect).reshape(-1)
            aspect2_pred = np.array(predict_aspect2).reshape(-1)
            sentiment_score_pred = np.array(predict_sentiment_score).reshape(-1)
            aspect_score_pred = np.array(predict_aspect_score).reshape(-1)
            
            ids_input = data['ids'].numpy().reshape(-1)


    #     # remove padding indices
            indices_to_remove = np.where((ids_input == 2) | (ids_input == 3) | (ids_input == 0))
            sentiment_pred = np.delete(sentiment_pred, indices_to_remove)
            aspect_pred = np.delete(aspect_pred, indices_to_remove)
            aspect2_pred = np.delete(aspect2_pred, indices_to_remove)
            sentiment_score_pred = np.delete(sentiment_score_pred, indices_to_remove)
            aspect_score_pred = np.delete(aspect_score_pred, indices_to_remove)
            ids_input = np.delete(ids_input, indices_to_remove)



        # 모델의 예측 결과를 저장
        
            sentiment_preds.extend(sentiment_pred)
            aspect_preds.extend(aspect_pred)
            aspect2_preds.extend(aspect2_pred)
            sentiment_score_preds.extend(sentiment_score_pred)
            aspect_score_preds.extend(aspect_score_pred)

            ids_inputs.extend(ids_input)

        

        # encoding 된 Sentiment와 Aspect Category를 Decoding (원 형태로 복원)
        sentiment_pred_names = enc_sentiment.inverse_transform(sentiment_preds)
        aspect_pred_names = enc_aspect.inverse_transform(aspect_preds)
        aspect2_pred_names = enc_aspect2.inverse_transform(aspect2_preds)
        sentiment_score_pred_names = enc_sentiment_score.inverse_transform(sentiment_score_preds)
        aspect_score_pred_names = enc_aspect_score.inverse_transform(aspect_score_preds)

    
        ids_input_names = tokenizer.decode(ids_inputs)

        words_list_names = []

        final_sentiment_pred_names = []
        final_aspect_pred_names = []
        final_aspect2_pred_names = []
        final_sentiemnt_score_pred_names = []
        final_aspect_score_pred_names = []

        start_idx = 0
        end_idx = 0
        

        for i in range(len(words_list_for_file)): # 리뷰 차원 늘려라 --> 해결
            # if (len(ids_list_for_file[i]) != 0): # slicing 하는 과정에서 길이가 길어서 짤리면 []가 들어가는 경우가 있어서 처리
            words_list_names_for_content = []
            final_sentiment_pred_names_for_content = []
            final_aspect_pred_names_for_content = []
            final_aspect2_pred_names_for_content = []
            final_sentiemnt_score_pred_names_for_content = []
            final_aspect_score_pred_names_for_content = []
            for j in range(len(words_list_for_file[i])):
                end_idx += len(words_list_for_file[i][j])
                words_list_names_for_content.append(tokenizer.decode(words_list_for_file[i][j]))
                final_sentiment_pred_names_for_content.append(sentiment_pred_names[start_idx:end_idx])
                final_aspect_pred_names_for_content.append(aspect_pred_names[start_idx:end_idx])
                final_aspect2_pred_names_for_content.append(aspect2_pred_names[start_idx:end_idx])
                final_sentiemnt_score_pred_names_for_content.append(sentiment_score_pred_names[start_idx:end_idx])
                final_aspect_score_pred_names_for_content.append(aspect_score_pred_names[start_idx:end_idx])
                start_idx = end_idx

            words_list_names.append(words_list_names_for_content) # content 별 단어를 모은 2차원 리스트(리뷰, 단어)
            final_sentiment_pred_names.append(final_sentiment_pred_names_for_content) # content 별 단어에 대한 토큰별 예측값 리스트를 모은 2차원 리스트
            final_aspect_pred_names.append(final_aspect_pred_names_for_content)
            final_aspect2_pred_names.append(final_aspect2_pred_names_for_content)
            final_sentiemnt_score_pred_names.append(final_sentiemnt_score_pred_names_for_content)
            final_aspect_score_pred_names.append(final_aspect_score_pred_names_for_content)


        new_data = []
        sentence_count_list = []
        # sentence_counter = 0
        for i in range(len(words_in_sent_for_file)):
            sentence_counter = 0 # 얘 위치로 sentence # 수정 가능
            for j in words_in_sent_for_file[i]:
                sentence_counter += 1
                for k in range(j):
                    sentence_count_list.append('Sentence '+ str(sentence_counter))


        sentence_count_list_idx = 0
        for i in range(len(words_list_names)):
            for j in range(len(words_list_names[i])):
                row = [df.at[i, 'Review #'], sentence_count_list[sentence_count_list_idx], words_list_names[i][j], final_sentiment_pred_names[i][j], final_aspect_pred_names[i][j],
                       final_aspect2_pred_names[i][j], final_sentiemnt_score_pred_names[i][j], final_aspect_score_pred_names[i][j]]
                new_data.append(row)
                sentence_count_list_idx += 1

        # 단어 단위로 정리한 df
        new_df = pd.DataFrame(new_data, columns=["review #", "sentence #", "word", "sentiment", "aspect", "aspect2", "sentiment_score", "aspect_score"])
        
        
        columns_to_process = ['sentiment', 'aspect', 'aspect2', 'sentiment_score', 'aspect_score']
        for col in columns_to_process:
            new_df[col] = new_df[col].apply(remove_bio_prefix_for_list)
        


        # sentence #을 기준으로 word들을 합쳐서 하나의 문장을 만듭니다
        # 문장 단위로 정리한 df
        new_df_grouped_by_sentence = new_df.groupby(['review #', 'sentence #']).agg({
            'word': ' '.join,
            'sentiment': lambda x: majority_vote(x),  # 다수결 방식으로 sentiment 결정
            'aspect': lambda x: majority_vote(x),     # 다수결 방식으로 aspect 결정
            'aspect2': lambda x: majority_vote(x),    # 다수결 방식으로 aspect2 결정
            'sentiment_score': lambda x: majority_vote(x),  # 다수결 방식으로 sentiment_score 결정
            'aspect_score': lambda x: majority_vote(x)      # 다수결 방식으로 aspect_score 결정
        }).reset_index()

        new_df_grouped_by_sentence = new_df_grouped_by_sentence.rename(columns={'review #': 'Review #', 'sentence #': 'Sentence #',
                                                                                'word': 'Sentence','sentiment': 'Sentiment',
                                                                                'aspect': 'Aspect','aspect2': 'Aspect2',
                                                                                'sentiment_score': 'Sentiment_Score',
                                                                                'aspect_score': 'Aspect_Score'
                                                                                })
        new_df_grouped_by_sentence['sentence_num'] = new_df_grouped_by_sentence['Sentence #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
        new_df_grouped_by_sentence['review_num'] = new_df_grouped_by_sentence['Review #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
        new_df_grouped_by_sentence = new_df_grouped_by_sentence.sort_values(by=['review_num', 'sentence_num'], ascending=[True, True]).reset_index(drop=True)
        new_df_grouped_by_sentence = new_df_grouped_by_sentence.drop(['review_num', 'sentence_num'], axis=1)
        new_df_grouped_by_sentence = new_df_grouped_by_sentence[['Review #', 'Sentence #', 'Sentence', 'Sentiment', 'Aspect', 'Aspect2', 'Sentiment_Score', 'Aspect_Score']]
        

        new_df_grouped_by_sentence.to_csv(file.replace("data_valid","tagged_results_valid").replace('.csv', '_역변환.csv'), encoding='utf-8-sig',
                                          index=False)
        
        # -------------------------------------------------------------------------------------------여기까지 수정
        # df에서 같은 review #에 해당하는 원본 content를 따로 new_df_grouped_by_sentence에 추가하고 [UNK]에 대해서 찾는 아이디어

        # utf-8-sig로 인코딩 방식 설정하기 위한 코드(더 좋은 게 있을 듯 무조건)
        # with open(file.replace("data_untagged_json","tagged_results_json").replace('.json', '_역변환.json'), 'r', encoding='cp949') as f:
        #     json_data = f.read()
        
        # json_data = json.loads(json_data) # json 읽은 거 list로 변환

        # # find로 df["our_topics"]에서 찾고 못 찾으면 그냥 -1 입력(df["our_topics"]는 dict in list임)
        # # 'our_topics' 안에 있는 'start_pos' 값을 'content'와 비교하여 변경
        # for item in json_data:
        #     content = item['content']
        #     for topic in item['our_topics']:
        #         text = topic['text']
        #         if ('[UNK]' in topic['text']): # [UNK]가 있으면 replace하는 부분
        #             topic['text'] = replace_all_unk_to_original(content, text)
        #         start_pos = content.find(topic['text'])
        #         topic['start_pos'] = start_pos
        #         if (topic['start_pos'] == -1):
        #             topic['end_pos'] = -1
        #         else:
        #             topic['end_pos'] = start_pos + len(topic['text'])
                
        #         if (topic['topic_score'] == 0):
        #             if (len(topic['text']) <= 5):
        #                 topic['topic_score'] = 1
        #             elif (len(topic['text']) <= 10):
        #                 topic['topic_score'] = 2
        #             elif (len(topic['text']) <= 15):
        #                 topic['topic_score'] = 3
        #             elif (len(topic['text']) <= 20):
        #                 topic['topic_score'] = 4
        #             else:
        #                 topic['topic_score'] = 5

        #         if (topic['start_pos'] == -1): # 이제 남은 건 특수문자 띄어쓰기 문제(제발)
        #             text = topic['text']
        #             topic['text'] = remove_space_before_special_char(text)
        #             start_pos = content.find(topic['text'])
        #             topic['start_pos'] = start_pos
        #             if (topic['start_pos'] != -1):
        #                 topic['end_pos'] = start_pos + len(topic['text'])




        # json_data = json.dumps(json_data, ensure_ascii=False) # 다시 json 형식으로 맞게끔 변환

        # with open(file.replace("data_untagged_json","tagged_results_json").replace('.json', '_역변환.json'), 'w', encoding='utf-8-sig') as f:
        #     f.write(json_data)

        print(file.replace("data_valid","tagged_results_valid").replace('.csv', '_역변환.csv')+" tagging finished")

    tagging_end_time = time.time() - tagging_start_time  # 모든 데이터에 대한 태깅 소요 시간
    # tagging_sample_per_sec = str(datetime.timedelta(seconds=(tagging_end_time / loader_len)))  # 한 샘플에 대한 태깅 소요 시간
    tagging_times = str(datetime.timedelta(seconds=tagging_end_time))  # 시:분:초 형식으로 변환
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("현재 시간:", current_time)








def valid_eval_fn(config):
    print("valid_eval_start")
    file_list = get_file_list(config.valid_fp, 'csv')

    valid_eval_time = time.time()  # valid_eval 시작한 시간을 저장 (소요 시간 측정을 위함)

    for file in file_list:
        df = read_csv(file)


        df["Aspect2"] = df["Aspect"]
        df = df.replace({"Aspect2": label_changing_rule})

        columns_to_process = ['Sentiment', 'Aspect', 'Aspect2', 'Sentiment_Score', 'Aspect_Score']
        for col in columns_to_process:
            df[col] = df[col].apply(remove_bio_prefix)

        df.rename(columns={'Word': 'Sentence'}, inplace=True)


        df_true = df.groupby(['Review #', 'Sentence #']).agg({
                    'Sentence': ' '.join,
                    'Sentiment': lambda x: majority_vote_for_valid_eval(x),  # 다수결 방식으로 sentiment 결정
                    'Aspect': lambda x: majority_vote_for_valid_eval(x),     # 다수결 방식으로 aspect 결정
                    'Aspect2': lambda x: majority_vote_for_valid_eval(x),     # 다수결 방식으로 aspect2 결정
                    'Sentiment_Score': lambda x: majority_vote_for_valid_eval(x),  # 다수결 방식으로 sentiment_score 결정
                    'Aspect_Score': lambda x: majority_vote_for_valid_eval(x)      # 다수결 방식으로 aspect_score 결정
                }).reset_index()

        df_true['Review_Num'] = df_true['Review #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
        df_true['Sentence_Num'] = df_true['Sentence #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))

        df_true = df_true.sort_values(by=['Review_Num', 'Sentence_Num'], ascending=[True, True])
        df_true = df_true.drop(['Review_Num', 'Sentence_Num'], axis=1).reset_index(drop=True)
        
        if not (os.path.exists(file.replace('data_valid', 'data_valid_for_eval').replace('.csv', '_for_eval.csv'))):
            df_true.to_csv(file.replace('data_valid', 'data_valid_for_eval').replace('.csv', '_for_eval.csv'), encoding='utf-8-sig', index=False)

        df_pred = read_csv(file.replace('data_valid', 'tagged_results_valid').replace('.csv', '_역변환.csv'), encoding='utf-8-sig')
        # df_pred = df_pred.drop(['Aspect2'], axis=1).reset_index(drop=True)
        merged_df = pd.merge(df_true, df_pred, on=['Review #', 'Sentence #'], suffixes=('_True', '_Pred'))

        merged_df.rename(columns={'Sentiment_True': 'Sentiment_actual',
                                  'Sentiment_Pred': 'Sentiment_predicted',
                                  'Aspect_True': 'Aspect_actual',
                                  'Aspect_Pred': 'Aspect_predicted',
                                  'Aspect2_True': 'Aspect2_actual',
                                  'Aspect2_Pred': 'Aspect2_predicted',
                                  'Sentiment_Score_True': 'Sentiment_Score_actual',
                                  'Sentiment_Score_Pred': 'Sentiment_Score_predicted',
                                  'Aspect_Score_True': 'Aspect_Score_actual',
                                  'Aspect_Score_Pred': 'Aspect_Score_predicted'}, inplace=True)
        
        merged_df.reset_index(drop=True, inplace=True)

        # 실제 값과 예측 값 추출
        y_true_sentiment = merged_df['Sentiment_actual']
        y_pred_sentiment = merged_df['Sentiment_predicted']

        y_true_aspect = merged_df['Aspect_actual']
        y_pred_aspect = merged_df['Aspect_predicted']

        y_true_aspect2 = merged_df['Aspect2_actual']
        y_pred_aspect2 = merged_df['Aspect2_predicted']

        y_true_sentiment_score = merged_df['Sentiment_Score_actual']
        y_pred_sentiment_score = merged_df['Sentiment_Score_predicted']

        y_true_aspect_score = merged_df['Aspect_Score_actual']
        y_pred_aspect_score = merged_df['Aspect_Score_predicted']


        # 성능 분석
        report_sentiment = classification_report(y_true_sentiment, y_pred_sentiment, digits=4)
        report_aspect = classification_report(y_true_aspect, y_pred_aspect, digits=4)
        report_aspect2 = classification_report(y_true_aspect2, y_pred_aspect2, digits=4)
        report_sentiment_score = classification_report(y_true_sentiment_score, y_pred_sentiment_score, digits=4)
        report_aspect_score = classification_report(y_true_aspect_score, y_pred_aspect_score, digits=4)
                
        # 결과 출력
        print(file + "성능 측정\n")
        print("Sentiment Report:\n", report_sentiment)
        print("Aspect Report:\n", report_aspect)
        print("Aspect2 Report:\n", report_aspect2)
        print("Sentiment Score Report:\n", report_sentiment_score)
        print("Aspect Score Report:\n", report_aspect_score)