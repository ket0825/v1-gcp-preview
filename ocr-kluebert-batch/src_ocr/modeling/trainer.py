from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from data_manager.parsers.label_unification.label_map import label_changing_rule
import torch
import time
import datetime
import json
import os
from typing import List, Dict, Any, Tuple
import traceback


from utils.file_io import get_file_list, read_json, read_csv
from collections import Counter
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
    aspect_loss_total = 0
    aspect2_loss_total = 0
    loader_len = data_loader.dataset.get_length()
    for data in tqdm(data_loader, total=loader_len):
        data = parsing_batch(data, device)
        optimizer.zero_grad() # backward를 위한 gradients 초기화
        aspect_loss, aspect2_loss, aspect, aspect2 = model(**data)

        # 각각의 손실에 대해 역전파
        aspect_loss.backward(retain_graph=True)
        aspect2_loss.backward(retain_graph=True)

        optimizer.step()
        scheduler.step()

        final_loss += (aspect_loss.item() + aspect2_loss.item()) / 2
        aspect_loss_total += aspect_loss.item()
        aspect2_loss_total += aspect2_loss.item()

    return final_loss / loader_len, aspect_loss_total / loader_len, aspect2_loss_total / loader_len


# 모델 평가
def eval_fn(data_loader, model, enc_aspect, enc_aspect2, device, log, f1_mode='micro', flag='valid'):
    model.eval()
    final_loss = 0
    aspect_loss_total = 0
    aspect2_loss_total = 0
    nb_eval_steps = 0
    # 성능 측정 변수 선언
    aspect_accuracy, aspect2_accuracy = 0, 0
    aspect_f1score, aspect2_f1score = 0, 0
    aspect_preds, aspect_labels = [], []
    aspect2_preds, aspect2_labels = [], []

    loader_len = len(data_loader)

    eval_start_time = time.time() # evaluation을 시작한 시간을 저장 (소요 시간 측정을 위함)
    for data in tqdm(data_loader, total=loader_len):
        data = parsing_batch(data, device)
        aspect_loss, aspect2_loss, predict_aspect, predict_aspect2 = model(**data)
        
        aspect_label = data['target_aspect'].cpu().numpy().reshape(-1)
        aspect2_label = data['target_aspect2'].cpu().numpy().reshape(-1)

        aspect_pred = np.array(predict_aspect).reshape(-1)
        aspect2_pred = np.array(predict_aspect2).reshape(-1)

        #remove padding indices
        pad_label_indices = np.where(aspect_label == 0)  # pad 레이블
        aspect_label = np.delete(aspect_label, pad_label_indices)
        aspect_pred = np.delete(aspect_pred, pad_label_indices)

        pad_label_indices = np.where(aspect2_label == 0)  # pad 레이블
        aspect2_label = np.delete(aspect2_label, pad_label_indices)
        aspect2_pred = np.delete(aspect2_pred, pad_label_indices)

        # Accuracy 및 F1-score 계산
        aspect_accuracy += accuracy_score(aspect_label, aspect_pred)
        aspect2_accuracy += accuracy_score(aspect2_label, aspect2_pred)

        aspect_f1score += f1_score(aspect_label, aspect_pred, average=f1_mode)
        aspect2_f1score += f1_score(aspect2_label, aspect2_pred, average=f1_mode)

        # target label과 모델의 예측 결과를 저장 => classification report 계산 위함
        aspect_labels.extend(aspect_label)
        aspect_preds.extend(aspect_pred)
        aspect2_labels.extend(aspect2_label)
        aspect2_preds.extend(aspect2_pred)

        final_loss += (aspect_loss.item() + aspect2_loss.item()) / 2
        aspect_loss_total += aspect_loss.item()
        aspect2_loss_total += aspect2_loss.item()
        nb_eval_steps += 1

    # encoding 된 Aspect Category를 Decoding (원 형태로 복원)
    aspect_pred_names = enc_aspect.inverse_transform(aspect_preds)
    aspect_label_names = enc_aspect.inverse_transform(aspect_labels)
    aspect2_pred_names = enc_aspect2.inverse_transform(aspect2_preds)
    aspect2_label_names = enc_aspect2.inverse_transform(aspect2_labels)

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

    eval_loss = final_loss / loader_len # model의 loss
    eval_end_time = time.time() - eval_start_time # 모든 데이터에 대한 평가 소요 시간
    eval_sample_per_sec = str(datetime.timedelta(seconds=(eval_end_time/loader_len)))# 한 샘플에 대한 평가 소요 시간
    eval_times = str(datetime.timedelta(seconds=eval_end_time)) # 시:분:초 형식으로 변환

    # validation 과정일 때는, 각 sample에 대한 개별 결과값은 출력하지 않음
    if flag == 'eval':
        for i in range(len(aspect_label_names)):
            if aspect_label_names[i] != aspect_pred_names[i]:
                asp_result = "X"
            else:
                asp_result = "O"
            if aspect2_label_names[i] != aspect2_pred_names[i]:
                asp2_result = "X"
            else:
                asp2_result = "O"
            log.info(f"[{i} >> Aspect : {asp_result} | Aspect2 : {asp2_result}] "
                     f"predicted aspect label: {aspect_pred_names[i]}, gold sentiment label: {aspect_label_names[i]} | "
                     f"predicted aspect2 label: {aspect2_pred_names[i]}, gold aspect label: {aspect2_label_names[i]} | ")

    log.info("*****" + "eval metrics" + "*****")
    log.info(f"eval_loss: {eval_loss}")
    log.info(f"eval_runtime: {eval_times}")
    log.info(f"eval_samples: {loader_len}")
    log.info(f"eval_samples_per_second: {eval_sample_per_sec}")
    log.info(f"Aspect Accuracy: {aspect_accuracy}")
    log.info(f"Aspect f1score {f1_mode} : {aspect_f1score}")
    log.info(f"Aspect2 Accuracy: {aspect2_accuracy}")
    log.info(f"Aspect2 f1score {f1_mode} : {aspect2_f1score}")
    log.info(f"Aspect Accuracy Report:")
    log.info(aspect_report)
    log.info(f"Aspect2 Accuracy Report:")
    log.info(aspect2_report)

    return eval_loss











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
    
    #TODO: 추가 여부 확인
    # words_list도 slicing 필요한듯.    
    # words_list_slicing_idx = 0
    token_counter = 0
    
    
    # for i, word in enumerate(words_list):        
    #     token_counter += len(word)
    #     if token_counter > len(ids):
    #         words_list_slicing_idx = i
    #         break
    
    # if token_counter > len(ids):            
    #     words_list = words_list[:words_list_slicing_idx] + words_list[words_list_slicing_idx][:token_counter - len(ids)]    
    #     updated_words_in_sent = words_in_sent[:words_list_slicing_idx] + [len(words_list[words_list_slicing_idx])]
    
    
        
    sliced_flattened_ids_list = flattened_ids_list[:max_len - 2]

    # sliced_flattened_ids_list를 다시 원래 구조로 되돌리기
    sliced_words_list = []
    current_length = 0


    last_idx = len(words_list) # 포함된 단어의 총 개수
    for i, sublist in enumerate(words_list):
        if current_length + len(sublist) > max_len - 2:
            if (max_len - 2 - current_length == 0): # 해당 단어는 토큰이 하나도 포함이 안되는거지
                last_idx = i
                pass
            else:# 단어 단위로 자르는 게 맞지 않나??
                sliced_words_list.append(sublist[:max_len - 2 - current_length])                
                last_idx = (i+1)
                # updated_words_in_sent.append(min(words_in_sent[i], max_len - 2 - current_length))
            break
        else:
            sliced_words_list.append(sublist)
            # updated_words_in_sent.append(words_in_sent[i])
            current_length += len(sublist)
    
    # 자르고 난 다음 sentence의 단어 개수 업데이트
    updated_words_in_sent = []  # 업데이트된 words_in_sent 리스트
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


def replace_newline(text):
    return text.replace('\n', ' ')


def preprocess_content(content:str):   
#     kss_sent_list = split_sentences(content)    
#     text = '\n'.join(kss_sent_list)
    # sentences = text.split('\n')
    # sentences = [[sent] for sent in sentences]
    # sentences_period_added = [[replace_newline(sent[0].strip()) + '.'] for sent in sentences if sent[0].strip()]
    # sentences_regexp = [regexp(sent_list) for sent_list in sentences_period_added]
    # preprocessed_content = ' '.join([''.join(row) for row in sentences_regexp])    
    # return text, preprocessed_content, sentences_regexp
    pass


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



def majority_vote(lists):
    all_elements = [element for sublist in lists for element in sublist]
    count = Counter(all_elements)
    return count.most_common(1)[0][0]
    

def create_our_topics_dict(row): # 각 문장 별로 dict 만드는 함수
    if (row['aspect'] == 'O' or row['aspect'] == '[PAD]'):
        return None
    else:
        return {
        "text": row['sentence'],
        "topic": row['aspect'],
        "sentence_pos": row['sentence_num'],
        "start_pos": 0,  # Start position placeholder
        "end_pos": 0,  # End position placeholder
        "bbox": 0
        }

def find_bbox(text_og, text_preprocessed, topic_data, bbox_text): # text는 텍스트 전체 / topic_data는 태깅된 딕셔너리 리스트 / bbox_text = ocr_list[n][2:]
    splitted_text_list = text_og.split('\n')
    
    cur_dict_list = []
    text_spacingx = text_og.replace('\n', '')
    text_spacingx = text_spacingx.replace(' ', '')
    text_spacingx = text_spacingx.strip()

    start_pos_spacingx = 0
    end_pos_spacingx = 0
    for i in range(len(bbox_text)):
        end_pos_spacingx = start_pos_spacingx + len(bbox_text[i]['text'].replace('\n', '').strip().replace(' ', '')) ### 수정(\n)
        bbox_text[i]['start_pos_spacingx'] = start_pos_spacingx
        bbox_text[i]['end_pos_spacingx'] = end_pos_spacingx
        start_pos_spacingx = end_pos_spacingx


    try:
        new_json_topic_list = [] # json file에 bbox 추가해주기 위한 list(딕셔너리를 넣는 리스트)
        
        cur_pos = 0 # 텍스트 분리를 위한 pos
        find_start_idx = 0 # bbox 찾기 위한 텍스트 pos
        bbox_idx = 0 # 텍스트 찾으면서 bbox 추가하기 위한 인덱스
    
        if (type(topic_data) == list and len(topic_data) > 0): # 태깅 되어있는 경우
            topic_data = sorted([item for item in topic_data if isinstance(item, dict)], key=lambda x: len(x['text']), reverse=True)
            new_topic_data = []
            for i in range(len(topic_data)): # len(topic_data)
                text_to_find = splitted_text_list[topic_data[i]['sentence_pos'] - 1]
                new_start_pos = text_og.find(text_to_find)
                new_end_pos = new_start_pos + len(text_to_find)
                for j in range(len(new_topic_data)): # 검사하면서 new_pos들 업데이트해주는 부분
                    if(new_topic_data[j]['start_pos'] <= new_start_pos < new_topic_data[j]['end_pos']): # pos가 중복된 경우
                        new_start_pos = text_og.find(text_to_find,new_end_pos)
                        new_end_pos = new_start_pos + len(text_to_find)
                    else:
                        continue
                new_dict = {'text': topic_data[i]['text'], 'topic': topic_data[i]['topic'], 'start_pos': new_start_pos, 'end_pos': new_end_pos}
                new_topic_data.append(new_dict)
                new_topic_data = sorted([item for item in new_topic_data if isinstance(item, dict)], key=lambda x: x['start_pos'], reverse=False)

            
            for j in range(2 * len(new_topic_data) + 1):
                if(j % 2 == 0 and j != 2*len(new_topic_data)): # 태깅 안 되어있는 case
                    start_pos = cur_pos
                    end_pos = new_topic_data[j//2]['start_pos']
                    if(start_pos == end_pos): # 이 구간에 해당하는 텍스트가 없다는 의미
                        continue
                    cur_dict = {'original_text' : text_og[start_pos:end_pos], 'topic' : 'O', 'bbox' : []}
                    if(cur_dict['original_text'] == ' ' or cur_dict['original_text'] == '\n'):
                        continue
                    cur_dict_list.append(cur_dict)
                    cur_pos = end_pos
                elif(j % 2 == 0 and j == 2*len(new_topic_data)): # 태깅 안 되어있는 case 중 마지막 부분 분리
                    start_pos = cur_pos
                    end_pos = len(text_og)
                    if(start_pos == end_pos):
                        continue
                    cur_dict = {'original_text' : text_og[start_pos:end_pos], 'topic' : 'O', 'bbox' : []}
                    if(cur_dict['original_text'] == ' ' or cur_dict['original_text'] == '\n'):
                        continue
                    cur_dict_list.append(cur_dict)
                    cur_pos = end_pos
                else: # 태깅되어 있는 데이터 처리
                    start_pos = cur_pos
                    end_pos = new_topic_data[j//2]['end_pos']
                    if (start_pos >= end_pos):
                        continue
                    cur_dict = {'original_text' : text_og[start_pos:end_pos], 'topic' : new_topic_data[j//2]['topic'], 'bbox' : []}
                    

                    # start_idx_found --> 띄어쓰기 없앤 텍스트에서 찾은 시작 인덱스
                    start_idx_found = text_spacingx.find(cur_dict['original_text'].strip().replace(' ', '').replace('\n', '') , find_start_idx)
                    # end_idx_found --> 띄어쓰기 없앤 텍스트에서 찾은 마지막 인덱스
                    end_idx_found = start_idx_found + len(cur_dict['original_text'].strip().replace(' ', '').replace('\n', ''))

                    
                    while (not(bbox_text[bbox_idx]['start_pos_spacingx'] <= start_idx_found <= (bbox_text[bbox_idx]['end_pos_spacingx'] - 1))):
                        bbox_idx += 1
                    bbox_start_idx = bbox_idx
                    while (not(bbox_text[bbox_idx]['start_pos_spacingx'] <= (end_idx_found - 1) <= (bbox_text[bbox_idx]['end_pos_spacingx'] - 1))):
                        bbox_idx += 1
                    bbox_end_idx = bbox_idx


                    for idx in range(bbox_start_idx, bbox_end_idx + 1):
                        cur_dict['bbox'].append(bbox_text[idx]['bbox'])
                    
                    new_json_topic_dict = {'text' : new_topic_data[j//2]['text'], 'topic' : new_topic_data[j//2]['topic'],
                                           'start_pos' : text_preprocessed.find(new_topic_data[j//2]['text']),
                                           'end_pos' : text_preprocessed.find(new_topic_data[j//2]['text']) + len(new_topic_data[j//2]['text']),
                                           'bbox' : cur_dict['bbox']}
                    new_json_topic_list.append(new_json_topic_dict)
                    
                    

                    find_start_idx = end_idx_found

                    cur_dict_list.append(cur_dict)
                    cur_pos = end_pos
                    # print("처리 끝")

        else: # 태깅 안 되어있는 경우
            cur_dict = {'original_text' : text_og, 'topic' : 'O', 'bbox' : []}
            cur_dict_list.append(cur_dict)
        
        # print("n 루프 하나 끝!")
        return new_json_topic_list
    
    except IndexError as e: # 태깅 실수로 인한 에러 발생 시 코드
        print("IndexError 발생!!!")
        new_json_topic_list = [] # json file에 bbox 추가해주기 위한 list(딕셔너리를 넣는 리스트)

        cur_pos = 0 # 텍스트 분리를 위한 pos

        if (type(topic_data) == list and len(topic_data) > 0): # 태깅 되어있는 경우
            topic_data = sorted([item for item in topic_data if isinstance(item, dict)], key=lambda x: len(x['text']), reverse=True)
            new_topic_data = []
            for i in range(len(topic_data)): # len(topic_data)
                text_to_find = splitted_text_list[topic_data[i]['sentence_pos'] - 1]
                new_start_pos = text_og.find(text_to_find)
                new_end_pos = new_start_pos + len(text_to_find)
                for j in range(len(new_topic_data)): # 검사하면서 new_pos들 업데이트해주는 부분
                    if(new_topic_data[j]['start_pos'] <= new_start_pos < new_topic_data[j]['end_pos']):
                        new_start_pos = text_og.find(text_to_find,new_end_pos)
                        new_end_pos = new_start_pos + len(text_to_find)
                    else:
                        continue
                new_dict = {'text': topic_data[i]['text'], 'topic': topic_data[i]['topic'], 'start_pos': new_start_pos, 'end_pos': new_end_pos}
                new_topic_data.append(new_dict)
                new_topic_data = sorted([item for item in new_topic_data if isinstance(item, dict)], key=lambda x: x['start_pos'], reverse=False)


            for j in range(2 * len(new_topic_data) + 1):
                if(j % 2 == 0 and j != 2*len(new_topic_data)): # 태깅 안 되어있는 case
                    start_pos = cur_pos
                    end_pos = new_topic_data[j//2]['start_pos']
                    if(start_pos == end_pos):
                        continue
                    cur_dict = {'original_text' : text_og[start_pos:end_pos], 'topic' : 'O', 'bbox' : []}
                    if(cur_dict['original_text'] == ' ' or cur_dict['original_text'] == '\n'):
                        continue
                    cur_dict_list.append(cur_dict)
                    cur_pos = end_pos
                elif(j % 2 == 0 and j == 2*len(new_topic_data)): # 태깅 안 되어있는 case 중 마지막 부분 분리
                    start_pos = cur_pos
                    end_pos = len(text_og)
                    if(start_pos == end_pos):
                        continue
                    cur_dict = {'original_text' : text_og[start_pos:end_pos], 'topic' : 'O', 'bbox' : []}
                    if(cur_dict['original_text'] == ' ' or cur_dict['original_text'] == '\n'):
                        continue
                    cur_dict_list.append(cur_dict)
                    cur_pos = end_pos
                else: # 태깅되어 있는 데이터 처리
                    # print("태깅된 데이터 처리 시작")
                    bbox_idx = 0 # 텍스트 찾으면서 bbox 추가하기 위한 인덱스

                    start_pos = cur_pos
                    end_pos = new_topic_data[j//2]['end_pos']
                    if (start_pos >= end_pos):
                        continue
                    cur_dict = {'original_text' : text_og[start_pos:end_pos], 'topic' : new_topic_data[j//2]['topic'], 'bbox' : []}
                    

                    # start_idx_found --> 띄어쓰기 없앤 텍스트에서 찾은 시작 인덱스
                    start_idx_found = text_spacingx.find(cur_dict['original_text'].strip().replace(' ', '').replace('\n', ''))
                    # end_idx_found --> 띄어쓰기 없앤 텍스트에서 찾은 마지막 인덱스
                    end_idx_found = start_idx_found + len(cur_dict['original_text'].strip().replace(' ', '').replace('\n', ''))

                    
                    while (not(bbox_text[bbox_idx]['start_pos_spacingx'] <= start_idx_found <= (bbox_text[bbox_idx]['end_pos_spacingx'] - 1))):
                        bbox_idx += 1
                    bbox_start_idx = bbox_idx
                    while (not(bbox_text[bbox_idx]['start_pos_spacingx'] <= (end_idx_found - 1) <= (bbox_text[bbox_idx]['end_pos_spacingx'] - 1))):
                        bbox_idx += 1
                    bbox_end_idx = bbox_idx


                    for idx in range(bbox_start_idx, bbox_end_idx + 1):
                        cur_dict['bbox'].append(bbox_text[idx]['bbox'])
                    
                    new_json_topic_dict = {'text' : new_topic_data[j//2]['text'], 'topic' : new_topic_data[j//2]['topic'],
                                           'start_pos' : text_preprocessed.find(new_topic_data[j//2]['text']),
                                           'end_pos' : text_preprocessed.find(new_topic_data[j//2]['text']) + len(new_topic_data[j//2]['text']),
                                           'bbox' : cur_dict['bbox']}
                    new_json_topic_list.append(new_json_topic_dict)

                    

                    cur_dict_list.append(cur_dict)
                    cur_pos = end_pos
                    # print("처리 끝")

        else: # 태깅 안 되어있는 경우
            cur_dict = {'original_text' : text_og, 'topic' : 'O', 'bbox' : []}
            cur_dict_list.append(cur_dict)
        
        # print("n 루프 하나 끝!")
        return new_json_topic_list
    
    

def replace_all_unk_to_original(content, text):
    # [UNK]를 찾아서 위치와 해당 단어를 저장할 리스트 초기화
    content_spacingx = content.replace(' ','')
    text_spacingx = text.replace(' ','')
    
    unk_positions = []
    unk_pattern = re.compile(r'\[UNK\]')

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
        text = re.sub(r'\[UNK\]', match.group(i+1), text, count=1)
    
    return text


def remove_space_before_special_char(text):
    # 정규 표현식을 컴파일하여 패턴 생성
    
    # 컴파일된 패턴을 사용하여 특수 문자 앞의 공백을 제거
    normalized_text = special_char_pattern.sub(r'\1', text)
    
    return normalized_text


def words_count_per_sent(sent_list):
    no_words_in_sentence_list = []
    for row in sent_list:
        for element in row:
            no_words_in_sentence_list.append(len(element.split()))
    
    return no_words_in_sentence_list

def preprocess_fn2(config):
    print("preprocessing_start")
    file_list = get_file_list(config.preprocessing_fp, 'json')        
    
    for fp in file_list:
        exist_file = fp.replace("data_json_copy","preprocessed_results_json").replace('.json', '_전처리.json')        
        
        if os.path.exists(exist_file):            
            print(f"{exist_file} exists --> continue")
            continue

        # 아마 형식 통일되면 아래와 같이 가능할듯.
        df = read_json(fp)
        df["img_str"] = df["img_str"].fillna(method="ffill")        

        df["temp"] = df['img_str'].apply(preprocess_content)                
        df['img_str_preprocessed'] = df['temp'].apply(lambda x: x[0])
        df['img_sent_list_preprocessed'] = df['temp'].apply(lambda x: x[1])
        df.drop('temp', axis=1, inplace=True)
        
        df = df[df['img_str'] != ''] # 빈 텍스트 삭제(의미 없으니까)
        df = df[df['img_str_preprocessed'] != ''] # 빈 텍스트 삭제(의미 없으니까)
        df = df.reset_index(drop=True)
        
        df['img_str_preprocessed'] = df['img_str_preprocessed'].apply(normalize_whitespace) # spacing 문제 해결 위해서 공백은 무조건 1칸으로 고정        
        df = df[['img_str', 'img_str_preprocessed', 'img_sent_list_preprocessed', 'bbox_text']]

        data_dict = df.to_dict(orient='records')        
        with open(exist_file, 'w', encoding='utf-8-sig') as json_file:
            json.dump(data_dict, json_file, indent=4, ensure_ascii=False)
        
        
        
def update_json_structure(config):
    print("Updating_json_structure")
    file_list = get_file_list(config.json_fp, 'json')

    for file in file_list:
        if os.path.exists(file.replace("data_json","data_json_structure")):
            print(file.replace("data_json","data_json_structure")+" exists --> continue")
            continue

        with open(file, 'r', encoding='utf-8-sig') as json_file:
            ocr_data = json.load(json_file) # JSON 파일 불러오기
        df = pd.DataFrame()
        df['img_str'] = []
        df['bbox_text'] = []

        for data in ocr_data:
            if (type(data[1]) == list):
                content = data[0]  # 긴 텍스트 추출
                our_topics = data[1]
                bbox_text = data[2:]  # 나머지 데이터를 DataFrame으로 변환
                df = df.append({'img_str': content, 'our_topics': our_topics, 'bbox_text': bbox_text}, ignore_index=True)
            else:
                content = data[0]  # 긴 텍스트 추출
                bbox_text = data[1:]  # 나머지 데이터를 DataFrame으로 변환
                df = df.append({'img_str': content, 'bbox_text': bbox_text, 'our_topics': [],}, ignore_index=True)

        data_dict = df.to_dict(orient='records')

        with open(file.replace("data_json","data_json_structure"), 'w', encoding='utf-8-sig') as json_file:
            json.dump(data_dict, json_file, indent=4, ensure_ascii=False)


def preprocess_fn(config): # 전처리와 모델 예측 분리(그 중에서 전처리 함수)
    if config.need_preprocessing:
        update_json_structure(config)
        print("preprocessing_start")
        file_list = get_file_list(config.preprocessing_fp, 'json')

        for file in file_list:
            fp = file.replace("data_json_structure","preprocessed_results_json").replace('.json', '_전처리.json')
            if os.path.exists(fp):
                print(fp+" exists --> continue")
                continue


            with open(file, 'r', encoding='utf-8-sig') as json_file:
                ocr_data = json.load(json_file) # JSON 파일 불러오기
            df = pd.DataFrame()            
            df['img_str'] = []
            df['bbox_text'] = []

            for data in ocr_data:
                content = data['img_str']  # 긴 텍스트 추출
                bbox_text = data['bbox_text']  # 나머지 데이터를 DataFrame으로 변환
                df = df.append({'img_str': content, 'img_str_preprocessed': content, 'bbox_text': bbox_text}, ignore_index=True)


            df["img_str"] = df["img_str"].fillna(method="ffill")
            df["img_str_preprocessed"] = df["img_str_preprocessed"].fillna(method="ffill")            
            
            # df['img_str'] ,  df['img_str_preprocessed'], df['img_sent_list_preprocessed'] = df['img_str'].apply(preprocess_content)
            
            df["temp"] = df['img_str'].apply(preprocess_content)
            df['img_str'] = df['temp'].apply(lambda x: x[0])
            df['img_str_preprocessed'] = df['temp'].apply(lambda x: x[1])
            df['img_sent_list_preprocessed'] = df['temp'].apply(lambda x: x[2])
            # 1. df.drop()
            # 2. df.drop을 df에 할당하는 거.            
            df.drop(['temp'], axis=1, inplace=True)


            df = df[df['img_str'] != '']
            df = df[df['img_str_preprocessed'] != ''] # 빈 텍스트 삭제(의미 없으니까)
            df.reset_index(drop=True, inplace=True)
            
            df['img_str_preprocessed'] = df['img_str_preprocessed'].apply(normalize_whitespace) # spacing 문제 해결 위해서 공백은 무조건 1칸으로 고정
            
            df = df[['img_str', 'img_str_preprocessed', 'img_sent_list_preprocessed', 'bbox_text']]

            data_dict = df.to_dict(orient='records')

            with open(fp, 'w', encoding='utf-8-sig') as json_file:
                json.dump(data_dict, json_file, indent=4, ensure_ascii=False)


# input tensor의 구조 변경을 위한 함수
def parsing_batch_data(data:dict, device): # Not dataloader
    d = {}
    for k in data.keys():
        d[k] = list(data[k])
    for k in d.keys():
        d[k] = torch.stack(d[k]).to(device)
    return d 
    
    
    

def inference_fn(config, seller_spec, tokenizer, model, enc_aspect, enc_aspect2, device, log):
        
    # 예측값 변수 선언 --> 파일 별로 선언해줘야 할듯
    aspect_preds = []
    aspect2_preds = []

    ids_inputs = []
    words_list_for_file = [] # 각 리뷰에서 단어 별로 인코딩된 값 저장한 2차원 리스트 (리뷰, 단어)
    words_in_sent_for_file = [] # 하나의 파일에 대해서 각 리뷰의 문장 별 단어 개수        
    
    try:
        df = pd.DataFrame(seller_spec)

        words_in_each_sentence = df['img_sent_list_preprocessed'].apply(words_count_per_sent).tolist() # 한 리뷰에 대해서 각 문장이 가지는 단어의 개수를 모은 2차원 리스트

        sentences = [text.split() for text in df["img_str_preprocessed"]]                
        
        for i in tqdm(range(len(sentences))):
            data, words_list, words_in_sent= parsing_data(tokenizer, sentences[i], words_in_each_sentence[i]) 
            # ids_list는 단어 단위로 묶은 것-->ids_list의 len이 단어 개수임 / words_in_sent는 리뷰 하나에 대한 문장이 가지는 단어의 개수(slicing)
            words_list_for_file.append(words_list)
            words_in_sent_for_file.append(words_in_sent)
            data = parsing_batch_data(data, device)
            predict_aspect, predict_aspect2 = model(**data)            
            
            aspect_pred = np.array(predict_aspect).reshape(-1)
            aspect2_pred = np.array(predict_aspect2).reshape(-1)
            ids_input = data['ids'].numpy().reshape(-1)

        # remove padding indices
            indices_to_remove = np.where((ids_input == 2) | (ids_input == 3) | (ids_input == 0))
            aspect_pred = np.delete(aspect_pred, indices_to_remove)
            aspect2_pred = np.delete(aspect2_pred, indices_to_remove)
            ids_input = np.delete(ids_input, indices_to_remove)

        # 모델의 예측 결과를 저장
            aspect_preds.extend(aspect_pred)
            aspect2_preds.extend(aspect2_pred)
            ids_inputs.extend(ids_input)

        # encoding 된 Sentiment와 Aspect Category를 Decoding (원 형태로 복원)
        aspect_pred_names = enc_aspect.inverse_transform(aspect_preds)
        aspect2_pred_names = enc_aspect2.inverse_transform(aspect2_preds)

        words_list_names = []
        final_aspect_pred_names = []
        final_aspect2_pred_names = []
        start_idx = 0
        end_idx = 0        
        for i in range(len(words_list_for_file)): # 리뷰 차원 늘려라 --> 해결
            # if (len(ids_list_for_file[i]) != 0): # slicing 하는 과정에서 길이가 길어서 짤리면 []가 들어가는 경우가 있어서 처리
            words_list_names_for_content = []
            final_aspect_pred_names_for_content = []
            final_aspect2_pred_names_for_content = []
            for j in range(len(words_list_for_file[i])):
                end_idx += len(words_list_for_file[i][j])
                words_list_names_for_content.append(tokenizer.decode(words_list_for_file[i][j]))
                final_aspect_pred_names_for_content.append(aspect_pred_names[start_idx:end_idx])
                final_aspect2_pred_names_for_content.append(aspect2_pred_names[start_idx:end_idx])
                start_idx = end_idx

            words_list_names.append(words_list_names_for_content) # content 별 단어를 모은 2차원 리스트(리뷰, 단어)
            final_aspect_pred_names.append(final_aspect_pred_names_for_content) # content 별 단어에 대한 토큰별 예측값 리스트를 모은 2차원 리스트
            final_aspect2_pred_names.append(final_aspect2_pred_names_for_content)


        # TODO: 코드 리팩토링 필요.
        # new_data = []
        # sentence_count_list = []
        # sentence_counter = 0
        # for i in range(len(words_in_sent_for_file)):    # c++ 코드와 같음.
        #     for j in words_in_sent_for_file[i]:
        #         sentence_counter += 1
        #         sentence_count_list.extend(['sentence '+ str(sentence_counter)]*j)

        # TODO: 코드 리팩토링 필요.
        new_data = []
        sentence_count_list = []
        sentence_counter = 0
        for i in range(len(words_in_sent_for_file)):
            for j in words_in_sent_for_file[i]:
                sentence_counter += 1                
                
                #1
                sentence_count_list.extend(['sentence '+ str(sentence_counter)]*j)
                #2
                # for k in range(j):
                #     sentence_count_list.append('sentence '+ str(sentence_counter))

        sentence_count_list_idx = 0
        for i in range(len(words_list_names)):
            for j in range(len(words_list_names[i])):
                row = ['ocr '+str(i+1), sentence_count_list[sentence_count_list_idx], words_list_names[i][j], final_aspect_pred_names[i][j],
                        final_aspect2_pred_names[i][j]]
                new_data.append(row)
                sentence_count_list_idx += 1

        # 단어 단위로 정리한 df
        new_df = pd.DataFrame(new_data, columns=["ocr #", "sentence #", "word", "aspect", "aspect2"])        
        
        columns_to_process = ['aspect', 'aspect2']
        for col in columns_to_process:
            new_df[col] = new_df[col].apply(remove_bio_prefix_for_list)
        
        # sentence #을 기준으로 word들을 합쳐서 하나의 문장을 만듭니다
        # 문장 단위로 정리한 df
        new_df_grouped_by_sentence = new_df.groupby('sentence #').agg({
            'ocr #': 'first',
            'word': ' '.join,
            'aspect': lambda x: majority_vote(x),     # 다수결 방식으로 aspect 결정
            'aspect2': lambda x: majority_vote(x)    # 다수결 방식으로 aspect2 결정
        }).reset_index()

        new_df_grouped_by_sentence = new_df_grouped_by_sentence.rename(columns={'word': 'sentence'})
        new_df_grouped_by_sentence['sentence_num'] = new_df_grouped_by_sentence['sentence #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
        new_df_grouped_by_sentence = new_df_grouped_by_sentence.sort_values(by='sentence_num').reset_index(drop=True)
        new_df_grouped_by_sentence = new_df_grouped_by_sentence.drop(['sentence_num'], axis=1)
        new_df_grouped_by_sentence = new_df_grouped_by_sentence[['ocr #', 'sentence #', 'sentence', 'aspect', 'aspect2']]

        # ocr 값의 이전 상태를 저장할 변수
        previous_ocr = None
        # sentence의 번호를 저장할 변수
        sentence_number = 1

        # 데이터를 순회하며 ocr 값이 변경될 때마다 sentence 값을 다시 설정
        for i in range(len(new_df_grouped_by_sentence)):
            current_ocr = new_df_grouped_by_sentence.values[i][0]  # 현재 ocr 값
            
            # ocr 값이 변경되었거나 처음인 경우 sentence를 다시 1로 설정
            if current_ocr != previous_ocr:
                sentence_number = 1
                previous_ocr = current_ocr
            
            # sentence 값을 업데이트
            new_df_grouped_by_sentence.values[i][1] = f'sentence {sentence_number}'
            sentence_number += 1  # 다음 sentence 번호를 위해 증가

        
        new_df_grouped_by_sentence['sentence_num'] = new_df_grouped_by_sentence['sentence #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))


        new_df_grouped_by_sentence["our_topics_dict"] = new_df_grouped_by_sentence.apply(create_our_topics_dict, axis=1)

        

        # 결과를 담을 리스트 초기화
        our_topics_list = []
        
        current_ocr = None
        current_list = []

        for index, row in new_df_grouped_by_sentence.iterrows():
            if current_ocr is None:
                current_ocr = row['ocr #']
            
            if row['ocr #'] == current_ocr:
                if row['our_topics_dict'] is not None:
                    current_list.append(row['our_topics_dict'])
            else:
                our_topics_list.append(current_list)
                current_ocr = row['ocr #']
                current_list = []
                if row['our_topics_dict'] is not None:
                    current_list.append(row['our_topics_dict'])
        # 마지막 review에 대한 처리
        our_topics_list.append(current_list)                

        # 처음 불러와서 전처리한 df에 our_topics 열 값 초기화
        df["our_topics"] = None

        # our_topics_list를 처음 불러온 df에다가 추가
        for i, lst in enumerate(our_topics_list):
            df.at[i, 'our_topics'] = lst

        df.drop(['img_sent_list_preprocessed'], axis=1, inplace=True)
        df = df[['img_str', 'img_str_preprocessed', 'our_topics', 'bbox_text']]

        # print(df)
        # df.to_json(file.replace("preprocessed_results_json","tagged_results_json").replace('_전처리.json', '_역변환.json') ,force_ascii=False, orient='records')
        
        # with open(file.replace("preprocessed_results_json","tagged_results_json").replace('_전처리.json', '_역변환.json'), 'r', encoding='cp949') as f:
        #     json_data = f.read()
            
        # 1. cpu 많이 쓰는거. 일반적으로 for문 많이. 이런것들... => multiprocessing.
        # 2. I/O bound. 파일 읽고, 쓰기. request, response 주고 받고 하는 것도 network. => (cpu 안쓰임. 파일 읽고 쓰는데 시간이 걸림.) => multi-threading. co routine, async, task...
        
        # json_data = json.loads(json_data) # json 읽은 거 list로 변환
        json_data  = df.to_dict(orient='records')

        for item in json_data:
            content = item['img_str_preprocessed']                                
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

                if (topic['start_pos'] == -1): # 이제 남은 건 특수문자 띄어쓰기 문제
                    text = topic['text']
                    topic['text'] = remove_space_before_special_char(text)
                    start_pos = content.find(topic['text'])
                    topic['start_pos'] = start_pos
                    if (topic['start_pos'] != -1):
                            topic['end_pos'] = start_pos + len(topic['text'])
            
        
        new_order = ['img_str', 'our_topics', 'bbox_text']
        new_ocr_list = []
        
        ocr_list = json_data
        for n in range(len(ocr_list)):
            text_og = ocr_list[n]['img_str']
            text_preprocessed = ocr_list[n]['img_str_preprocessed']
            topic_data = ocr_list[n]['our_topics']
            bbox_text = ocr_list[n]['bbox_text']
            new_json_topic_list = find_bbox(text_og, text_preprocessed, topic_data, bbox_text)
            if(new_json_topic_list != []):
                ocr_list[n]['our_topics'] = new_json_topic_list
            del ocr_list[n]['img_str']
            ocr_list[n]['img_str'] = ocr_list[n]['img_str_preprocessed']
            del ocr_list[n]['img_str_preprocessed']
            for item in bbox_text:
                del item["start_pos_spacingx"]
                del item["end_pos_spacingx"]
            new_ocr_dict = {key: ocr_list[n][key] for key in new_order}
            new_ocr_list.append(new_ocr_dict)
            
        return new_ocr_list    
    except Exception as e:
        log.error(f"[ERROR] {e}, \n{traceback.format_exc()}")
        return None



def tag_fn(config, tokenizer, model, enc_aspect, enc_aspect2, device, log):
    print("tagging_start")
    model.eval()


    file_list = get_file_list(config.tagging_fp, 'json')

    tagging_start_time = time.time()  # 태깅을 시작한 시간을 저장 (소요 시간 측정을 위함)

    for file in file_list:
        if os.path.exists(file.replace("preprocessed_results_json","tagged_results_json").replace('_전처리.json', '_역변환.json')):
            print(file.replace("preprocessed_results_json","tagged_results_json").replace('_전처리.json', '_역변환.json')+" exists --> continue")
            continue

        # 예측값 변수 선언 --> 파일 별로 선언해줘야 할듯
        aspect_preds = []
        aspect2_preds = []

        ids_inputs = []
        words_list_for_file = [] # 각 리뷰에서 단어 별로 인코딩된 값 저장한 2차원 리스트 (리뷰, 단어)
        words_in_sent_for_file = [] # 하나의 파일에 대해서 각 리뷰의 문장 별 단어 개수

        with open(file, 'r', encoding='utf-8-sig') as json_file:
            ocr_data = json.load(json_file) # JSON 파일 불러오기
        
        df = pd.DataFrame(ocr_data)


        df['# of words in each sentence'] = df['img_sent_list_preprocessed'].apply(words_count_per_sent)

        

        sentences = [text.split() for text in df["img_str_preprocessed"]]

        # sentences = np.array(sentences)
        words_in_each_sentence = df["# of words in each sentence"].tolist() # 한 리뷰에 대해서 각 문장이 가지는 단어의 개수를 모은 2차원 리스트


        print("Tagging "+ file)        
        
        for i in tqdm(range(len(sentences))):
            start_time = time.time()
            data, words_list, words_in_sent= parsing_data(tokenizer, sentences[i], words_in_each_sentence[i]) 
            # ids_list는 단어 단위로 묶은 것-->ids_list의 len이 단어 개수임 / words_in_sent는 리뷰 하나에 대한 문장이 가지는 단어의 개수(slicing)
            words_list_for_file.append(words_list)
            words_in_sent_for_file.append(words_in_sent)
            data = parsing_batch(data, device)
            predict_aspect, predict_aspect2 = model(**data)
            
            end_time = time.time()
            
            print(f"Time taken for prediction: {end_time - start_time:.2f} seconds")

           
            aspect_pred = np.array(predict_aspect).reshape(-1)
            aspect2_pred = np.array(predict_aspect2).reshape(-1)
           
            
            ids_input = data['ids'].numpy().reshape(-1)


        # remove padding indices
            indices_to_remove = np.where((ids_input == 2) | (ids_input == 3) | (ids_input == 0))
            aspect_pred = np.delete(aspect_pred, indices_to_remove)
            aspect2_pred = np.delete(aspect2_pred, indices_to_remove)
            ids_input = np.delete(ids_input, indices_to_remove)



        # 모델의 예측 결과를 저장
            aspect_preds.extend(aspect_pred)
            aspect2_preds.extend(aspect2_pred)


            ids_inputs.extend(ids_input)

        

        # encoding 된 Sentiment와 Aspect Category를 Decoding (원 형태로 복원)
        aspect_pred_names = enc_aspect.inverse_transform(aspect_preds)
        aspect2_pred_names = enc_aspect2.inverse_transform(aspect2_preds)


    
        ids_input_names = tokenizer.decode(ids_inputs)

        words_list_names = []
        final_aspect_pred_names = []
        final_aspect2_pred_names = []
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
                final_aspect_pred_names_for_content.append(aspect_pred_names[start_idx:end_idx])
                final_aspect2_pred_names_for_content.append(aspect2_pred_names[start_idx:end_idx])
                start_idx = end_idx

            words_list_names.append(words_list_names_for_content) # content 별 단어를 모은 2차원 리스트(리뷰, 단어)
            final_aspect_pred_names.append(final_aspect_pred_names_for_content) # content 별 단어에 대한 토큰별 예측값 리스트를 모은 2차원 리스트
            final_aspect2_pred_names.append(final_aspect2_pred_names_for_content)


        # TODO: 코드 리팩토링 필요.
        # new_data = []
        # sentence_count_list = []
        # sentence_counter = 0
        # for i in range(len(words_in_sent_for_file)):    # c++ 코드와 같음.
        #     for j in words_in_sent_for_file[i]:
        #         sentence_counter += 1
        #         sentence_count_list.extend(['sentence '+ str(sentence_counter)]*j)

        # TODO: 코드 리팩토링 필요.
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
                row = ['ocr '+str(i+1), sentence_count_list[sentence_count_list_idx], words_list_names[i][j], final_aspect_pred_names[i][j],
                       final_aspect2_pred_names[i][j]]
                new_data.append(row)
                sentence_count_list_idx += 1

        # 단어 단위로 정리한 df
        new_df = pd.DataFrame(new_data, columns=["ocr #", "sentence #", "word", "aspect", "aspect2"])
        
        
        columns_to_process = ['aspect', 'aspect2']
        for col in columns_to_process:
            new_df[col] = new_df[col].apply(remove_bio_prefix_for_list)
        

        # sentence #을 기준으로 word들을 합쳐서 하나의 문장을 만듭니다
        # 문장 단위로 정리한 df
        new_df_grouped_by_sentence = new_df.groupby('sentence #').agg({
            'ocr #': 'first',
            'word': ' '.join,
            'aspect': lambda x: majority_vote(x),     # 다수결 방식으로 aspect 결정
            'aspect2': lambda x: majority_vote(x)    # 다수결 방식으로 aspect2 결정
        }).reset_index()

        new_df_grouped_by_sentence = new_df_grouped_by_sentence.rename(columns={'word': 'sentence'})
        new_df_grouped_by_sentence['sentence_num'] = new_df_grouped_by_sentence['sentence #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
        new_df_grouped_by_sentence = new_df_grouped_by_sentence.sort_values(by='sentence_num').reset_index(drop=True)
        new_df_grouped_by_sentence = new_df_grouped_by_sentence.drop(['sentence_num'], axis=1)
        new_df_grouped_by_sentence = new_df_grouped_by_sentence[['ocr #', 'sentence #', 'sentence', 'aspect', 'aspect2']]

        # ocr 값의 이전 상태를 저장할 변수
        previous_ocr = None
        # sentence의 번호를 저장할 변수
        sentence_number = 1

        # 데이터를 순회하며 ocr 값이 변경될 때마다 sentence 값을 다시 설정
        for i in range(len(new_df_grouped_by_sentence)):
            current_ocr = new_df_grouped_by_sentence.values[i][0]  # 현재 ocr 값
            
            # ocr 값이 변경되었거나 처음인 경우 sentence를 다시 1로 설정
            if current_ocr != previous_ocr:
                sentence_number = 1
                previous_ocr = current_ocr
            
            # sentence 값을 업데이트
            new_df_grouped_by_sentence.values[i][1] = f'sentence {sentence_number}'
            sentence_number += 1  # 다음 sentence 번호를 위해 증가

        
        new_df_grouped_by_sentence['sentence_num'] = new_df_grouped_by_sentence['sentence #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))


        new_df_grouped_by_sentence["our_topics_dict"] = new_df_grouped_by_sentence.apply(create_our_topics_dict, axis=1)

        

        # 결과를 담을 리스트 초기화
        our_topics_list = []
        
        current_ocr = None
        current_list = []

        for index, row in new_df_grouped_by_sentence.iterrows():
            if current_ocr is None:
                current_ocr = row['ocr #']
            
            if row['ocr #'] == current_ocr:
                if row['our_topics_dict'] is not None:
                    current_list.append(row['our_topics_dict'])
            else:
                our_topics_list.append(current_list)
                current_ocr = row['ocr #']
                current_list = []
                if row['our_topics_dict'] is not None:
                    current_list.append(row['our_topics_dict'])
        # 마지막 review에 대한 처리
        our_topics_list.append(current_list)
        

        df = df.drop(['# of words in each sentence'], axis=1)


        # 처음 불러와서 전처리한 df에 our_topics 열 값 초기화
        df["our_topics"] = None

        # our_topics_list를 처음 불러온 df에다가 추가
        for i, lst in enumerate(our_topics_list):
            df.at[i, 'our_topics'] = lst


        df = df.drop(['img_sent_list_preprocessed'], axis=1)
        df = df[['img_str', 'img_str_preprocessed', 'our_topics', 'bbox_text']]

        # print(df)
        df.to_json(file.replace("preprocessed_results_json","tagged_results_json").replace('_전처리.json', '_역변환.json') ,force_ascii=False, orient='records')
        
        with open(file.replace("preprocessed_results_json","tagged_results_json").replace('_전처리.json', '_역변환.json'), 'r', encoding='cp949') as f:
            json_data = f.read()
            
        # 1. cpu 많이 쓰는거. 일반적으로 for문 많이. 이런것들... => multiprocessing.
        # 2. I/O bound. 파일 읽고, 쓰기. request, response 주고 받고 하는 것도 network. => (cpu 안쓰임. 파일 읽고 쓰는데 시간이 걸림.) => multi-threading. co routine, async, task...
        
        json_data = json.loads(json_data) # json 읽은 거 list로 변환

        for item in json_data:
            content = item['img_str_preprocessed']            
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

                if (topic['start_pos'] == -1): # 이제 남은 건 특수문자 띄어쓰기 문제
                    text = topic['text']
                    topic['text'] = remove_space_before_special_char(text)
                    start_pos = content.find(topic['text'])
                    topic['start_pos'] = start_pos
                    if (topic['start_pos'] != -1):
                        topic['end_pos'] = start_pos + len(topic['text'])


        
        json_data = json.dumps(json_data, ensure_ascii=False) # 다시 json 형식으로 맞게끔 변환

        with open(file.replace("preprocessed_results_json","tagged_results_json").replace('_전처리.json', '_역변환.json'), 'w', encoding='utf-8-sig') as f:
            f.write(json_data)
        
    
    directory = 'resources_ocr/tagged_results_json/'
    file_list_to_find_bbox = os.listdir(directory)
    file_list_to_find_bbox.sort()
    new_order = ['img_str', 'our_topics', 'bbox_text']
    for filename in file_list_to_find_bbox:
        if filename.endswith('.json'):  # JSON 파일인지 확인
            new_ocr_list = []
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8-sig') as file_to_find_bbox:
                ocr_list = json.load(file_to_find_bbox) # JSON 파일 불러오기
            print("Finding_bbox " + filename)
            for n in range(len(ocr_list)):
                text_og = ocr_list[n]['img_str']
                text_preprocessed = ocr_list[n]['img_str_preprocessed']
                topic_data = ocr_list[n]['our_topics']
                bbox_text = ocr_list[n]['bbox_text']
                new_json_topic_list = find_bbox(text_og, text_preprocessed, topic_data, bbox_text)
                if(new_json_topic_list != []):
                    ocr_list[n]['our_topics'] = new_json_topic_list
                del ocr_list[n]['img_str']
                ocr_list[n]['img_str'] = ocr_list[n]['img_str_preprocessed']
                del ocr_list[n]['img_str_preprocessed']
                for item in bbox_text:
                    del item["start_pos_spacingx"]
                    del item["end_pos_spacingx"]
                new_ocr_dict = {key: ocr_list[n][key] for key in new_order}
                new_ocr_list.append(new_ocr_dict)
                
                

            

            with open(filepath, 'w', encoding='utf-8-sig') as file_found_bbox:
                json.dump(new_ocr_list, file_found_bbox, indent='\t', ensure_ascii=False)


    print(" Process(Tagging / Finding_bbox) finished")
    tagging_end_time = time.time() - tagging_start_time  # 모든 데이터에 대한 태깅 소요 시간
    tagging_times = str(datetime.timedelta(seconds=tagging_end_time))  # 시:분:초 형식으로 변환
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("현재 시간:", current_time)












def tag_valid_fn(config, tokenizer, model, enc_aspect, enc_aspect2, device, log):
    print("valid_tagging_start")
    model.eval()


    file_list = get_file_list(config.valid_fp, 'csv')

    tagging_start_time = time.time()  # 태깅을 시작한 시간을 저장 (소요 시간 측정을 위함)

    for file in file_list:
        if os.path.exists(file.replace('valid/valid.csv', 'valid_tagged/valid_역변환.csv')):
            print(file.replace('valid/valid.csv', 'valid_tagged/valid_역변환.csv')+" exists --> continue")
            continue

        # 예측값 변수 선언 --> 파일 별로 선언해줘야 할듯
        aspect_preds = []
        aspect2_preds = []

        ids_inputs = []
        words_list_for_file = [] # 각 리뷰에서 단어 별로 인코딩된 값 저장한 2차원 리스트 (리뷰, 단어)
        words_in_sent_for_file = [] # 하나의 파일에 대해서 각 리뷰의 문장 별 단어 개수

        df_og = read_csv(file)

        df_og.loc[:, "Ocr #"] = df_og["Ocr #"].fillna(method="ffill")
        df_og['Word Count'] = df_og.groupby(['Ocr #', 'Sentence #'])['Sentence #'].transform('size')
        df = df_og.groupby(['Ocr #', 'Sentence #']).agg({
            'Word': lambda x: ' '.join(x),  # sentence로 나중에 column 이름 변경
            'Word Count': 'first'
                      }).reset_index()
        
        df.rename(columns={'Word': 'Sentence'}, inplace=True)

        df['Ocr_Num'] = df['Ocr #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
        df['Sentence_Num'] = df['Sentence #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))

        df = df.sort_values(by=['Ocr_Num', 'Sentence_Num'], ascending=[True, True])

        df = df.drop(['Ocr_Num', 'Sentence_Num'], axis=1)

        df = df.groupby(['Ocr #']).agg({
            'Sentence': lambda x: ' '.join(x), # sentence로 나중에 column 이름 변경
            'Word Count': list
                      }).reset_index()
        
        df.rename(columns={'Sentence': 'content', 'Word Count': '# of words in each sentence'}, inplace=True)
        df['Ocr_Num'] = df['Ocr #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
        df = df.sort_values(by=['Ocr_Num'], ascending=[True])
        df = df.drop(['Ocr_Num'], axis=1)



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
            predict_aspect, predict_aspect2= model(**data)

            
            aspect_pred = np.array(predict_aspect).reshape(-1)
            aspect2_pred = np.array(predict_aspect2).reshape(-1)

            
            ids_input = data['ids'].numpy().reshape(-1)


    #     # remove padding indices
            indices_to_remove = np.where((ids_input == 2) | (ids_input == 3) | (ids_input == 0))

            aspect_pred = np.delete(aspect_pred, indices_to_remove)
            aspect2_pred = np.delete(aspect2_pred, indices_to_remove)

            ids_input = np.delete(ids_input, indices_to_remove)



        # 모델의 예측 결과를 저장
        

            aspect_preds.extend(aspect_pred)
            aspect2_preds.extend(aspect2_pred)

            ids_inputs.extend(ids_input)

        

        # encoding 된 Sentiment와 Aspect Category를 Decoding (원 형태로 복원)

        aspect_pred_names = enc_aspect.inverse_transform(aspect_preds)
        aspect2_pred_names = enc_aspect2.inverse_transform(aspect2_preds)


    
        ids_input_names = tokenizer.decode(ids_inputs)

        words_list_names = []


        final_aspect_pred_names = []
        final_aspect2_pred_names = []


        start_idx = 0
        end_idx = 0
        

        for i in range(len(words_list_for_file)): # 리뷰 차원 늘려라 --> 해결
            # if (len(ids_list_for_file[i]) != 0): # slicing 하는 과정에서 길이가 길어서 짤리면 []가 들어가는 경우가 있어서 처리
            words_list_names_for_content = []

            final_aspect_pred_names_for_content = []
            final_aspect2_pred_names_for_content = []

            for j in range(len(words_list_for_file[i])):
                end_idx += len(words_list_for_file[i][j])
                words_list_names_for_content.append(tokenizer.decode(words_list_for_file[i][j]))

                final_aspect_pred_names_for_content.append(aspect_pred_names[start_idx:end_idx])
                final_aspect2_pred_names_for_content.append(aspect2_pred_names[start_idx:end_idx])

                start_idx = end_idx

            words_list_names.append(words_list_names_for_content) # content 별 단어를 모은 2차원 리스트(리뷰, 단어)

           # content 별 단어에 대한 토큰별 예측값 리스트를 모은 2차원 리스트
            final_aspect_pred_names.append(final_aspect_pred_names_for_content)
            final_aspect2_pred_names.append(final_aspect2_pred_names_for_content)



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
                row = [df.at[i, 'Ocr #'], sentence_count_list[sentence_count_list_idx], words_list_names[i][j], final_aspect_pred_names[i][j],
                       final_aspect2_pred_names[i][j]] 
                new_data.append(row)
                sentence_count_list_idx += 1

        # 단어 단위로 정리한 df
        new_df = pd.DataFrame(new_data, columns=["ocr #", "sentence #", "word", "aspect", "aspect2"])
        
        
        columns_to_process = ['aspect', 'aspect2']
        for col in columns_to_process:
            new_df[col] = new_df[col].apply(remove_bio_prefix_for_list)
        


        # sentence #을 기준으로 word들을 합쳐서 하나의 문장을 만듭니다
        # 문장 단위로 정리한 df
        new_df_grouped_by_sentence = new_df.groupby(['ocr #', 'sentence #']).agg({
            'word': ' '.join,
            'aspect': lambda x: majority_vote(x),     # 다수결 방식으로 aspect 결정
            'aspect2': lambda x: majority_vote(x)    # 다수결 방식으로 aspect2 결정
            }).reset_index()

        new_df_grouped_by_sentence = new_df_grouped_by_sentence.rename(columns={'ocr #': 'Ocr #', 'sentence #': 'Sentence #',
                                                                                'word': 'Sentence','aspect': 'Aspect','aspect2': 'Aspect2'
                                                                                })
        new_df_grouped_by_sentence['sentence_num'] = new_df_grouped_by_sentence['Sentence #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
        new_df_grouped_by_sentence['ocr_num'] = new_df_grouped_by_sentence['Ocr #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
        new_df_grouped_by_sentence = new_df_grouped_by_sentence.sort_values(by=['ocr_num', 'sentence_num'], ascending=[True, True]).reset_index(drop=True)
        new_df_grouped_by_sentence = new_df_grouped_by_sentence.drop(['ocr_num', 'sentence_num'], axis=1)
        new_df_grouped_by_sentence = new_df_grouped_by_sentence[['Ocr #', 'Sentence #', 'Sentence', 'Aspect', 'Aspect2']]
        

        new_df_grouped_by_sentence.to_csv(file.replace('valid/valid.csv', 'valid_tagged/valid_역변환.csv'), encoding='utf-8-sig',
                                          index=False)
        
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

        print(file.replace('valid/valid.csv', 'valid_tagged/valid_역변환.csv')+" tagging finished")

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

        columns_to_process = ['Aspect', 'Aspect2']
        for col in columns_to_process:
            df[col] = df[col].apply(remove_bio_prefix)

        df.rename(columns={'Word': 'Sentence'}, inplace=True)


        df_true = df.groupby(['Ocr #', 'Sentence #']).agg({
                    'Sentence': ' '.join,
                    'Aspect': lambda x: majority_vote_for_valid_eval(x),     # 다수결 방식으로 aspect 결정
                    'Aspect2': lambda x: majority_vote_for_valid_eval(x),     # 다수결 방식으로 aspect2 결정
                }).reset_index()

        df_true['Ocr_Num'] = df_true['Ocr #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))
        df_true['Sentence_Num'] = df_true['Sentence #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))

        df_true = df_true.sort_values(by=['Ocr_Num', 'Sentence_Num'], ascending=[True, True])
        df_true = df_true.drop(['Ocr_Num', 'Sentence_Num'], axis=1).reset_index(drop=True)
        
        if not (os.path.exists(file.replace('valid/valid.csv', 'valid_for_eval/valid_for_eval.csv'))):
            df_true.to_csv(file.replace('valid/valid.csv', 'valid_for_eval/valid_for_eval.csv'), encoding='utf-8-sig', index=False)

        df_pred = read_csv(file.replace('valid/valid.csv', 'valid_tagged/valid_역변환.csv'), encoding='utf-8-sig')
        # df_pred = df_pred.drop(['Aspect2'], axis=1).reset_index(drop=True)
        merged_df = pd.merge(df_true, df_pred, on=['Ocr #', 'Sentence #'], suffixes=('_True', '_Pred'))

        merged_df.rename(columns={'Aspect_True': 'Aspect_actual',
                                  'Aspect_Pred': 'Aspect_predicted',
                                  'Aspect2_True': 'Aspect2_actual',
                                  'Aspect2_Pred': 'Aspect2_predicted'}, inplace=True)
        
        merged_df.reset_index(drop=True, inplace=True)

        # 실제 값과 예측 값 추출
        y_true_aspect = merged_df['Aspect_actual']
        y_pred_aspect = merged_df['Aspect_predicted']

        y_true_aspect2 = merged_df['Aspect2_actual']
        y_pred_aspect2 = merged_df['Aspect2_predicted']


        # 성능 분석
        report_aspect = classification_report(y_true_aspect, y_pred_aspect, digits=4)
        report_aspect2 = classification_report(y_true_aspect2, y_pred_aspect2, digits=4)

                
        # 결과 출력
        print(file + "성능 측정\n")
        print("Aspect Report:\n", report_aspect)
        print("Aspect2 Report:\n", report_aspect2)
