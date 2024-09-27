import os
import json
import pandas as pd
import re


def add_period_to_texts(text_list): # 문장 마침표 추가 함수
    texts_with_period = [text.strip() + '.' if not text.strip().endswith('.') else text.strip() for text in text_list]
    return texts_with_period


def concat_find_bbox(text, topic_data, bbox_list):
    cur_dict_list = []
    text_spacingx = text.replace('\n', '')  
    text_spacingx = text_spacingx.replace(' ', '')
    text_spacingx = text_spacingx.strip()
 

    start_pos_spacingx = 0
    end_pos_spacingx = 0
    for i in range(len(bbox_list)):
        end_pos_spacingx = start_pos_spacingx + len(bbox_list[i]['text'].strip().replace(' ', ''))
        bbox_list[i]['start_pos_spacingx'] = start_pos_spacingx
        bbox_list[i]['end_pos_spacingx'] = end_pos_spacingx
        start_pos_spacingx = end_pos_spacingx
 


    try:
        cur_pos = 0 # 텍스트 분리를 위한 pos
        find_start_idx = 0 # bbox 찾기 위한 텍스트 pos
        bbox_idx = 0 # 텍스트 찾으면서 bbox 추가하기 위한 인덱스
    
        if (type(topic_data) == list): # 태깅 되어있는 경우
            topic_data = sorted([item for item in topic_data if isinstance(item, dict)], key=lambda x: len(x['text']), reverse=True)
            new_topic_data = []
            for i in range(len(topic_data)): # len(topic_data)
                text_to_find = topic_data[i]['text']
                new_start_pos = text.find(text_to_find)
                new_end_pos = new_start_pos + len(text_to_find)
                for j in range(len(new_topic_data)): # 검사하면서 new_pos들 업데이트해주는 부분
                    if(new_topic_data[j]['start_pos'] <= new_start_pos < new_topic_data[j]['end_pos']):
                        new_start_pos = text.find(text_to_find,new_end_pos)
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
                    cur_dict = {'original_text' : text[start_pos:end_pos], 'topic' : 'O', 'bbox' : []}
                    if(cur_dict['original_text'] == ' ' or cur_dict['original_text'] == '\n'):
                        continue
                    cur_dict_list.append(cur_dict)
                    cur_pos = end_pos
                elif(j % 2 == 0 and j == 2*len(new_topic_data)): # 태깅 안 되어있는 case 중 마지막 부분 분리
                    start_pos = cur_pos
                    end_pos = len(text)
                    if(start_pos == end_pos):
                        continue
                    cur_dict = {'original_text' : text[start_pos:end_pos], 'topic' : 'O', 'bbox' : []}
                    if(cur_dict['original_text'] == ' ' or cur_dict['original_text'] == '\n'):
                        continue
                    cur_dict_list.append(cur_dict)
                    cur_pos = end_pos
                else: # 태깅되어 있는 데이터 처리
                    start_pos = cur_pos
                    end_pos = new_topic_data[j//2]['end_pos']
                    if (start_pos >= end_pos):
                        continue
                    cur_dict = {'original_text' : text[start_pos:end_pos], 'topic' : new_topic_data[j//2]['topic'], 'bbox' : []}
                    

                    # start_idx_found --> 띄어쓰기 없앤 텍스트에서 찾은 시작 인덱스
                    start_idx_found = text_spacingx.find(cur_dict['original_text'].strip().replace(' ', '').replace('\n', '') , find_start_idx)
                    # end_idx_found --> 띄어쓰기 없앤 텍스트에서 찾은 마지막 인덱스
                    end_idx_found = start_idx_found + len(cur_dict['original_text'].strip().replace(' ', '').replace('\n', ''))

                    
                    while (not(bbox_list[bbox_idx]['start_pos_spacingx'] <= start_idx_found <= (bbox_list[bbox_idx]['end_pos_spacingx'] - 1))):
                        bbox_idx += 1
                    bbox_start_idx = bbox_idx
                    while (not(bbox_list[bbox_idx]['start_pos_spacingx'] <= (end_idx_found - 1) <= (bbox_list[bbox_idx]['end_pos_spacingx'] - 1))):
                        bbox_idx += 1
                    bbox_end_idx = bbox_idx


                    for idx in range(bbox_start_idx, bbox_end_idx + 1):
                        cur_dict['bbox'].append(bbox_list[idx]['bbox'])
                    

                    find_start_idx = end_idx_found

                    cur_dict_list.append(cur_dict)
                    cur_pos = end_pos
                    

        else: # 태깅 안 되어있는 경우
            cur_dict = {'original_text' : text, 'topic' : 'O', 'bbox' : []}
            cur_dict_list.append(cur_dict)
        
        
        return cur_dict_list
    
    except IndexError as e: # 태깅 실수로 인한 에러 발생 시 코드
        cur_pos = 0 # 텍스트 분리를 위한 pos

        if (type(topic_data) == list): # 태깅 되어있는 경우
            topic_data = sorted([item for item in topic_data if isinstance(item, dict)], key=lambda x: len(x['text']), reverse=True)
            new_topic_data = []
            for i in range(len(topic_data)): # len(topic_data)
                text_to_find = topic_data[i]['text']
                new_start_pos = text.find(text_to_find)
                new_end_pos = new_start_pos + len(text_to_find)
                for j in range(len(new_topic_data)): # 검사하면서 new_pos들 업데이트해주는 부분
                    if(new_topic_data[j]['start_pos'] <= new_start_pos < new_topic_data[j]['end_pos']):
                        new_start_pos = text.find(text_to_find,new_end_pos)
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
                    cur_dict = {'original_text' : text[start_pos:end_pos], 'topic' : 'O', 'bbox' : []}
                    if(cur_dict['original_text'] == ' ' or cur_dict['original_text'] == '\n'):
                        continue
                    cur_dict_list.append(cur_dict)
                    cur_pos = end_pos
                elif(j % 2 == 0 and j == 2*len(new_topic_data)): # 태깅 안 되어있는 case 중 마지막 부분 분리
                    start_pos = cur_pos
                    end_pos = len(text)
                    if(start_pos == end_pos):
                        continue
                    cur_dict = {'original_text' : text[start_pos:end_pos], 'topic' : 'O', 'bbox' : []}
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
                    cur_dict = {'original_text' : text[start_pos:end_pos], 'topic' : new_topic_data[j//2]['topic'], 'bbox' : []}
                    

                    # start_idx_found --> 띄어쓰기 없앤 텍스트에서 찾은 시작 인덱스
                    start_idx_found = text_spacingx.find(cur_dict['original_text'].strip().replace(' ', '').replace('\n', ''))
                    # end_idx_found --> 띄어쓰기 없앤 텍스트에서 찾은 마지막 인덱스
                    end_idx_found = start_idx_found + len(cur_dict['original_text'].strip().replace(' ', '').replace('\n', ''))

                    
                    while (not(bbox_list[bbox_idx]['start_pos_spacingx'] <= start_idx_found <= (bbox_list[bbox_idx]['end_pos_spacingx'] - 1))):
                        bbox_idx += 1
                    bbox_start_idx = bbox_idx
                    while (not(bbox_list[bbox_idx]['start_pos_spacingx'] <= (end_idx_found - 1) <= (bbox_list[bbox_idx]['end_pos_spacingx'] - 1))):
                        bbox_idx += 1
                    bbox_end_idx = bbox_idx


                    for idx in range(bbox_start_idx, bbox_end_idx + 1):
                        cur_dict['bbox'].append(bbox_list[idx]['bbox'])
                    

                    

                    cur_dict_list.append(cur_dict)
                    cur_pos = end_pos
                    # print("처리 끝")

        else: # 태깅 안 되어있는 경우
            cur_dict = {'original_text' : text, 'topic' : 'O', 'bbox' : []}
            cur_dict_list.append(cur_dict)
        
        # print("n 루프 하나 끝!")
        return cur_dict_list


def preprocessing_ocr(args):
    sentence_dict_list = []
    directory = args.fp

    file_list = os.listdir(directory)

    file_list.sort()

    ocr_count = 0

    for filename in file_list:
        if filename.endswith('.json'):  # JSON 파일인지 확인
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8-sig') as file:
                ocr_list = json.load(file) # JSON 파일 불러오기
            for n in range(len(ocr_list)):
                ocr_count += 1
                text = ocr_list[n][0]
                topic_data = ocr_list[n][1]
                bbox_list = ocr_list[n][2:]
                cur_dict_list = concat_find_bbox(text, topic_data, bbox_list)
                for cur_dict in cur_dict_list:
                    cur_dict["Ocr #"] = "Ocr " + str(ocr_count)
                sentence_dict_list.extend(cur_dict_list)
                # sentence_dict_list.extend(concat_find_bbox(text, topic_data, bbox_list))
            print(f"{filename} Processed")
        

    split_sentence_dict_list = []


    for i in range(len(sentence_dict_list)): # 전체 텍스트에 대한 리스트 원소 개수
        for j in range(len(sentence_dict_list[i]['original_text'].split('\n'))): # 하나의 원본 텍스트에 대해서 분리한 문장의 개수
            if(sentence_dict_list[i]['original_text'].split('\n')[j].strip() != ''):
                split_sentence_dict_list.append({'original_text': sentence_dict_list[i]['original_text'].split('\n')[j],
                                                'topic': sentence_dict_list[i]['topic'],
                                                'bbox': sentence_dict_list[i]['bbox'],
                                                'Ocr #': sentence_dict_list[i]['Ocr #']})

    for i in range(len(split_sentence_dict_list)): # 분리한 텍스트의 전체 개수
        split_sentence_dict_list[i]['original_text'] = split_sentence_dict_list[i]['original_text'].strip()
        split_sentence_dict_list[i]['words'] = split_sentence_dict_list[i]['original_text'].split()


    final_sentence_dict_list = []
    sentence_count = 0
    word_count = 0

    # 태깅 부분
    for i in range(len(split_sentence_dict_list)):
        sentence_count += 1
        for j in range(len(split_sentence_dict_list[i]['words'])):
            word_count += 1
            if(split_sentence_dict_list[i]['topic'] != 'O' and j == 0):
                topic = 'B-'+split_sentence_dict_list[i]['topic']
            elif(split_sentence_dict_list[i]['topic'] != 'O' and j != 0):
                topic = 'I-'+split_sentence_dict_list[i]['topic']
            else:
                topic = split_sentence_dict_list[i]['topic']
            

            row_dict = {'Ocr #': split_sentence_dict_list[i]['Ocr #'],
                        'Sentence #': 'Sentence '+ str(sentence_count),
                        'Word': split_sentence_dict_list[i]['words'][j],
                        'Aspect': topic,
                        "Bbox": split_sentence_dict_list[i]['bbox']}
            final_sentence_dict_list.append(row_dict)

    print("단어 개수:", word_count)
    print("문장 개수:", sentence_count)
    # print("OCR 이미지 개수:", ocr_count)
    return final_sentence_dict_list



def json_2_csv(args):
    if not os.path.exists(args.save_p):
        os.makedirs(args.save_p)

    csv_file = os.path.join(args.save_p, 'output.csv')

    final_sent_dict_list = preprocessing_ocr(args)

    df = pd.DataFrame(final_sent_dict_list)
    # df['New_Sentence #'] = df.groupby('Ocr #').cumcount() + 1
    df['New_Sentence #'] = df.groupby('Ocr #')['Sentence #'].apply(lambda x: x.rank(method='dense').astype(int))

    df = df.drop(['Sentence #'], axis=1).reset_index(drop=True)
    df.rename(columns={'New_Sentence #': 'Sentence #'}, inplace=True)


    df['Ocr_Num'] = df['Ocr #'].apply(lambda x: int(re.findall(r'\d+', x)[0]))

    df = df.sort_values(by=['Ocr_Num', 'Sentence #'], ascending=[True, True])

    df = df.drop(['Ocr_Num'], axis=1).reset_index(drop=True)

    df['Sentence #'] = df['Sentence #'].astype(str)  # 정수를 문자열로 변환
    df['Sentence #'] = 'Sentence ' + df['Sentence #']  # 문자열 연결

    df = df[['Ocr #', 'Sentence #', 'Word', 'Aspect', 'Bbox']]

    df.to_csv(csv_file, encoding='utf-8-sig', index=False)
