import os
import re
import json
import pandas as pd
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
    # "\U00002702-\U000027B0"  # 기타 기호
    # "\U000024C2-\U0001F251"  # 추가 기호 및 픽토그램      # 이거 2줄까지 하면 한글이 사라짐
    "]+", flags=re.UNICODE)
pattern5 = re.compile(r"([~,!%&-.]){2,}") # 특수문자는 동일한 문자 2개 이상이면 삭제

def preprocess_text(text):
    return text.replace('\n', ' ').strip()

def split_content_into_sentences(content):
    sentences = split_sentences(content)    
    return [preprocess_text(sent.strip()) + "." for sent in sentences if sent.strip()]

# def tag_review(sentences, topics, review_counter):
#     tagged_sentences = [{
#     'Review #': f"Review {review_counter}",
#     'Sentence #': f"Sentence {sent_num+1}", 
#     'Word': word, 
#     'Sentiment': 'O', 
#     'Aspect': 'O', 
#     'Sentiment_Score': 'O', 
#     'Aspect_Score': 'O',
#     } for sent_num, sent in enumerate(range(len(sentences))) for word in sent if word.strip()]
    
#     # 1. label 역순 정렬.
#     # 2. label이 먼저 태깅되어 있는 경우, 그것보다 더 뒤에서 확인.
#     # 3. 체크는 sentence 단위로. 단 비교시에는 .을 제외한 단어로 비교.
#     # 4. sentence 단위일 때, 띄어쓰기 없는 부분이 나눠지는 경우를 고려하여 label의 마지막 부분은 sentence의 마지막 단어 + 1까지 in으로 체크.
#     # 5. 첫번째 또한 짤리는 경우가 존재하므로 in으로 체크.

#     tagged_sentences_idx = 0
#     for topic in topics:
#         topic_text = preprocess_text(topic['text'])        
#         for topic_word in topic['topic'].split():
#             pass

def regexp_text(text):
    replaced_str = ' '    
    new_text = pattern1.sub(replaced_str, text)
    new_text = pattern2.sub(replaced_str, new_text)
    new_text = pattern3.sub(replaced_str, new_text)        
    new_text = emoticon_pattern.sub(replaced_str, new_text)
    new_text = pattern4.sub(replaced_str, new_text)
    new_text = pattern5.sub(replaced_str, new_text)
    new_text = new_text.replace('  ', ' ').strip()
    return new_text


def clean_data(our_topics):
    if not our_topics:
        return []
    
    cleansed_topics = []
    for topic in our_topics:
        if (not topic.get('text')
            or not topic.get("topic")
            or not topic.get("start_pos") != -1
            or not topic.get("end_pos")
            or not topic.get("positive_yn")
            or not topic.get("sentiment_scale")
            or not topic.get("topic_score")
            ):
            continue
        
        topic['text'] = regexp_text(topic['text']).strip() # 수정한 부분.
        # topic['text'] = topic['text'].strip() # regexp 처리 안한 부분.
        cleansed_topics.append(topic)
    
    return cleansed_topics


def process_json_file(file_path, output_csv_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)    
    
    rows = []
    review_counter = 1
    
    for item in data:
        if ('our_topics' not in item 
            or not item['our_topics'] 
            or 'content' not in item
            ):
            continue    
        
        our_topics = clean_data(item['our_topics']) # 이 안에 regexp 들어가 있음.
        if not our_topics:
            continue        

        # print(f"review_counter: {review_counter}")
        content = regexp_text(preprocess_text(item['content'])) # 이거 빼고도 해봐야 함.
        # content = preprocess_text(item['content']) # 이거 빼고도 해봐야 함.
        content = more_than_one_space.sub(" ", content)
        sentences = split_content_into_sentences(content)        
        

        sentence_dict_list = []        
        word_index = 0
        word_order = 0
        word_idx_to_sentences_mapping = {}
        sentence_idx = 0

        for sent_num, sent in enumerate(sentences):
            sent_start = content.find(sent[:-1], sentence_idx)

            if sent_start > 0:
                exist_space = content[sent_start-1] == " "
            elif sent_num == 0: # 첫 번째 문장인 경우 반드시 공백이 있다고 해야 아래에서 정상 작동.
                exist_space = True
            else:
                exist_space = False
            
            for word in sent.split():                                      
                sentence_dict = {
                    'Review #': f"Review {review_counter}",
                    'Sentence #': f"Sentence {sent_num+1}", 
                    'Word': word, 
                    'Sentiment': 'O', 
                    'Aspect': 'O', 
                    'Sentiment_Score': 'O', 
                    'Aspect_Score': 'O',
                    }                
                
                if not exist_space and exist_period: # 전 번째에서 실행됨.
                    word_index -= 1
                # mapping 만들어주는 곳.
                word_idx_to_sentences_mapping[word_index] = word_order        
                
                # For debugging        
                # print("CONTENT 부분:",content[word_index: content.find(' ', word_index+1) if content.find(' ', word_index+1) != -1 else None].replace(" ", "----"))     
                # if "----" in content[word_index: content.find(' ', word_index+1) if content.find(' ', word_index+1) != -1 else None]:
                #     print("stop")
                # print("SENTENCE_DICT_LIST 부분:",sentence_dict['Word'].replace(" ", "----").replace(".", ""))
                
                exist_period = word[-1] == "."
                if exist_space and exist_period:
                    word_index += len(word) # word에 .이 있으니 .을 제외하고 공백의 길이만큼 더해줌.
                elif exist_space and not exist_period:
                    word_index += len(word)+1
                elif not exist_space and exist_period:
                    # 이 부분은 문장분리가 되었는데, 단어가 띄어쓰기 없이 나눠진 경우에 해당됨.
                    # ~했습니다감사합니다. (상관없을 듯)
                    word_index += len(word)
                elif not exist_space and not exist_period:
                    # 이 부분은 문장분리가 되었는데, 단어가 띄어쓰기 없이 나눠진 경우에 문장분리 다음에서의 첫 단어인 경우에 해당됨.                    
                    word_index += len(word)+1

                word_order += 1
                sentence_dict_list.append(sentence_dict)
        # 0~5가 단어. sentence_dict_list의 0 index가 단어를 표현하는 거에요.
        # 7~10가 단어. sentence_dict_list의 1 index가 해당하는 단어인거에요.
        # 12~13가 단어. sentence_dict_list의 2 index가 해당하는 단어인거에요.

            sentence_idx = sent_start + len(sent) - 1 # 온점 제거.

        # .이 있다. 그러면 12~14까지가 단어지만 sentence_dict_list의 3 index가 해당하는 단어인거에요.

        # For debugging
        # for word, sentence_dict_list_idx in word_idx_to_sentences_mapping.items():
        #     content_word = content[word:content.find(' ', word+1) if content.find(' ', word+1) != -1 else None].replace(" ", "----") # ----는 없어야 하는거임. 일부로 알아보기 편하게 하려고 넣은 것.
        #     print(f"mapping key 부분 (content에서의 단어): {content_word}")
        #     if "----" in content_word:
        #         print("stop")
        #     # 위는 단어 단위로 출력하는 코드.
        #     sent_dict_list_word = sentence_dict_list[sentence_dict_list_idx]['Word'].replace(" ", "----").replace(".", "") # ----는 없어야 하는거임.
        #     print(f"mapping의 value 부분 (sentence_dict_list에서의 단어): {sent_dict_list_word}")
        #     if "----" in sent_dict_list_word:
        #         print("stop")
        #     if content_word != sent_dict_list_word:
        #         print("[MISMATCH]")
     
        sentence_words_concat = content                                              
        our_topics = sorted(our_topics, key=lambda x: len(x['text']), reverse=True)        
        checked_indice_set = set()
        for topic in our_topics:
            try:
                topic_text = more_than_one_space.sub(" ", preprocess_text(topic['text']))
                    
                start_idx = sentence_words_concat.find(topic_text) 
                # 공백이 없는 경우도 존재하기에 x. 문제는 태깅 자체가 ~번에 라고 되어 있는 경우도 존재함.
                # start_candidate_idx = sentence_words_concat.find(topic_text)
                # start_idx = sentence_words_concat[:start_candidate_idx].rfind(" ")+1 if start_candidate_idx != -1 else -1 # 가장 가까운 공백 찾기.
                # 이미 체크한 인덱스인지 확인
                while start_idx != -1 and start_idx in checked_indice_set:
                    # 체크한 인덱스라면 다음 인덱스부터 찾기
                    start_idx = sentence_words_concat.find(topic_text, start_idx+1)
                    # start_candidate_idx = sentence_words_concat.find(topic_text, start_candidate_idx+1)
                    # start_idx = sentence_words_concat[:start_candidate_idx].rfind(" ")+1 if start_candidate_idx != -1 else -1
                # 토픽이 발견되지 않았을 때 (이 경우 에러임)
                if start_idx == -1:
                    # raise ValueError(f"Topic '{topic_text}' not found in review {review_counter}")            
                    print(f"Topic '{topic_text}' not found or duplicate in review {review_counter} at {file_path}")
                    continue
                
                # 처음 발견된 토픽이었을 때
                # text length
                topic_text_len = len(topic_text)
                end_idx = start_idx + topic_text_len - len(topic_text[topic_text.rfind(" ")+1:]) # 마지막 단어의 시작 인덱스 찾기
                # if start_candidate_idx == start_idx:
                #     end_idx = start_idx + topic_text_len - len(topic_text[topic_text.rfind(" ")+1:]) # 마지막 단어의 시작 인덱스 찾기
                # else:
                #     end_idx = start_candidate_idx + topic_text_len - len(topic_text[topic_text.rfind(" ")+1:])                                               
                               
                # 체크 인덱스에 추가
                for i in range(start_idx, start_idx+len(topic_text)):
                    checked_indice_set.add(i)

                # human error (덜 태깅한 부분 잡는 부분)    
                try:                    
                    start_word_idx = word_idx_to_sentences_mapping[start_idx] 
                except:                    
                    start_idx = sentence_words_concat[:start_idx].rfind(" ")+1 if start_idx != -1 else -1
                    start_word_idx = word_idx_to_sentences_mapping[start_idx]
                # human error (덜 태깅한 부분 잡는 부분)
                try:
                    end_word_idx = word_idx_to_sentences_mapping[end_idx]
                except:
                    end_idx = sentence_words_concat[:end_idx].rfind(" ")+1 if end_idx != -1 else -1
                    end_word_idx = word_idx_to_sentences_mapping[end_idx]

                sentence_dict_list[start_word_idx]['Sentiment'] = f"{'B-긍정' if topic['positive_yn'] == 'Y' else 'B-부정'}"
                sentence_dict_list[start_word_idx]['Aspect'] = f"B-{topic['topic']}"
                sentence_dict_list[start_word_idx]['Sentiment_Score'] = f"B-{topic['sentiment_scale']}"
                sentence_dict_list[start_word_idx]['Aspect_Score'] = f"B-{topic['topic_score']}"                
                for word_idx in range(start_word_idx+1, end_word_idx+1):
                    sentence_dict_list[word_idx]['Sentiment'] = f"{'I-긍정' if topic['positive_yn'] == 'Y' else 'I-부정'}"
                    sentence_dict_list[word_idx]['Aspect'] = f"I-{topic['topic']}"
                    sentence_dict_list[word_idx]['Sentiment_Score'] = f"I-{topic['sentiment_scale']}"
                    sentence_dict_list[word_idx]['Aspect_Score'] = f"I-{topic['topic_score']}"
            except Exception as e:
                print(f"Error in review {review_counter} at {file_path}: error message: 보통 KeyError: {e}")
                print(f"topic_text: {topic_text}")
                print(f"start_idx: {start_idx}, end_idx: {end_idx},\
                      start_word_idx: {start_word_idx}, end_word_idx: {end_word_idx}")
                content[start_idx:end_idx]
                f_idx, s_idx = sorted(word_idx_to_sentences_mapping.keys(), key=lambda x: abs(x-start_idx))[0:2]
                print(f"[START INDEX 근처] First word: {sentence_dict_list[word_idx_to_sentences_mapping[f_idx]]}, Last index: {sentence_dict_list[word_idx_to_sentences_mapping[s_idx]]}")
                f_idx, s_idx = sorted(word_idx_to_sentences_mapping.keys(), key=lambda x: abs(x-end_idx))[0:2]
                print(f"[END INDEX 근처] First word: {sentence_dict_list[word_idx_to_sentences_mapping[f_idx]]}, Last index: {sentence_dict_list[word_idx_to_sentences_mapping[s_idx]]}")
     
        rows.extend(sentence_dict_list)
        review_counter += 1
     
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"Processed {file_path} and saved as {output_csv_path}")    
    else:
        print(f"Skipping {file_path} due to no valid tagging data")

    return
    
def process_json_files_in_folder(folder_path, output_folder):
    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # 폴더 내의 모든 JSON 파일 처리    
    for filename in os.listdir(folder_path):                
        if filename.endswith(".json"):
            json_file_path = os.path.join(folder_path, filename)
            output_csv_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.csv")
            
            # JSON 파일 처리 및 CSV 파일 생성
            
            process_json_file(json_file_path, output_csv_path)
            

def json_2_csv(args):        
    process_json_files_in_folder(args.fp, args.save_p)
    