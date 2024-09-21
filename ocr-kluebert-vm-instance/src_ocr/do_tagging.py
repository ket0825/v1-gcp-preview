# -*- coding: utf-8 -*-
from data_manager.loaders.loader import set_loader
from modeling.model import ABSAModel
from modeling.trainer import tag_valid_fn, valid_eval_fn, tag_fn, preprocess_fn, preprocess_fn2 ### 함수import 수정
from utils.model_utils import device_setting, load_model
from utils.set_logger import Log
import os
import joblib
import argparse
import datetime
import warnings
import transformers
warnings.filterwarnings(action='ignore')



def tag_valid(config):
    valid_fp = config.valid_fp
    log_name = "tagging_valid" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    logger = Log()
    log = logger.set_log(config.base_path, filename=log_name, level="DEBUG")

    model_path = os.path.join(config.base_path, config.out_model_path)
    log.info(f"[실행 명령어]: {config.cmd}")
    log.info("[Config]")
    log.info(f'{config.__dict__}')

    log.info('>>>>>>> Now Loading Sentiment&Aspect Category Encoder')
    meta_data = joblib.load(os.path.join(config.base_path, config.label_info_file))
    enc_aspect, enc_aspect2 = meta_data["enc_aspect"], meta_data["enc_aspect2"]


    num_aspect, num_aspect2 = len(list(meta_data["enc_aspect"].classes_)), len(list(meta_data["enc_aspect2"].classes_))


    # Device Setting (GPU/CPU)
    device = device_setting(log)
    
    start_time_creating_architecture = datetime.datetime.now()
    # model Architecture 생성
    model = ABSAModel(num_aspect=num_aspect, num_aspect2=num_aspect2, config=config,
                      need_birnn=bool(config.need_birnn))
    end_time_creating_architecture = datetime.datetime.now()

    print(f"모델 아키텍쳐 생성 시간: {end_time_creating_architecture - start_time_creating_architecture}")
    

    start_time_loading_model = datetime.datetime.now()
    # 저장된 모델 load
    model = load_model(model=model, state_dict_path=model_path, device=device)
    end_time_loading_model = datetime.datetime.now()

    print(f"저장된 모델 loading 시간: {end_time_loading_model - start_time_loading_model}")

    tokenizer = transformers.BertTokenizer.from_pretrained(config.init_model_path, do_lower_case=False)
    # 태깅 수행
    tag_valid_fn(config, tokenizer, model, enc_aspect, enc_aspect2, device, log)




def valid_eval(config):
    valid_fp = config.valid_fp
    log_name = "valid_eval" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    logger = Log()
    log = logger.set_log(config.base_path, filename=log_name, level="DEBUG")

    log.info(f"[실행 명령어]: {config.cmd}")
    log.info("[Config]")
    log.info(f'{config.__dict__}')

    valid_eval_fn(config)



def tag(config):
    tagging_fp = config.tagging_fp
    log_name = "tagging_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    logger = Log()
    log = logger.set_log(config.base_path, filename=log_name, level="DEBUG")

    model_path = os.path.join(config.base_path, config.out_model_path)
    log.info(f"[실행 명령어]: {config.cmd}")
    log.info("[Config]")
    log.info(f'{config.__dict__}')

    log.info('>>>>>>> Now Loading Sentiment&Aspect Category Encoder')
    meta_data = joblib.load(os.path.join(config.base_path, config.label_info_file))
    enc_aspect, enc_aspect2 = meta_data["enc_aspect"], meta_data["enc_aspect2"] ### enc_aspect2 추가함

    num_aspect, num_aspect2 = len(list(meta_data["enc_aspect"].classes_)), len(list(meta_data["enc_aspect2"].classes_))
   
    # Device Setting (GPU/CPU)
    device = device_setting(log)
    
    start_time_creating_architecture = datetime.datetime.now()
    # model Architecture 생성
    model = ABSAModel(num_aspect=num_aspect, num_aspect2=num_aspect2, config=config,
                      need_birnn=bool(config.need_birnn)) ### num_score 추가
    
    end_time_creating_architecture = datetime.datetime.now()

    print(f"모델 아키텍쳐 생성 시간: {end_time_creating_architecture - start_time_creating_architecture}")
    

    start_time_loading_model = datetime.datetime.now()
    # 저장된 모델 load
    model = load_model(model=model, state_dict_path=model_path, device=device)
    end_time_loading_model = datetime.datetime.now()

    print(f"저장된 모델 loading 시간: {end_time_loading_model - start_time_loading_model}")

    tokenizer = transformers.BertTokenizer.from_pretrained(config.init_model_path, do_lower_case=False)
    # 태깅 수행
    tag_fn(config, tokenizer, model, enc_aspect, enc_aspect2, device, log)



def preprocess(config):
    log_name = "preprocessing_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    logger = Log()
    log = logger.set_log(config.base_path, filename=log_name, level="DEBUG")

    model_path = os.path.join(config.base_path, config.out_model_path)
    log.info(f"[실행 명령어]: {config.cmd}")
    log.info("[Config]")
    log.info(f'{config.__dict__}')

    # Device Setting (GPU/CPU)
    device = device_setting(log)

    preprocess_fn(config)
    # preprocess_fn2(config)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_batch_size", type=int, default=1, help="한 batch에 속할 테스트 데이터 샘플의 size")
    ##### help 멘트 수정 필요 / batch size도 1로 바꾸고 돌린거면 고쳐야 할 듯? / batch size에 대해서 잘 봐야지 역변환할 수 있을 거 같음
    # ============= [Model Parameter] =================
    # 평가를 수행할 모델을 생성할 때, 사용 했던 parameter와 동일하지 않을 시, 오류 발생!
    parser.add_argument("--init_model_path", type=str, default="klue/bert-base", help="사용된 BERT의 종류")
    parser.add_argument("--max_length", type=int, default=512, help="토큰화된 문장의 최대 길이(bert는 기본 512)")
    parser.add_argument("--need_birnn", type=int, default=0, help="model에 Birnn Layer를 추가했는지 여부 (True: 1/False: 0)")
    parser.add_argument("--sentiment_drop_ratio", type=float, default=0.3,
                        help="Sentiment 속성의 과적합 방지를 위해 dropout을 수행한 비율")
    parser.add_argument("--aspect_drop_ratio", type=float, default=0.3,
                        help="Aspect Category 속성의 과적합 방지를 위해 dropout을 수행한 비율")
    parser.add_argument("--sentiment_in_feature", type=int, default=768,
                        help="각 Sentiment input sample의 size")
    parser.add_argument("--aspect_in_feature", type=int, default=768,
                        help="각 Aspect Category input sample의 size")
    # ===================================================
    parser.add_argument("--json_fp", type=str, help="json 구조 바꿔야하는 데이터들이 포함된 디렉토리 경로 or 전처리할 데이터 파일 경로 설정", default="./resources_ocr/data_json/")

    parser.add_argument("--preprocessing_fp", type=str, help="전처리할 데이터들이 포함된 디렉토리 경로 or 전처리할 데이터 파일 경로 설정", default="./resources_ocr/data_json_structure/")
    ### path 수정
    parser.add_argument("--tagging_fp", type=str, help="태깅할 데이터들이 포함된 디렉토리 경로 or 태깅할 데이터 파일 경로 설정", default="./resources_ocr/preprocessed_results_json/")
    parser.add_argument("--valid_fp", type=str, help="valid 데이터 포함된 디렉토리 경로 or valid 데이터 파일 경로 설정", default="./resources_ocr/parsing_data/valid/")
    parser.add_argument("--base_path", type=str, help="평가를 수행할 Model과 Encoder가 저장된 경로", default="./ckpt_ocr/model/")
    ### 경로 수정


    parser.add_argument("--need_preprocessing", type=int, default=1, help="json 데이터에 대해서 전처리를 진행해야 하는지 여부 (True: 1/False: 0)")
    # 전처리가 되어 있다는 것은 concat_find_bbox를 통해 bbox를 찾은 상태라는 의미도 된다--> need_preprocessing이 0이면 bbox 안 찾아도 된다

    parser.add_argument("--label_info_file", type=str, help="사용할 Encoder 파일명", default="meta.bin")
    parser.add_argument("--out_model_path", type=str, help="평가할 model의 파일명", default="pytorch_model.bin")
    parser.add_argument("--print_sample", type=int, default=1, help="각 sample의 예측 결과 출력 여부를 결정 (True: 1/False: 0)")
    parser.add_argument("--cmd", type=str, help="실행된 명령어", default="sh ./scripts/model/do_eval.sh")
    ##### 나중에 sh 만들고 수정 필요
    config = parser.parse_args()

    # valid.csv에 대해서 모델이 예측한 결과를 csv로 저장
    # tag_valid(config)

    # tag_valid에서 나온 csv 파일과 원본 valid.csv 파일을 비교해서 성능 측정
    # valid_eval(config)
    
    # 전처리하는 함수
    preprocess(config)

    # json 파일들에 대해서 모델이 예측한 결과를 json으로 저장하는 함수
    tag(config)
    

