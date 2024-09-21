# -*- coding: utf-8 -*-
# 감정 별로 형태소 태깅을 통한 형용사 및 동사 top10 어휘 추출
# 사용된 형태소 분석기: kma-black
import os
from utils.set_logger import Log
from wisekmapy.wisekma import Wisekma
import datetime
import argparse
import pandas as pd
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')


# res(형태소 분석을 수행한 문장) 내에 형태소 태그가 형용사/동사인 word를 추출하여 반환하는 함수
# res 예시 >> [('롱테이크', 'NNG'), ('로', 'JKB'), ('이루어지', 'VV'), ('ㄴ', 'ETM'), ('영화', 'NNG')]
# VV: 동사 / VA: 형용사
def get_verbs_adjs(res):
    verb_list, adj_list = [], []
    for x in res:
        if "/VV" not in x and "/VA" not in x:
            continue
        else:
            if "/VV" in x:
                verb_list.append(x)
            else:
                adj_list.append(x)
    return Counter(verb_list), Counter(adj_list)

'''
    [args 내 argument list]
    - fp: 형태소 분석을 수행할 데이터의 경로 혹은 파일명 (단, 모델 학습용으로 파싱된 csv 형식의 데이터일 것)
    - encoding: 형태소 분석을 수행할 데이터의 인코딩 형식 (default: utf-8-sig)
    - log_fp: 로그를 저장할 경로, 존재하지 않을 시 생성됨 (default: './logs')
    - log_filename : 저장될 로그의 파일 명 (default: 실행시점 + _pos_analysis.log)
'''
def pos_tagging(args, extension=".csv"):
    base_feature = "Aspect"
    nowDate = datetime.datetime.now()
    log_filename = nowDate.strftime("%Y-%m-%d_%H-%M-%S_") + args.log_filename

    # 로그를 저장할 디렉토리가 없다면 생성
    if not os.path.exists(args.log_fp):
        os.makedirs(args.log_fp)

    # logger 생성
    logger = Log()
    log = logger.set_log(log_path=args.log_fp, filename=log_filename)

    log.info(f"[실행 명령어]: {args.cmd}")
    log.info("[Config]")
    log.info(f"{args.__dict__}")

    # 사용할 형태소 분석기를 정의 (Wisekma-black)
    log.info(">> Start to set pos Tagger")
    tok = Wisekma()
    log.info(">> pos Tagger setting Success!")

    # 형태소 분석을 수행할 데이터의 리스트를 생성
    # fp argument에 해당하는 경로 내 csv file들로 구성
    if os.path.isdir(args.fp):
        file_list = [os.path.join(args.fp, file) for file in os.listdir(args.fp) if extension in file]
    else:
        file_list = [args.fp]

    # 만약 file_list가 비어있다면, 분석을 수행할 데이터가 없음을 뜻하기 때문에 오류를 발생시킴
    if len(file_list) == 0:
        log.error("File List is empty")
        raise FileExistsError()

    # 감정 별, 형태소/동사 분포를 저장하기 위한 dictionary를 initializing
    verb_dict = {}
    adj_dict = {}

    verb_dict["O"] = Counter()
    adj_dict["O"] = Counter()

    for now_fp in tqdm(file_list):
        df = pd.read_csv(now_fp, encoding=args.encoding)
        df[base_feature] = df[base_feature].str.split("-").str[-1]
        group_words = df.groupby(base_feature)["Word"].apply(list).to_dict()

        for key in group_words.keys():
            if key in verb_dict:
                continue
            else:
                verb_dict[key] = Counter()
                adj_dict[key] = Counter()

        for feature in group_words.keys():
            for word in group_words[feature]:
                res = tok.pos(word, join=True)
                now_verb_dict, now_adj_dict = get_verbs_adjs(res)
                verb_dict[feature] += now_verb_dict
                adj_dict[feature] += now_adj_dict

    for feature in verb_dict.keys():
        log.info(f"***************[{feature}]***************")
        log.info("[형용사 상위 분포 Top 10]")
        log.info(adj_dict[feature].most_common(10))
        log.info("[동사 상위 분포 Top 10]")
        log.info(verb_dict[feature].most_common(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp", type=str, help="형태소 분석을 수행할 데이터의 경로, 데이터는 모델 학습용으로 파싱 된 데이터 (csv) 일 것")
    parser.add_argument("--encoding", type=str, default='utf-8-sig', help="데이터 encoding type")
    parser.add_argument("--log_fp", type=str, default='./logs/', help="로그를 저장할 경로")
    parser.add_argument("--log_filename", type=str, default='pos_analysis.log', help="저장될 로그 파일명 "
                                                                                     "- default: 실행시점 +"
                                                                                     " pos_analysis.log")
    parser.add_argument("--cmd", type=str, help="실행된 명령어", default="sh ./scripts/data/pos_analysis.sh")
    args = parser.parse_args()

    pos_tagging(args=args) # 형태소 분석 함수 호출