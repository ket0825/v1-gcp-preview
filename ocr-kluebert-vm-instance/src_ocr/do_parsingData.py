# -*- coding: utf-8 -*-
# 학습용 데이터를 생성하기 위한 모듈
# 1. json 형식의 원본 데이터를 csv 형식의 데이터로 파싱 후,
# 2. train/test/valid용 데이터로 분할

import argparse
from data_manager.parsers.json2csv import json_2_csv
from data_manager.parsers.split_csv import file_split
import warnings
warnings.filterwarnings(action='ignore')

'''
    [args 내 argument list]
    - fp: 학습 데이터 형태로 파싱할 json 데이터의 경로 혹은 파일명
    - save_p: 파싱된 학습 데이터를 저장할 경로 (해당 경로가 존재하지 않을 시, 자동 생성됨)
    - encoding: 파싱 후 데이터를 저장할 때, 어떤 인코딩 형식으로 저장할지 설정 (default: utf-8-sig)
    - val_ratio: 원본 데이터에서 분할을 수행할 검증 데이터의 비율 (default: 0.1)
    - test_ratio: 원본 데이터에서 분할을 수행할 평가 데이터의 비율 (default: 0.1)
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parsing json files to csv')
    parser.add_argument('--fp', help='to parse Directory', default="./resources_ocr/data/")
    parser.add_argument('--save_p', help='Directory to save parsed Files', default="./resources_ocr/parsing_data/")
    parser.add_argument('--encoding', help='encode', default="utf-8-sig")
    parser.add_argument('--val_ratio', type=float, help='Directory to save parsed Files', default=0.1)
    parser.add_argument('--test_ratio', type=float, help='Directory to save parsed Files', default=0.1)
    parser.add_argument("--cmd", type=str, help="실행된 명령어", default="sh ./scripts_ocr/data/data_parsing.sh")
    args = parser.parse_args()
    
    print(f"[실행 명령어]: {args.cmd}")
    print(">> Starting Parsing json files to csv files")
    json_2_csv(args=args)
    print(">> file splitter start to making training/valid/test set")
    args.fp = args.save_p
    print(args.fp)
    file_split(args=args) # json_2_csv를 통해 생성된 파일을 학습/검증/테스트셋으로 분할