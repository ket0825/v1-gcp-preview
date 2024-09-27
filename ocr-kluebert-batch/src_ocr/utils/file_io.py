import os
import pandas as pd


# fp (디렉터리 경로) 내의 extension 확장자를 갖는 파일 list를 얻어옴
def get_file_list(fp, extension="csv"):
    filelist = []
    if os.path.isfile(fp):
        filelist.append(fp)
    elif os.path.isdir(fp):
        filelist = [os.path.join(fp, file) for file in os.listdir(fp) if extension in file]
    return filelist


def read_csv(data_path, encoding='utf-8'):
    df = pd.read_csv(data_path, encoding=encoding)
    return df

def read_json(data_path, encoding='utf-8-sig'):
    df = pd.read_json(data_path, encoding=encoding)
    return df