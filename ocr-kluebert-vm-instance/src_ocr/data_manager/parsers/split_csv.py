import pandas as pd
import os
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit


def print_dist(d, flag):
    print(d.head())
    asp = pd.DataFrame([dict(Counter(d["Aspect"]))]).T
    print("\n" + '*' * 50)
    print("About {example}\t".format(example=flag))
    print("<속성 레이블 분포>")
    print(asp)


def file_split(args):
    # 파싱할 파일 list 구성
    file_list = [os.path.join(args.fp, file) for file in os.listdir(args.fp) if "csv" in file]
    # 파일을 저장할 경로가 없다면 생성
    if not os.path.exists(args.save_p):
        os.makedirs(args.save_p)

    # 저장 경로 내, 학습/검증/테스트 디렉터리 생성
    save_dir_name = ["train", "valid", "test"]
    for p in save_dir_name:
        if not os.path.exists(os.path.join(args.save_p, p)):
            os.makedirs(os.path.join(args.save_p, p))
    print(file_list)
    
    
    df = pd.read_csv(file_list[0], encoding=args.encoding)
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill") # 결측치 fill
    df.loc[:, "Ocr #"] = df["Ocr #"].fillna(method="ffill") # 결측치 fill



    # Sentnece ID를 기준으로 group화하여 test set 랜덤 추출
    test_split = GroupShuffleSplit(test_size=args.test_ratio, n_splits=1,
                                       random_state=42).split(df, groups=df['Ocr #'])
    train_val_idxs, test_idxs = next(test_split)

    train_val = df.iloc[train_val_idxs]
    test = df.iloc[test_idxs]

        # Sentnece ID를 기준으로 group화하여 validation set 랜덤 추출
    val_split = GroupShuffleSplit(test_size=args.val_ratio,
                                      n_splits=1, random_state=42).split(train_val, groups=train_val['Ocr #'])
    train_idxs, val_idxs = next(val_split)
    train = train_val.iloc[train_idxs]
    val = train_val.iloc[val_idxs]

        # 결과 파일 저장
    train.to_csv(os.path.join(args.save_p, save_dir_name[0], save_dir_name[0] + ".csv"), encoding=args.encoding, index=False)
    val.to_csv(os.path.join(args.save_p, save_dir_name[1], save_dir_name[1] + ".csv"), encoding=args.encoding, index=False)
    test.to_csv(os.path.join(args.save_p, save_dir_name[2], save_dir_name[2] + ".csv"), encoding=args.encoding, index=False)











