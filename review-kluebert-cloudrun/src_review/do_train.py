# -*- coding: utf-8 -*-
from data_manager.loaders.loader import set_loader
from data_manager.dataset.absa import Encoder
from modeling.model import ABSAModel
from modeling.trainer import train_fn, eval_fn
from utils.model_utils import EarlyStopping, device_setting
from utils.set_logger import Log
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import os
import argparse
import warnings
warnings.filterwarnings(action='ignore')

def train(config):
    # 생성된 모델 및 Encoder / log 등을 저장할 base 경로의 존재 유무 체킹 - 없다면 새로 생성
    if not os.path.exists(config.base_path):
        os.makedirs(config.base_path)
    # 학습에 사용할 학습 데이터 경로 및 검증 데이터 경로를 세팅
    train_fp, valid_fp = config.train_fp, config.valid_fp
    # 모델이 저장될 경로를 세팅
    save_model_path = os.path.join(config.base_path, config.out_model_path)

    # Set UP Log
    logger = Log()
    log = logger.set_log(config.base_path, level="DEBUG")
    log.info(f"[실행 명령어]: {args.cmd}")
    log.info("[Config]")
    log.info(f'{config.__dict__}')

    # 학습을 위한 Device (GPU / CPU) setting
    device = device_setting(log)

    # 감정 및 속성 카테고리를 Encoding할 Encoder를 세팅 (LabelEncoder)
    log.info('>>>>>>> Now setting Sentiment&Aspect Category Encoder')
    enc = Encoder(config=config, fp=train_fp)

    # enc_aspect >> 일반 속성 카테고리를 위한 Encoder
    # enc_aspect2 >> 대분류 속성 카테고리를 위한 Encoder
    # enc_sentiment >> 감성 속성을 위한 Encoder
    # enc_sentiment_score >> 감성 스코어를 위한 Encoder
    # enc_aspect_score >> 속성 스코어를 위한 Encoder
    enc_aspect, enc_aspect2, enc_sentiment, enc_aspect_score, enc_sentiment_score = enc.get_encoder()
    meta_data = {
        "enc_aspect": enc_aspect,
        "enc_aspect2": enc_aspect2,
        "enc_sentiment": enc_sentiment,
        "enc_aspect_score": enc_aspect_score,
        "enc_sentiment_score": enc_sentiment_score,
    }
    # Encoder에 fitting 된 class의 수를 get
    num_aspect = len(list(enc_aspect.classes_))
    num_aspect2 = len(list(enc_aspect2.classes_))
    num_sentiment = len(list(enc_sentiment.classes_))
    num_aspect_score = len(list(enc_aspect_score.classes_))
    num_sentiment_score = len(list(enc_sentiment_score.classes_))

    # classes 확인
    log.info('>>>>>>> Now checking classes')
    log.info(f'enc_aspect.classes_: {enc_aspect.classes_}')
    log.info(f'enc_aspect2.classes_: {enc_aspect2.classes_}')
    log.info(f'enc_sentiment.classes_: {enc_sentiment.classes_}')
    log.info(f'enc_aspect_score.classes_: {enc_aspect_score.classes_}')
    log.info(f'enc_sentiment_score.classes_: {enc_sentiment_score.classes_}')
    

    log.info('>>>>>>> Now setting train/valid DataLoaders')
    train_data_loader = set_loader(fp=train_fp, config=config, meta_data=meta_data, batch_size=config.train_batch_size)
    valid_data_loader = set_loader(fp=valid_fp, config=config, meta_data=meta_data, batch_size=config.valid_batch_size)

    log.info('>>>>>>> Now setting Model Architecture')
    model = ABSAModel(
        config=config,
        num_sentiment=num_sentiment,
        num_aspect=num_aspect,
        num_aspect2=num_aspect2,
        num_sentiment_score=num_sentiment_score,
        num_aspect_score=num_aspect_score,
        need_birnn=bool(config.need_birnn)
    )
    model.to(device)

    # Optimzier Setting
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(
                    nd in n for nd in no_decay
                )
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay
                )
            ],
            "weight_decay": 0.0,
        },
    ]

    # train steps calculation
    num_train_steps = int(
        train_data_loader.dataset.get_length() / config.train_batch_size * config.epochs
    )
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # Early Stopper 세팅
    early_stopper = EarlyStopping(patience=config.stop_patience, verbose=True, path=save_model_path)
    log.info('>>>>>>> Training Start!')
    for epoch in range(config.epochs):
        log.info(f'[Now Epoch: {epoch}]')
        # Training
        train_loss = train_fn(
            train_data_loader,
            model,
            optimizer,
            device,
            scheduler
        )
        test_loss = eval_fn(
            valid_data_loader,
            model,
            enc_sentiment,
            enc_aspect,
            enc_aspect2,
            enc_sentiment_score,
            enc_aspect_score,
            device,
            log
        )

        log.info(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        # [early stopping 여부를 체크하는 부분]
        early_stopper(test_loss[0], model)  # 첫 번째 손실 값만 사용하여 현재 과적합 상황 추적
        if early_stopper.early_stop:  # 조건 만족 시 조기 종료
            log.info("EarlyStopping!!!!")
            break
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="데이터셋을 학습할 횟수")
    parser.add_argument("--train_batch_size", type=int, default=4, help="한 batch에 속할 학습 데이터 샘플의 size")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="한 batch에 속할 검증 데이터 샘플의 size")
    parser.add_argument("--init_model_path", type=str, default="klue/bert-base", help="사용할 BERT의 종류")
    parser.add_argument("--max_length", type=int, default=512, help="토큰화된 문장의 최대 길이를 설정 (bert는 기본 512)")
    parser.add_argument("--need_birnn", type=int, default=1, help="model에 Birnn Layer를 추가 여부 (True: 1/False: 0)")
    parser.add_argument("--sentiment_drop_ratio", type=float, default=0.3,
                        help="Sentiment 속성의 과적합 방지를 위해 dropout을 수행할 비율")
    parser.add_argument("--aspect_drop_ratio", type=float, default=0.3,
                        help="Aspect Category 속성의 과적합 방지를 위해 dropout을 수행할 비율")
    parser.add_argument("--sentiment_in_feature", type=int, default=768,
                        help="각 Sentiment input sample의 size")
    parser.add_argument("--aspect_in_feature", type=int, default=768,
                        help="각 Aspect Category input sample의 size")
    parser.add_argument("--stop_patience", type=int, default=3, help="validation loss를 기준으로 성능이 증가하지 않는 "
                                                                     "epoch을 몇 번이나 허용할 것인지 설정")
    parser.add_argument("--train_fp", type=str, default="./KoElectra_review_tagging/resources_review/parsing_data/train/", help="학습 데이터들이 포함된 디렉토리 경로 or 학습 데이터 파일 경로 설정")
    parser.add_argument("--valid_fp", type=str, default="./KoElectra_review_tagging/resources_review/parsing_data/valid/", help="검증 데이터들이 포함된 디렉토리 경로 or 검증 데이터 파일 경로 설정")
    parser.add_argument("--base_path", type=str, help="Model이나 Encoder를 저장할 경로 설정", default="./ckpt_review/model/")
    parser.add_argument("--label_info_file", type=str, help="Encoder의 저장 파일명", default="meta.bin")
    parser.add_argument("--out_model_path", type=str, help="model의 저장 파일명", default="pytorch_model.bin")
    parser.add_argument("--cmd", type=str, help="실행된 명령어", default="sh ./scripts_review/model/do_train.sh")
    args = parser.parse_args()
    
    train(config=args)