# -*- coding: utf-8 -*-
from data_manager.loaders.loader import set_loader
from modeling.model import ABSAModel
from modeling.trainer import eval_fn
from utils.model_utils import device_setting, load_model
from utils.set_logger import Log
import os
import joblib
import argparse
import datetime
import warnings
warnings.filterwarnings(action='ignore')


def eval(config):
    eval_fp = config.eval_fp
    log_name = "eval_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    logger = Log()
    log = logger.set_log(config.base_path, filename=log_name, level="DEBUG")

    model_path = os.path.join(config.base_path, config.out_model_path)
    log.info(f"[실행 명령어]: {config.cmd}")
    log.info("[Config]")
    log.info(f'{config.__dict__}')

    log.info('>>>>>>> Now Loading Aspect Category Encoder')
    meta_data = joblib.load(os.path.join(config.base_path, config.label_info_file))
    enc_aspect = meta_data["enc_aspect2"]
    num_aspect, num_aspect2 = len(list(meta_data["enc_aspect"].classes_)), len(list(meta_data["enc_aspect2"].classes_))

    # Device Setting (GPU/CPU)
    device = device_setting(log)

    # model Architecture 생성
    model = ABSAModel(num_aspect=num_aspect, num_aspect2=num_aspect2, config=config,
                      need_birnn=bool(config.need_birnn))
    # 저장된 모델 load
    model = load_model(model=model, state_dict_path=model_path, device=device)

    # Eval DataLoader setting
    eval_data_loader = set_loader(config=config, fp=eval_fp, batch_size=config.eval_batch_size, meta_data=meta_data)

    # 각 sample의 예측 결과 출력 여부
    flag = "eval" if config.print_sample == 1 else "valid"

    # 평가 수행
    eval_fn(eval_data_loader, model, enc_aspect, device, log, f1_mode='micro', flag=flag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_batch_size", type=int, default=4, help="한 batch에 속할 테스트 데이터 샘플의 size")
    # ============= [Model Parameter] =================
    # 평가를 수행할 모델을 생성할 때, 사용 했던 parameter와 동일하지 않을 시, 오류 발생!
    parser.add_argument("--init_model_path", type=str, default="klue/bert-base", help="사용된 BERT의 종류")
    parser.add_argument("--max_length", type=int, default=512, help="토큰화된 문장의 최대 길이(bert는 기본 512)")
    parser.add_argument("--need_birnn", type=int, default=0, help="model에 Birnn Layer를 추가했는지 여부 (True: 1/False: 0)")
    parser.add_argument("--aspect_drop_ratio", type=float, default=0.3,
                        help="Aspect Category 속성의 과적합 방지를 위해 dropout을 수행한 비율")
    parser.add_argument("--aspect_in_feature", type=int, default=768,
                        help="각 Aspect Category input sample의 size")
    # ===================================================
    parser.add_argument("--eval_fp", type=str, help="평가 데이터들이 포함된 디렉토리 경로 or 평가 데이터 파일 경로 설정")
    parser.add_argument("--base_path", type=str, help="평가를 수행할 Model과 Encoder가 저장된 경로", default="./ckpt/model/")
    parser.add_argument("--label_info_file", type=str, help="사용할 Encoder 파일명", default="meta.bin")
    parser.add_argument("--out_model_path", type=str, help="평가할 model의 파일명", default="pytorch_model.bin")
    parser.add_argument("--print_sample", type=int, default=1, help="각 sample의 예측 결과 출력 여부를 결정 (True: 1/False: 0)")
    parser.add_argument("--cmd", type=str, help="실행된 명령어", default="sh ./scripts/model/do_eval.sh")
    config = parser.parse_args()

    eval(config)