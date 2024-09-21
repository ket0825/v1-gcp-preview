import numpy as np
import torch
from collections import OrderedDict

# 과적합 방지를 위한 조기 학습 종료 클래스
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# 사용가능한 Device를 setting (GPU/CPU)
def device_setting(log):
    log.info('>>>>>>> Device Setting')
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info('Now using GPU')
    else:
        device = torch.device("cpu")
        log.info('Now using CPU')
    return device


# 모델에 저장된 state dict를 탑재
def load_model(model, state_dict_path, device):
    current_model_dict = model.state_dict()
    loaded_state_dict = torch.load(state_dict_path, map_location=device)
    
    new_state_dict = OrderedDict()
    # 현재 모델의 키 순서대로 반복
    for k in current_model_dict.keys():
        # 로드된 상태 사전에 키가 있고 크기가 일치하는 경우
        if k in loaded_state_dict and loaded_state_dict[k].size() == current_model_dict[k].size():
            new_state_dict[k] = loaded_state_dict[k]
        else:
            new_state_dict[k] = current_model_dict[k]

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)

    return model
