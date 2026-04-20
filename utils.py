import random
from collections import defaultdict

import numpy as np
import torch

# CIFAR-10 클래스 이름
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def deprocess_image(x, mean, std):
    # mean, std: train.py에서 계산한 값을 인자로 받음
    x = x.clone().detach().cpu().numpy()
    x = x.transpose(1, 2, 0)  # CHW → HWC
    x = x * std + mean
    x = np.clip(x, 0, 1)
    return x

def decode_label(pred_idx):
    # 원본은 ImageNet 1000클래스 → CIFAR-10 10클래스로 교체
    return CIFAR10_CLASSES[pred_idx]


def normalize(x):
    # 원본은 K.sqrt 등 Keras 백엔드 사용 → numpy로 교체
    return x / (np.sqrt(np.mean(np.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    # 변경 없음 (numpy 연산이라 그대로 사용 가능)
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
              start_point[1]:start_point[1] + rect_shape[1]] = \
        gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                  start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    # 변경 없음
    new_grads = np.ones_like(gradients)
    grad_mean = 1e4 * np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(10, 10)):
    # 변경 없음
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]),
        random.randint(0, gradients.shape[2] - rect_shape[1])
    )
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                          start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
                      start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


def init_coverage_tables(model1, model2):
    # 원본은 모델 3개 고정 → 2개로 변경
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    return model_layer_dict1, model_layer_dict2


def init_dict(model, model_layer_dict):
    # 원본은 Keras model.layers 순회 → PyTorch named_modules로 교체
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
            # 뉴런 수 = 출력 채널 수
            if hasattr(module, 'out_channels'):
                num_neurons = module.out_channels
            elif hasattr(module, 'out_features'):
                num_neurons = module.out_features
            else:
                continue
            for index in range(num_neurons):
                model_layer_dict[(name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index)
                   for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(list(model_layer_dict.keys()))
    return layer_name, index


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons   = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def scale(x, rmax=1, rmin=0):
    # 변경 없음
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        return np.zeros_like(x)
    X_std    = (x - x_min) / (x_max - x_min)
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def update_coverage(input_tensor, model, model_layer_dict, threshold=0.5):
    # 원본은 Keras intermediate_layer_model + xrange →
    # PyTorch forward hook으로 중간 레이어 출력 수집
    activation = {}

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
            def make_hook(n):
                def hook(module, input, output):
                    activation[n] = output.detach()
                return hook
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    for name, act in activation.items():
        # (batch, C, H, W) or (batch, C) → 채널별 평균
        if act.dim() == 4:
            act_mean = act[0].mean(dim=(1, 2)).cpu().numpy()  # (C,)
        else:
            act_mean = act[0].cpu().numpy()  # (C,)

        scaled = scale(act_mean)
        for num_neuron in range(len(scaled)):
            if scaled[num_neuron] > threshold and not model_layer_dict[(name, num_neuron)]:
                model_layer_dict[(name, num_neuron)] = True


def full_coverage(model_layer_dict):
    # 변경 없음
    return all(model_layer_dict.values())


def diverged(predictions1, predictions2):
    # 원본은 모델 3개 → 2개로 변경
    return predictions1 != predictions2