'''
reference/deepxplore/ImageNet/gen_diff.py 를 PyTorch + CIFAR-10 + ResNet50 x2 에 맞게 수정
usage: python deepxplore.py --weight_diff 1.0 --weight_nc 0.5 --step 0.01 --seeds 100 --grad_iterations 50 --threshold 0.5 --transformation light
'''

import os
import random
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.models import resnet50

from configs import bcolors
from utils import *

# ── argument parsing ──────────────────────────────────────────────────────────
# 원본: ImageNet 전용 args → CIFAR-10 + 모델 2개에 맞게 수정
parser = argparse.ArgumentParser(
    description='DeepXplore: difference-inducing input generation for CIFAR-10 ResNet50 models')
parser.add_argument('--transformation', help="transformation type", choices=['light', 'occl', 'blackout'], default='light')
parser.add_argument('--weight_diff', help="weight for differential behavior", type=float, default=1.0)
parser.add_argument('--weight_nc', help="weight for neuron coverage", type=float, default=0.5)
parser.add_argument('--step', help="step size of gradient ascent", type=float, default=0.01)
parser.add_argument('--seeds', help="number of seed inputs", type=int, default=100)
parser.add_argument('--grad_iterations', help="number of gradient iterations", type=int, default=50)
parser.add_argument('--threshold', help="threshold for neuron activation", type=float, default=0.5)
# 원본: target_model 0,1,2 (모델 3개) → 0,1 (모델 2개)
parser.add_argument('-t', '--target_model', help="target model to predict differently",
                    choices=[0, 1], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(10, 10), type=tuple)

args = parser.parse_args()

# ── device 설정 ───────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'Device: {device}')

# ── CIFAR-10 이미지 크기 (원본: 224x224 ImageNet → 32x32 CIFAR-10) ────────────
img_rows, img_cols = 32, 32

# ── 모델 로드 (원본: VGG16, VGG19, ResNet50 공유 input tensor → ResNet50 x2) ──
def load_model(path):
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

model1 = load_model('models/model1.pth')
model2 = load_model('models/model2.pth')

# ── coverage table 초기화 (원본: 모델 3개 → 2개) ──────────────────────────────
model_layer_dict1, model_layer_dict2 = init_coverage_tables(model1, model2)

# ── CIFAR-10 test set을 seed로 사용 (원본: ./seeds/ 폴더의 ImageNet jpeg) ──────
mean, std = compute_mean_std()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
os.makedirs('./results', exist_ok=True)

# ── 메인 루프 ─────────────────────────────────────────────────────────────────
# 원본: xrange → range (Python 3)
for i in range(args.seeds):
    # 랜덤 seed 이미지 선택 (원본: random.choice(img_paths))
    idx = random.randint(0, len(testset) - 1)
    gen_img, true_label = testset[idx]
    gen_img = gen_img.unsqueeze(0).to(device)  # (1, 3, 32, 32)
    orig_img = gen_img.clone()

    # ── 이미 불일치하는 경우 처리 ──────────────────────────────────────────────
    with torch.no_grad():
        pred1 = model1(gen_img).argmax(1).item()
        pred2 = model2(gen_img).argmax(1).item()

    if diverged(pred1, pred2):
        print(bcolors.OKGREEN +
              f'[seed {i}] already differs: model1={decode_label(pred1)}, model2={decode_label(pred2)}' +
              bcolors.ENDC)

        update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
        update_coverage(gen_img, model2, model_layer_dict2, args.threshold)

        # 커버리지 출력
        _, _, nc1 = neuron_covered(model_layer_dict1)
        _, _, nc2 = neuron_covered(model_layer_dict2)
        avg_nc = (nc1 + nc2) / 2.0
        print(bcolors.OKGREEN +
              f'coverage: model1={nc1:.3f}, model2={nc2:.3f}, avg={avg_nc:.3f}' +
              bcolors.ENDC)

        # 결과 저장 (원본: scipy.misc.imsave → matplotlib)
        img_vis = deprocess_image(gen_img.squeeze(0), np.array(mean), np.array(std))
        plt.imsave(f'./results/already_differ_{decode_label(pred1)}_{decode_label(pred2)}_{i}.png', img_vis)
        continue

    # ── 모든 모델이 동의하는 경우: gradient ascent로 불일치 유도 ────────────────
    orig_label = pred1

    # 커버할 뉴런 선택
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2)

    # gradient iteration (원본: xrange → range)
    for iters in range(args.grad_iterations):
        gen_img = gen_img.detach().requires_grad_(True)

        out1 = model1(gen_img)
        out2 = model2(gen_img)

        # ── joint loss 계산 ────────────────────────────────────────────────────
        # 원본: K.mean(model.get_layer('predictions').output[..., orig_label])
        # → PyTorch: output[0, orig_label]
        if args.target_model == 0:
            # model1이 orig_label 확률을 낮추고, model2는 높이도록
            loss_diff = -args.weight_diff * out1[0, orig_label] + out2[0, orig_label]
        else:
            # model2가 orig_label 확률을 낮추고, model1은 높이도록
            loss_diff = out1[0, orig_label] - args.weight_diff * out2[0, orig_label]

        # ── neuron coverage loss ───────────────────────────────────────────────
        # 원본: K.mean(model.get_layer(layer_name).output[..., index])
        # → PyTorch: forward hook으로 중간 레이어 출력 추출
        activation1, activation2 = {}, {}

        def make_hook(store, name):
            def hook(module, input, output):
                store[name] = output
            return hook

        hooks = []
        for name, module in model1.named_modules():
            if name == layer_name1:
                hooks.append(module.register_forward_hook(make_hook(activation1, layer_name1)))
        for name, module in model2.named_modules():
            if name == layer_name2:
                hooks.append(module.register_forward_hook(make_hook(activation2, layer_name2)))

        # hook이 등록된 상태에서 forward
        out1 = model1(gen_img)
        out2 = model2(gen_img)

        for h in hooks:
            h.remove()

        # 뉴런 활성화값 추출
        act1 = activation1.get(layer_name1)
        act2 = activation2.get(layer_name2)

        loss_nc = torch.tensor(0.0, device=device)
        if act1 is not None:
            if act1.dim() == 4:
                loss_nc += args.weight_nc * act1[0, index1].mean()
            else:
                loss_nc += args.weight_nc * act1[0, index1]
        if act2 is not None:
            if act2.dim() == 4:
                loss_nc += args.weight_nc * act2[0, index2].mean()
            else:
                loss_nc += args.weight_nc * act2[0, index2]

        # 최종 loss
        final_loss = loss_diff + loss_nc
        final_loss.backward()

        # ── gradient constraint 적용 후 이미지 업데이트 ────────────────────────
        # 원본: grads = normalize(K.gradients(...)) → PyTorch autograd
        grads_value = gen_img.grad.detach().cpu().numpy()
        grads_value = normalize(grads_value)

        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point, args.occlusion_size)
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)

        with torch.no_grad():
            gen_img = gen_img + args.step * torch.tensor(grads_value, device=device)

        # ── 불일치 감지 ────────────────────────────────────────────────────────
        with torch.no_grad():
            pred1 = model1(gen_img).argmax(1).item()
            pred2 = model2(gen_img).argmax(1).item()

        if diverged(pred1, pred2):
            update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold)

            _, _, nc1 = neuron_covered(model_layer_dict1)
            _, _, nc2 = neuron_covered(model_layer_dict2)
            avg_nc = (nc1 + nc2) / 2.0
            print(bcolors.OKGREEN +
                  f'[seed {i}, iter {iters}] diverged! '
                  f'model1={decode_label(pred1)}, model2={decode_label(pred2)} | '
                  f'avg coverage={avg_nc:.3f}' +
                  bcolors.ENDC)

            # 결과 저장 (원본: ./generated_inputs/ → ./results/)
            img_vis = deprocess_image(gen_img.squeeze(0), np.array(mean), np.array(std))
            orig_vis = deprocess_image(orig_img.squeeze(0), np.array(mean), np.array(std))
            plt.imsave(
                f'./results/{args.transformation}_{decode_label(pred1)}_{decode_label(pred2)}_{i}.png',
                img_vis)
            plt.imsave(
                f'./results/{args.transformation}_{decode_label(pred1)}_{decode_label(pred2)}_{i}_orig.png',
                orig_vis)
            break

print(bcolors.OKBLUE + '\n=== 최종 결과 ===' + bcolors.ENDC)
_, _, nc1 = neuron_covered(model_layer_dict1)
_, _, nc2 = neuron_covered(model_layer_dict2)
print(f'model1 neuron coverage: {nc1*100:.2f}%')
print(f'model2 neuron coverage: {nc2*100:.2f}%')
print(f'averaged neuron coverage: {(nc1+nc2)/2*100:.2f}%')