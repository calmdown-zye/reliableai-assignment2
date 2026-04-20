import os
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet50
from collections import defaultdict

from configs import bcolors
from utils import (init_coverage_tables, neuron_to_cover, neuron_covered,
                   update_coverage, diverged, decode_label,
                   deprocess_image, compute_mean_std,
                   constraint_light, normalize)


# ── 디바이스 설정 ─────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'Device: {device}')

# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model(path):
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

print('\n 모델 로드 ')
model1 = load_model('models/model1.pth')
model2 = load_model('models/model2.pth')
print('모델 2개 로드 완료')


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print('\n 데이터 로드 ')
mean, std = compute_mean_std()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
os.makedirs('./results', exist_ok=True)



# ── DeepXplore 실행 ───────────────────────────────────────────────────────────
print('\n DeepXplore 실행 중')

# 하이퍼파라미터
NUM_SEEDS       = 100
GRAD_ITERATIONS = 50
STEP            = 0.01
WEIGHT_DIFF     = 1.0
WEIGHT_NC       = 0.5
THRESHOLD       = 0.5

model_layer_dict1, model_layer_dict2 = init_coverage_tables(model1, model2)

disagreements = []  # (orig_img, gen_img, pred1, pred2) 저장

for i in range(NUM_SEEDS):
    idx = random.randint(0, len(testset) - 1)
    gen_img, true_label = testset[idx]
    gen_img = gen_img.unsqueeze(0).to(device)
    orig_img = gen_img.clone()

    with torch.no_grad():
        pred1 = model1(gen_img).argmax(1).item()
        pred2 = model2(gen_img).argmax(1).item()

    # 이미 불일치하는 경우
    if diverged(pred1, pred2):
        update_coverage(gen_img, model1, model_layer_dict1, THRESHOLD)
        update_coverage(gen_img, model2, model_layer_dict2, THRESHOLD)
        disagreements.append((orig_img.clone(), gen_img.clone(), pred1, pred2, true_label))
        continue

    # gradient ascent로 불일치 유도
    orig_label = pred1
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2)

    for iters in range(GRAD_ITERATIONS):
        gen_img = gen_img.detach().requires_grad_(True)

        out1 = model1(gen_img)
        out2 = model2(gen_img)
        loss_diff = -WEIGHT_DIFF * out1[0, orig_label] + out2[0, orig_label]

        # neuron coverage loss
        activation1, activation2 = {}, {}
        hooks = []
        for name, module in model1.named_modules():
            if name == layer_name1:
                def make_hook(store, n):
                    def hook(m, i, o): store[n] = o
                    return hook
                hooks.append(module.register_forward_hook(make_hook(activation1, layer_name1)))
        for name, module in model2.named_modules():
            if name == layer_name2:
                def make_hook(store, n):
                    def hook(m, i, o): store[n] = o
                    return hook
                hooks.append(module.register_forward_hook(make_hook(activation2, layer_name2)))

        out1 = model1(gen_img)
        out2 = model2(gen_img)
        for h in hooks:
            h.remove()

        loss_nc = torch.tensor(0.0, device=device)
        act1 = activation1.get(layer_name1)
        act2 = activation2.get(layer_name2)
        if act1 is not None:
            loss_nc += WEIGHT_NC * (act1[0, index1].mean() if act1.dim() == 4 else act1[0, index1])
        if act2 is not None:
            loss_nc += WEIGHT_NC * (act2[0, index2].mean() if act2.dim() == 4 else act2[0, index2])

        final_loss = loss_diff + loss_nc
        final_loss.backward()

        grads_value = normalize(gen_img.grad.detach().cpu().numpy())
        grads_value = constraint_light(grads_value)

        with torch.no_grad():
            gen_img = gen_img + STEP * torch.tensor(grads_value, device=device)

        with torch.no_grad():
            pred1 = model1(gen_img).argmax(1).item()
            pred2 = model2(gen_img).argmax(1).item()

        if diverged(pred1, pred2):
            update_coverage(gen_img, model1, model_layer_dict1, THRESHOLD)
            update_coverage(gen_img, model2, model_layer_dict2, THRESHOLD)
            disagreements.append((orig_img.clone(), gen_img.clone(), pred1, pred2, true_label))
            break



# ── 결과 출력 및 시각화 ───────────────────────────────────────────────────────
print('\n 결과 저장 및 식가화')

_, _, nc1 = neuron_covered(model_layer_dict1)
_, _, nc2 = neuron_covered(model_layer_dict2)

print('\n' + '='*50)
print(f'  불일치 입력 총 개수 : {len(disagreements)}')
print(f'  model1 neuron coverage: {nc1*100:.2f}%')
print(f'  model2 neuron coverage: {nc2*100:.2f}%')
print(f'  평균 neuron coverage  : {(nc1+nc2)/2*100:.2f}%')
print('='*50)

# 5개 시각화 (원본 + 변형 이미지 나란히)
n_vis = min(5, len(disagreements))
fig, axes = plt.subplots(n_vis, 2, figsize=(6, n_vis * 3))

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

for idx in range(n_vis):
    orig_img, gen_img, pred1, pred2, true_label = disagreements[idx]
    orig_vis = deprocess_image(orig_img.squeeze(0), np.array(mean), np.array(std))
    gen_vis  = deprocess_image(gen_img.squeeze(0),  np.array(mean), np.array(std))

    axes[idx][0].imshow(orig_vis)
    axes[idx][0].set_title(f'Original\nTrue: {CIFAR10_CLASSES[true_label]}', fontsize=9)
    axes[idx][0].axis('off')

    axes[idx][1].imshow(gen_vis)
    axes[idx][1].set_title(
        f'Disagreement\nModel1: {decode_label(pred1)} / Model2: {decode_label(pred2)}',
        fontsize=9)
    axes[idx][1].axis('off')

plt.suptitle('DeepXplore: Disagreement-inducing Inputs', fontsize=12)
plt.tight_layout()
save_path = './results/disagreements_summary.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'\n시각화 저장 5개: {save_path}')


# 간단한 분석 출력
print('\n[분석] 불일치 유발 입력 패턴:')
from collections import Counter
pairs = [(decode_label(d[2]), decode_label(d[3])) for d in disagreements]
common = Counter(pairs).most_common(5)
for (p1, p2), cnt in common:
    print(f'  model1={p1} vs model2={p2}: {cnt}회')