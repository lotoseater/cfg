# 用类别速度替换 CFG 的 unconditional 分支

本轮使用 locpu3 的七张 GPU（0–6）。locpu3 与 locpu4 共用 SSD/HDD，实验代码在 `/home/zjiaak/SSD/projects/cfg`，模型资产位于各项目共用的 HDD 目录。首次 batch512 pilot 在共享显存条件下 OOM，未生成图片；当前统一改为 batch128，处于停止状态。

## 公共模型资产

| 目录 | 内容 | 权重大小 |
|---|---|---:|
| `/home/zjiaak/HDD/dinov2/` | 官方 ViT-B/14 权重、`source/` 官方源码、来源/哈希清单 | 346,378,731 bytes |
| `/home/zjiaak/HDD/resnet50/` | Torchvision ResNet-50 ImageNet1K V2 权重、来源/哈希清单 | 102,540,417 bytes |

权重合计 449 MB（428 MiB）。DINOv2 源码固定到 `2302b6bf46953431b969155307b9bed152754069`，支持已有 fds 的 Python 3.9。使用 PyTorch 原生 attention，不需要安装 xFormers 或改变 fds 环境。资产完整 SHA256 在 `configs/replacement.json` 和各模型目录的 `weights-manifest.json`；ResNet 权重同时匹配官方文件名的哈希前缀。

来源：[DINOv2 官方模型](https://github.com/facebookresearch/dinov2)、[ResNet 官方权重定义](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)。下载的模型权重和官方源码不进入 cfg Git 仓库。

## 固定方案

主实验使用标准 Heun 50 步/99 NFE、CFG=3，复用 JiT-B/16 `model_ema1`、256 分辨率、BF16 模型计算/FP32 状态、原有 CFG interval 与后处理，batch128（CFG 双分支模型输入为 256）。pilot 和正式阶段的全部设置使用同一 batch。只替换参考分支的标签：

`v_guided = v_reference + scale * (v_target - v_reference)`。

共六种设置：unconditional、四个固定随机类别、目标类自身。同类对照在 CFG 生效区间也退化为条件速度，单列解释，排除在主要 Holm 检验族之外。各设置保持相同的双分支网络 batch，不额外评估模型。

类别使用 Python `random.Random(20261012)`，先从 1,000 类抽八个目标，再从剩余类别抽四个固定参考，结果锁定在配置中。它们代表有限类别上的检验，不能证明所有类别都等效。

| 目标类 index | 类名 |
|---:|---|
| 1 | goldfish |
| 8 | hen |
| 44 | alligator lizard |
| 51 | triceratops |
| 416 | balance beam |
| 575 | golfcart |
| 808 | sombrero |
| 994 | stinkhorn |

四个替代类为 548 entertainment center、690 oxcart、784 screwdriver、843 swing。

pilot 每个目标类 × 设置 128 张，合计 6,144 张；正式阶段每组 1,000 张，合计 48,000 张。六个设置各自产生全部目标类，分配到 locpu3 的 0–5 号 GPU 上，6 号卡备用。正式阶段使用独立的新 seed，不能直接复用 pilot 样本。CPU 噪声按 sample ID 创建，分组、分 batch 都不会改变噪声。每组保存逐图噪声哈希；完成后核验配对、样本集合与固定配置。

## 指标与统计

两个主指标：独立 ResNet-50 的目标类 top-1 命中率，以及 DINOv2 CLS 特征上的多项式 MMD²（DINO-KID）。同时保存目标类概率、主要混淆类别、归一化类内多样性，以及真实验证图的分类表现。

DINO 特征使用 FP32、resize256/bicubic、center224、ImageNet normalization，取原始最终归一化 CLS 输出。KID 核为 `(x·y / 768 + 1)^3`，采用去除对角线的无偏估计，允许负值。多样性使用 unit-normalized DINO 特征的平均两两 cosine distance，除以真实类内距离。较高多样性不自动表示更好质量。

2,000 次 basic bootstrap：生成样本按 seed 成对重采样，真实样本在两个设置之间共享同一重采样。报告每个设置点估计/95%CI、相对 baseline 差值/95%CI。10,000 次成对随机交换生成样本所属设置，计算双侧 permutation p；KID 的检验以真实参照固定为条件，而 CI 包含真实图重采样的不确定性。多样性、概率也给出探索性 p。

Holm 校正族包含八个目标 × 四个替代类 × 两个主指标，共 64 项；pilot 所有结果均为探索性。只有正式阶段的命中率/KID 检验用于主要差异结论。只有每类 50 张真实图，区间可能较宽。命中率的普通 bootstrap 在无错误或无 discordant pairs 时可能退化，不能凭退化区间宣布等效。

非显著不是等效。目前配置保留命中率 ±2pp 的候选界限，KID 等效界限尚未确定；代码不自动给出“替换无影响”的结论。

## 真实参照与 CPU 准备

读取 `/home/zjiaak/HDD/imagenet/val`，每个目标类 50 张。该路径通过用户 HDD 中的软链接访问共享数据；程序只读源图像，不将共享数据目录加入输出边界。验证图文件名、类别 WNID、SHA256 和分类器类别名称记录在 `real_manifest.json`。生成模型训练图不作为真实性参照。

准备命令（仓库根目录、fds 环境）：

```bash
export CFG_PROJECT_ROOT=/home/zjiaak
export PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/zjiaak/SSD/miniconda3/envs/fds/bin/python

$PYTHON -m scripts.prepare_replacement --device cpu
$PYTHON -m scripts.run_replacement pilot --plan-only
```

batch128 的 CPU 参考缓存为 `/home/zjiaak/HDD/cfg/reference/replacement_20261012_cpu_bs128/`，包含八个目标类的 400 张验证图。旧目录 `replacement_20261012_cpu/` 保留 batch512 协议，不用于当前配置的评测。参考缓存默认写入 `HDD/cfg/reference/replacement_<timestamp>/`。每次生成/评测输出独立文件夹。真实与生成特征必须使用相同 evaluator 版本、权重、预处理、精度和计算设备；若改为 GPU 特征评估，需先重建 GPU 真实参照。

## CUDA 恢复后

先真实模型 pilot，检查显存、配对图片、真实分类器表现和每类指标。生成入口先逐卡进行 CUDA 实际运算；失败时不创建生成运行目录、不修改系统驱动。

```bash
$PYTHON -m scripts.run_replacement pilot --gpus 0,1,2,3,4,5
$PYTHON -m scripts.evaluate_replacement --root /path/to/pilot --reference /path/to/ready_reference --device cpu

# 查看 pilot 后再执行；要求同配置/源码且已完成评测的 pilot
$PYTHON -m scripts.run_replacement formal --pilot /path/to/pilot --gpus 0,1,2,3,4,5
$PYTHON -m scripts.evaluate_replacement --root /path/to/formal --reference /path/to/ready_reference --device cpu
```

batch128 是当前统一设置，尚未完成该 batch 的完整 pilot；先前的 batch512 单次前向通过，但完整采样 OOM，不能把单次前向视为完整采样验证。若再次改 batch，所有设置统一修改并重新 pilot。

评测输出包含原始特征/分类概率、`classwise_results.json`、`report.md`，以及 `blind/` 的同 seed A/B 样图、真实图参照与打分 CSV。盲选页随机排列试次与左右位置，标明目标类名称；对应关系在页外的 `blind_key_private.json`。人工分数需由评审填写，程序不代填。
