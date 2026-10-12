# CFG 项目快速交接

供 Codex 开始新测试前同步基础信息。具体实验以所选配置和运行目录的 `plan.json` 为准；历史方案与结果见 `docs/`，本文不维护运行进度。

## 位置、服务器与边界

- GitHub：`https://github.com/lotoseater/cfg.git`；服务器账号：`zjiaak`。
- 代码：`/home/zjiaak/SSD/projects/cfg`。
- locpu3：7 张 RTX 3090，GPU **0–6**；locpu4：8 张 RTX 3090，GPU **0–7**。始终显式传 `--gpus`。
- 两台机器共用 NFS：`SSD -> /mnt/ssd1/zjiaak`，`HDD -> /export/zjiaak`；切计算节点不需要迁移代码、模型或结果，GPU/进程则属于各自节点。
- 新文件写在自己的 SSD/HDD 下。共享 ImageNet 数据只读，不修改其他用户的数据、任务或系统驱动。停止任务前确认所属用户及授权。
- h20 的历史根目录是 `/mnt/data/LONG/hongxu/zjiaak`，在 h20 操作限制在该目录。`src/paths.py` 未设置环境变量时仍默认这个根目录，因此在 locpu 上必须设置 `CFG_PROJECT_ROOT=/home/zjiaak`。

## 环境与启动方式

复用已有 fds 环境，直接使用其 Python，无需另建环境或激活 conda：

```bash
cd /home/zjiaak/SSD/projects/cfg
export CFG_PROJECT_ROOT=/home/zjiaak
export PYTHONDONTWRITEBYTECODE=1
export TZ=Asia/Hong_Kong
CFG_PYTHON=/home/zjiaak/SSD/miniconda3/envs/fds/bin/python
"$CFG_PYTHON" -m scripts.run --help
```

已确认的版本：Python 3.9.25、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、NumPy 2.0.2、Pillow 11.3.0、matplotlib 3.9.4。FID/IS 入口还需要 `torch_fidelity`；DINO 使用已有官方源码和 PyTorch 原生 attention，无需 xFormers。

`pyproject.toml` 目前仅配置 Ruff；项目按 `python -m scripts.<entry>` 使用。不要直接运行依赖包导入的 `scripts/*.py`。运行前查看本节点 GPU 占用，并确认当前 Python 能实际执行 CUDA 运算；`nvidia-smi` 恢复不等于模型采样已通过。CPU 测试：

```bash
"$CFG_PYTHON" -m unittest discover -s tests
```

## 项目结构

```text
cfg/
├── cli.md                 # 本文：基础交接
├── README.md              # 概览
├── configs/               # JSON 配置；不同测试单独建文件
├── scripts/               # CLI 入口、任务调度
├── src/
│   ├── paths.py           # 根目录、路径边界、worker 环境
│   ├── config.py          # 通用配置校验与组级覆盖
│   ├── models.py          # JiT 模型构造
│   ├── generation.py      # 生成、图片保存、逐 batch 记录
│   ├── engine.py          # 通用 smoke/formal 调度
│   ├── noise.py           # 噪声库或逐 sample ID 的 CPU RNG
│   ├── samplers/          # CFG 速度场、单步公式、采样循环
│   ├── diagnostics/       # 速度长度/夹角记录与汇总
│   ├── replacement/       # 类别 reference 协议、分片、特征与统计
│   ├── metrics.py         # Inception FID / IS
│   ├── provenance.py      # 源码、上游和权重来源校验
│   └── upstream/          # 固定的 JiT 原版源码与 MIT 许可
├── tests/                 # 数学公式、配对、调度和评估测试
└── docs/                  # 部署、历史实验、reference 实验说明
```

算法改动放在 `src/`，CLI 参数和编排放在 `scripts/`；不要把新测试脚本散放到项目根目录。`src/upstream/` 是核验过的原版模型，不随意修改。`scripts/h20/control.py` 是历史占卡控制器，不用于 locpu 的普通实验。`.refactor-check/` 是忽略的临时验证目录，不是生产入口。

## 公共输入与配置约定

JSON 中的资产路径相对 `$CFG_PROJECT_ROOT`，不是相对仓库。locpu 资产通常为：

| 输入 | 路径 |
|---|---|
| JiT-B/16 权重 | `HDD/fds/checkpoints/jit-b-16-official/checkpoint-last.pth` |
| 共享 10k 噪声库 | `HDD/fds/templates/imagenet256_shared_10k_noise_seed0.npy` |
| Inception 真实统计 | `HDD/fds/fid_stats/jit_in256_stats.npz` |
| Inception 权重 | `.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth` |
| DINOv2 ViT-B/14 | `HDD/dinov2/`，内含权重、固定官方 `source/`、来源清单 |
| ResNet50 ImageNet V2 | `HDD/resnet50/`，内含权重与来源清单 |
| ImageNet validation | `HDD/imagenet/val`，通过用户路径访问共享数据，只读 |

公共模型与 `HDD/cfg`、`HDD/fds` 平级，便于跨项目复用。完整 SHA256、EMA 和来源版本以配置/清单为准；通常使用 `model_ema1`。无需把 fds 项目或数据复制进 cfg。

- 通用配置可从 `configs/locpu4.json` 复制，保留本节点可用的资产路径，再修改算法、组、步数、batch 和诊断设置。名字 `locpu4` 不限制计算节点。
- `baseline50.json`、`followup50.json`、`velocity100.json` 含历史设置或 h20 路径，不应直接当作新节点默认值。历史配置的 batch512 也不是项目强制默认；新测试明确设置自己的 batch。
- 通用入口当前限定 JiT-B/16、256 分辨率、1000 类、10k/50k，以及 25/50/100 步；组级 `overrides` 只允许 `steps`、`cab_theta`。少数目标类/reference 测试使用 `replacement` 协议；新增模型、规模或覆盖字段需同步相关校验。
- 支持 `euler`、`heun`、`heun_flip`、`ab2`、`cab2`、`sg`、`sg_prev`；默认精度策略为 BF16 autocast + FP32 采样状态。所有参数仍以具体配置为准。
- CFG 下一个网络 forward 可以包含条件/reference 两个分支；生成 batch128 通常对应网络 batch256。NFE 按时刻/状态评估计数：Heun/Heun-flip 为 `2*steps-1`，SG 为 `2*steps`，其余为 `steps`。
- 配对输入以 `samples.json` 的 `sample_id/class_id/seed/noise_index` 为准。噪声库为 FP32 NCHW；无噪声库时用 CPU generator、`eval_seed + sample_id` 逐图生成，分片不改变初始噪声。各组的 `noise_hashes.json` 用于证明逐图配对，不能仅凭 seed 相同判断输入一致。
- reference 参数：`None` 使用 null 类1000；整数0–999使用该类别；`"target"` 使用目标类自身。替代 reference 不启用旧的 unconditional 命名几何诊断，避免字段含义混淆。

## 输出约定

```text
/home/zjiaak/HDD/cfg/
├── smoketest/<run_name>/   # smoke、pilot；一次运行一个独立目录
├── runs/<run_name>/        # 正式生成
├── reference/<cache_name>/ # 真实图特征缓存
├── visuals/<run_name>/     # 配对图、合并表、抽样清单；名称与源运行一致
└── cache/                 # 模型运行缓存、后台 launch 日志与控制信息
```

运行根目录包含 `plan.json`、`samples.json`、`status.txt`，各设置使用独立子目录：`images/<sample_id:06d>.png`、`config.json`、`manifest.json`、`noise_hashes.json`、`batches.jsonl`、`network_audit.json`、`generation.json`。图像是256×256 RGB PNG；后处理为 `(x+1)/2`、裁剪到 `[0,1]`、乘255后四舍五入为 uint8。

速度诊断保存到组内 `velocity_geometry/`；通用正式 FID/IS 写入组内 `metrics.json`，运行级汇总为 `summary.json/results.md`。reference 入口保留 `shards/gpu_<id>/` 的原始日志和审计，完成后合并为常规组目录；合并组的 `manifest.json/generation.json` 含各 worker 记录，逐 batch 细节看分片。

reference 评估结果在运行目录的 `classwise/`：各组 `.npz`、`classwise_results.json`、`report.md`、`blind/`；盲选对应表在 `blind_key_private.json`，不放进盲选页。已有可复用的 CPU 真实参照为 `HDD/cfg/reference/replacement_20261012_cpu/`；复用需匹配类别、数据路径、评估权重/源码、预处理、精度、设备和特征 batch。只改生成 batch/负类/采样 seed 不需重提真实特征。

输出采用独占创建，不覆盖已有结果；失败后保留日志和原始分片，新的尝试使用新目录。图片、模型、噪声和结果不提交 Git。

## 常用入口与新测试流程

以下 `configs/my_test.json`、`configs/my_reference_test.json` 是新测试配置的占位名称，先从相应模板复制并编辑：

```bash
# 通用采样器：小烟测；正式阶段需要同配置/源码/全部组的完整 batch 烟测
"$CFG_PYTHON" -m scripts.run smoke --config configs/my_test.json --gpus 0
"$CFG_PYTHON" -m scripts.run formal --config configs/my_test.json --smoke /path/to/smoke --gpus 0,1

# 速度分析
"$CFG_PYTHON" -m scripts.analyze_velocity --input /path/to/run --output /path/to/analysis

# 少数目标类/reference：先看计划，再生成；生成入口不自动评估
"$CFG_PYTHON" -m scripts.run_replacement pilot --config configs/my_reference_test.json --plan-only
"$CFG_PYTHON" -m scripts.run_replacement pilot --config configs/my_reference_test.json --gpus 0,1,2,3,4,5,6
"$CFG_PYTHON" -m scripts.evaluate_replacement --root /path/to/run --reference /path/to/ready_cache --device cpu
```

需要新真实参照时用 `scripts.prepare_replacement --config ... --device ... --output ...`，先确认旧缓存能否复用。`scripts.run_replacement formal --config ... --pilot ... --gpus ...` 要求同协议、同源码且已完成评估的 pilot。通用 `scripts.run formal` 会自动算 FID/IS；reference 入口需显式调用自己的评估入口，两种评估不可混用。

启动新测试前明确比较变量、样本/seed、GPU 列表与输出目录；按改动范围运行必要测试和真实模型烟测。运行中不要改 `src/scripts/configs/tests` 下被记录的源码或 JSON；当前校验把这些目录计入 `code_sha256`，修改配置会使之前的正式运行前置引用失配。`cli.md/README.md/docs` 不计入该哈希，文档更新不会改变生成代码身份。

后台任务使用独立日志和状态文件，交接时看对应 `status.txt` 与错误日志，采用低频完成通知，不持续刷卡或反复核对。不要把历史部署文档中的驱动状态、batch 或正在运行的实验当成当前机器事实。
