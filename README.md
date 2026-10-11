# CFG Sampling Experiments

JiT-B/16 在 ImageNet-256 上的配对采样实验：比较 AB2、CAB2、Self-Guidance 和交替 Heun-flip，并记录条件/无条件速度、历史速度与算法修正的大小和角度。

目录沿用 h20 FDS 项目的分工：`src/` 放实现，`scripts/` 放运行入口。配置、测试和实验记录单独存放。

```text
cfg/
├── src/
│   ├── samplers/       # CFG 速度计算与各算法的单步公式
│   ├── diagnostics/    # 向量几何、逐样本记录、跨 batch 汇总
│   ├── engine.py       # 运行计划和 GPU 任务调度
│   ├── generation.py   # 模型生成与输出记录
│   ├── metrics.py      # FID / IS
│   └── upstream/       # 经哈希核验的原始模型代码
├── scripts/            # 生成、评测、分析、审计入口
├── configs/            # 各轮实验与服务器配置
├── tests/              # 公式、原版回归、诊断和调度测试
└── docs/               # 部署说明与历史实验规范
```

## 使用

复用现有 `fds` 环境，在仓库根目录运行。locpu4 的配置复用本机已有资产；checkpoint、噪声、FID 统计和 Inception 权重已核验与 h20 相同。

```bash
cd /home/zjiaak/SSD/projects/cfg
export CFG_PROJECT_ROOT=/home/zjiaak
export PYTHONDONTWRITEBYTECODE=1

# 查看入口与参数
/home/zjiaak/SSD/miniconda3/envs/fds/bin/python -m scripts.run --help

# CPU 测试
/home/zjiaak/SSD/miniconda3/envs/fds/bin/python -m unittest discover -s tests
```

启动烟测和正式实验时，通过 `--config configs/locpu4.json` 和 `--gpus` 指定实验及设备。正式实验需要引用同配置、同源码的完整 batch 烟测目录。也可用 `bash scripts/run_locpu4.sh` 启动同一入口。

## 最新比较

| 方法 | 步数 | NFE | 设置 |
|---|---:|---:|---|
| AB2 | 100 | 100 | 当前/历史 CFG 速度外推 |
| CAB2 | 100 | 100 | θ=0.9，变换坐标中的历史修正 |
| SG | 50 | 100 | 同状态，时间偏移 0.05，scale=1 |
| Heun-flip | 50 | 99 | 交替水平翻转，末步 Euler |

每个方法分别使用 CFG=1/3，共八组；同一 EMA、初始噪声、标签与 sample ID。最新配置保留 h20 的 batch512，目标硬件的可用 batch 需另做真实模型烟测。

## 输出与分析

每次运行创建独立目录：`$CFG_PROJECT_ROOT/HDD/cfg/runs/` 或 `HDD/cfg/smoketest/`。各组保存图片、配置、源文件哈希、调用记录、指标，以及 `velocity_geometry/` 中的逐样本统计和汇总。诊断不增加模型评估次数。

```bash
python -m scripts.analyze_velocity --input /path/to/run --output /path/to/analysis
```

分析入口也接受已保存的 `velocity_curves.json`，报告中的数值从输入统计计算。

部署细节见 [docs/deployment.md](docs/deployment.md)；公式与历史记录见 [实验规范](docs/experiment-spec.md) 和 [实验记录](docs/experiment-history.md)。上游源文件的许可保留在 [src/upstream/LICENSE](src/upstream/LICENSE)。

## 替代 unconditional 的每类实验

公共模型目录为 `/home/zjiaak/HDD/dinov2/` 和 `/home/zjiaak/HDD/resnet50/`，与 `cfg`、`fds` 平级。方案、资产、CPU 准备及 GPU 恢复后的入口见 [docs/reference-replacement.md](docs/reference-replacement.md)。配置在 `configs/replacement.json`；指标按目标类别计算，使用冻结分类器、DINO-KID 与类内多样性。
