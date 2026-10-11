# GitHub 与 locpu4 部署

仓库：<https://github.com/lotoseater/cfg>。

代码目录：`/home/zjiaak/SSD/projects/cfg`。

已有 fds Python：`/home/zjiaak/SSD/miniconda3/envs/fds/bin/python`。

本次只迁移 cfg 源码、实验配置和三份经过哈希核验的 JiT 上游源文件。复用 locpu4 的 fds 环境，不安装依赖，不迁移权重、噪声、FID 参考统计或运行结果。

## 路径

```bash
export CFG_PROJECT_ROOT=/home/zjiaak
export PYTHONDONTWRITEBYTECODE=1
cd /home/zjiaak/SSD/projects/cfg
```

未设置 `CFG_PROJECT_ROOT` 时仍使用 h20 的 `/mnt/data/LONG/hongxu/zjiaak`。配置中的资产路径相对该根目录解析，运行输出为 `HDD/cfg/runs` 和 `HDD/cfg/smoketest`。路径校验支持根目录内的 SSD/HDD 挂载软链接；其他软链接仍不能逃出这些边界。

原配置保留不变。最新一轮的原始设置在 `config_geometry_20261011.json`，包含 AB2/CAB2 的 100 步、SG shift=.05 的 50 步、交替 Heun-flip 的 50 步，以及 CFG1/3 的逐样本速度统计。

locpu4 使用 `config_locpu4.json`。它只改变 checkpoint、噪声和 Inception 权重的三个路径，其他设置与 geometry 配置完全一致。2026-10-11 已完整计算 locpu4 的 checkpoint、噪声、FID 参考统计和 Inception 权重 SHA256，四项都与 h20 配置一致，可以保持逐元素输入配对而无需迁移资产。

`config.json` 引用了 h20 的历史 Heun 结果；这些结果不随代码迁移。准备新实验时使用不引用旧输出的 geometry 配置，并先核对已有资产哈希。不同的噪声文件不能仅凭 seed 相同就视为逐元素配对。

## 验证代码

```bash
/home/zjiaak/SSD/miniconda3/envs/fds/bin/python test_common.py
/home/zjiaak/SSD/miniconda3/envs/fds/bin/python test_samplers.py
/home/zjiaak/SSD/miniconda3/envs/fds/bin/python test_geometry.py
```

这三项测试在 CPU 上进行；通过不代表 CUDA 计算或真实模型生成已通过烟测。

`run.py` 使用指定的 GPU 列表分配各组，多组可以在同一张卡上依次执行。locpu4 的八张卡可传 `--gpus 0,1,2,3,4,5,6,7`。正式运行要求同配置、同源码且覆盖所有组的完整 batch 烟测。

最新一轮 batch512 来自 H20 实验，迁移时不自动改 batch。新硬件上的可用 batch 应通过真实模型烟测确定。

`run_locpu4.sh` 设置工作根目录并使用已有 fds Python，参数原样传给 `run.py`。例如查看用法：

```bash
bash run_locpu4.sh --help
```

## 驱动状态

2026-10-11 检查时，locpu4 的普通 `nvidia-smi` 报 `Driver/library version mismatch`，NVML 库版本为 580.178。此次迁移不修改系统驱动，不启动 GPU 实验；fds 环境的真实 GPU 计算仍需在驱动可用后验证。

## 控制脚本与来源

`control.py` 是此前 h20 占卡程序的专用控制器。在 locpu4 直接使用 `run.py`，不使用该控制器，也不停止其他人的进程。

`upstream/` 保留实验使用的 JiT 源码快照，文件 SHA256 在配置中，启动时会核验。JiT 上游来源为 <https://github.com/LTH14/JiT>，MIT 许可见 `upstream/LICENSE`。原始采样公式和精度策略不因部署改变。
