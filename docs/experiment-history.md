> 本文保留各轮实验当时的目录、命令与运行记录。当前代码组织及入口以仓库 README 为准。

# AB2 / CAB2 / SG / SG-prev paired comparison

跨服务器部署见 [部署说明](deployment.md)。下面保留 h20 上各轮实验的配置与运行说明；启动根目录现在可通过 `CFG_PROJECT_ROOT` 指定，manifest 同时记录 Git revision 与源文件 SHA256。

新项目在 `/mnt/data/LONG/hongxu/zjiaak/SSD/projects/cfg`，复用 `envs/fds/bin/python`。
服务器实际目录名为大写SSD/HDD；所有写入位于zjiaak，唯一经用户授权的额外操作是暂停/恢复
`/mnt/data/LONG/hongxu/take_gpu.py`。不修改fds代码或旧结果，不安装依赖。

## Locked comparison

`config.json`统一所有组：JiT-B/16、ImageNet256、EMA1、50步uniform-t、CFG1/3、
strict interval(.1,1)、noise_scale1、t_eps.05、每组10000张、每类10张、batch128。
四方法×两CFG共八个新组，一卡一组。已有Heun只读引用，不重跑10k或重算指标：

- `HDD/fds/runs/baseline8_20261007_cfg13/heun_cfg1`，FID-10k 28.750298349513855。
- `HDD/fds/runs/baseline8_20261007_cfg13/heun_cfg3`，FID-10k 6.063870346846045。

新checkpoint：`HDD/checkpoints/jit-b-16/checkpoint-last.pth`，SHA256与旧Heun完全相同。
共享噪声：`HDD/fds/templates/baseline8_10k_seed0.npy`，float32(10000,3,256,256)，
SHA256 112a421f399acece9e64e7712b69c873696211c4d8944895157e80d38ec7d398。
按全局ID索引，class_id=sample_id//10；保留bank seed0及noise_index，禁止换新随机流。
batch、GPU数量和分片不改变初始噪声，与旧Heun逐元素配对。

工程参数对齐旧Heun：BF16 autocast、FP32状态、Flash attention、禁用torch.compile、冻结权重。
CFG1仍计算双分支，保持旧网络batch形状。PNG按原round/clip/uint8 RGB流程，六位ID；旧五位PNG不改名。
FID/IS对齐旧baseline8：torch-fidelity0.4.0、inception-v3-compat、同一参考和权重，
reference/generated mu和sigma都float64，IS logits_unbiased、10 splits、rng_seed0。
不采用acceptance8_50k的rng_seed2020/原统计精度规则。

## Formula details

原版Heun用于回归：50步99 NFE，末步Euler，未列入新组。
AB2第0步Euler，后续1.5*v-.5*v_prev，缓存原始CFG velocity。
CAB2第0步Euler，第1/2步AB2，i=3(t=.06)开始variable-step CAB2，theta=.9。
i=3是第一个有有限k_i、k_(i-1)、k_(i-2)的点；i=2仍涉及奇异lambda0。
用e=CFG clean-v，k=lambda_next-lambda_current<0，不再加负号，末步继续CAB。
SG同状态调用t和max(t-.001,0)，各自CFG及截断分母，scale1，100 NFE。
SG-prev用v+.05*(v-v_prev)，缓存原始v，50 NFE。
所有模型调用t<1，t=.98分母.05；AB2/CAB2/SG-prev都是50 NFE。
NFE为一次时间/状态评估，CFG合批计一次；另存真实网络batch尺寸和调用计数。
CAB规定的e=clean-v在截断段不严格等于dy/dlambda；仍保留任务公式。
SG-prev的.05为局部量级匹配，历史差含状态变化，不宣称与SG同状态偏移严格等价。

## Outputs and lifecycle

每次正式执行新建`HDD/cfg/runs/formal_<Hong Kong timestamp>`，内部八组子目录。
烟测：`HDD/cfg/smoketest/smoke_<timestamp>`；审计及控制日志也在HDD/cfg内。
每组保存images、config、manifest、generation、network_audit、noise_hashes和status。
外层保存plan、samples、reused_baselines、日志、summary和正式results.md。
结束核对八组噪声哈希、PNG数量/ID/损坏、每类10张和有限指标。
旧基线manifest/metrics哈希、源commit/diff及原路径记录到结果，不改旧文件。
新代码未提交Git，记录project_git_commit=null及文件SHA256；官方commit另记。
`upstream/`为fds acceptance8保存的官方79c0106原始字节副本，运行时核验哈希。

用户已授权停占卡，烟测、生成、算分全部结束后恢复。
`control.py`核验指定PID的解释器、argv、cwd、启动时间、脚本哈希和相关环境，
只对该take_gpu发SIGINT。先重放旧Heun首个满bs128 batch核对原PNG像素，
再八组各128张/bs128/50步空卡烟测；不对旧Heun全量重跑或重算FID。
烟测完成等待本聊天检查preview并写proceed.json，然后正式八卡生成和离线算分。
服务器tmux控制器等全部本轮子进程退出及GPU释放，再按原配置恢复占卡并核验新PID。
失败留日志、不自动重跑/改batch；已启动子进程退出后恢复占卡。其他程序不停止。

## Manual commands

```bash
cd /mnt/data/LONG/hongxu/zjiaak/SSD/projects/cfg
PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 \
  /mnt/data/LONG/hongxu/zjiaak/envs/fds/bin/python test_samplers.py

/mnt/data/LONG/hongxu/zjiaak/envs/fds/bin/python run.py smoke \
  --gpus 0,1,2,3,4,5,6,7 --smoke-images 128 --smoke-batch-size 128

/mnt/data/LONG/hongxu/zjiaak/envs/fds/bin/python run.py formal \
  --gpus 0,1,2,3,4,5,6,7 --smoke /absolute/completed/smoke/path

/mnt/data/LONG/hongxu/zjiaak/envs/fds/bin/python run.py validate --root /absolute/run/path
```

配置/代码SHA256须与已完成烟测相同才可formal。
CPU测试包括原始Euler/Heun精确回归、CFG边界、缓存、CAB解析路径和独立非线性公式、
SG方向、NFE/终点和噪声跨batch/shard一致性。生命周期测试模拟额外GPU进程、PID复用和正常/失败恢复，
不向真实进程发信号。真实EMA回归已逐元素复现原始Euler/Heun（CFG1/3）。

## Followup round (2026-10-11)

`config_followup_20261011.json` adds eight paired groups in one run folder:
Euler50 (50 NFE), Heun25 (49 NFE), SG25 (50 NFE), CAB2/50 theta=0 (50 NFE), each CFG1/3.
Per-group overrides are limited to steps and cab_theta; all input, batch128, precision,
checkpoint, PNG and metric settings remain the same. SG shift=.001, scale=1 and CAB startup unchanged.
`plan.json` records all overrides; each group config records its effective settings.
Run smoke with `--config config_followup_20261011.json --smoke-images 128 --smoke-batch-size 128`,
then formal with the same config and `--smoke` referencing that smoke.
This round uses available memory alongside existing GPU tasks; no occupancy process is stopped.
FID/IS remain paired; timings are from shared GPUs and must not be directly compared to the first round's exclusive timings.


## 2026-10-11: 100 NFE velocity geometry round

User confirmed alternating Heun-flip. Eight paired 10k groups, CFG=1/3, one GPU per group, generation batch512 (network conditional/unconditional combined batch1024): AB2 and CAB2 theta=.9 at 100 uniform-t steps; SG shift=.05/scale=1 at 50 steps; alternating Heun-flip at 50 steps/99 NFE, final Euler. Same EMA1, noise bank, label IDs, BF16/Flash/eager, FP32 state, CFG interval and t_eps as preceding rounds. Metric batch128 and FID/IS protocol unchanged. No shared-state auxiliary diagnostic or extra diagnostic forwards in this round. No old bs128 result is treated as a bitwise bs512 reproduction.

Use config_geometry_20261011.json. For alternating flip, zero-based odd update intervals flip inputs and reverse predictions into canonical pixel coordinates. Both evaluations in one Heun interval have the same phase; the last Euler interval uses its interval phase. This reproduces SSD/projects/fds/scripts/baseline8/official_flip.py heun_flip, not native FDS-flip divergence selection.

velocity_geometry/schema.json describes metric columns. batch_XXXXXX.npz contains sample_ids, class_ids, and FP32 values[step,sample,metric]. Per-image CHW L2/RMS, cosine, stable atan2 angle in degrees, dot mean, norm ratio and explicit difference RMS are recorded for basic conditional/unconditional velocities, CFG velocity, history and method-specific branches. CFG1 also evaluates unconditional prediction. AB histories cache raw conditional/unconditional/CFG velocities. SG evaluates the same state at max(t-.05,0); each branch retains its independent denominator and CFG gate. Heun compares initial and predictor evaluations in canonical coordinates; it does not additionally evaluate both flip frames at the same state/time.

All methods record combined velocity and effective state update (x_next-x)/dt, correction=combined-current CFG velocity, signed parallel/perpendicular correction magnitudes and ratios, and alignment to conditional-minus-unconditional direction. CAB additionally records e=clean_cfg-v_cfg and two histories, transformed predictor/defect, k/ratios/coefficients, and equivalent pixel-space drift/history/theta/predictor contributions. CAB-startup fields are absent/NaN until step index3. Zero-vector angles and zero-denominator ratios are NaN, not zero degrees; summary.json contains valid/undefined counts, per-time mean/std and p10/p50/p90. Norms use FP32 reduction with autocast disabled; close-vector angles use atan2 rather than acos.

Geometry observers never evaluate the network, mutate sampling tensors, or change trajectories. Sampling timing includes geometry reduction and CPU transfer; file compression/saving and final per-time summary are included in total generation time. Timings therefore differ from uninstrumented preceding rounds. Config, source hashes, call count and geometry coverage are audited. Full batch512 smoke of all eight groups must pass before formal launch; no automatic batch reduction.

Outputs: HDD/cfg/smoketest/smoke_<timestamp> and HDD/cfg/runs/formal_<timestamp>; each group has images/, metrics.json, velocity_geometry/. Existing other-user GPU services are retained.
