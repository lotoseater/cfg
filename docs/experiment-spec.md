> 原始实验规范；历史路径和命令按当时记录保留。当前入口见仓库 README。

# JiT-B/16：AB2、CAB2、SG、SG-prev 实验规范

## 1. 实验目标

在 JiT-B/16、ImageNet 256×256 上比较以下 8 组设置：

| 组号 | 方法 | CFG | 主要参数 | 采样步数 | 模型评估次数（NFE） |
|---:|---|---:|---|---:|---:|
| 1 | AB2 | 1 | — | 50 | 50 |
| 2 | AB2 | 3 | — | 50 | 50 |
| 3 | CAB2 | 1 | `theta=0.9` | 50 | 50 |
| 4 | CAB2 | 3 | `theta=0.9` | 50 | 50 |
| 5 | SG | 1 | `shift=1/1000, scale=1.0` | 50 | 100 |
| 6 | SG | 3 | `shift=1/1000, scale=1.0` | 50 | 100 |
| 7 | SG-prev | 1 | `scale=0.05` | 50 | 50 |
| 8 | SG-prev | 3 | `scale=0.05` | 50 | 50 |

每组生成 10,000 张图，计算 FID-10k。所有组必须使用完全相同的标签、初始噪声和样本 ID。

> 本实验固定的是 50 个状态更新步，而不是固定算力。SG 每步需要两个时间分支，因此约为 100 NFE；其余方法为 50 NFE。

## 2. 必须保留的 JiT 官方设置

- 模型：JiT-B/16。
- 分辨率：256×256。
- 采样步数：50。
- 初始噪声尺度：`noise_scale=1.0`。
- 速度分母截断：`t_eps=0.05`。
- CFG interval：`(0.1, 1.0)`，保持 JiT 官方严格边界判断。
- 使用同一个 EMA checkpoint。
- 不要在 `t=1` 调用模型。

JiT 的路径约定为

$$
x_t=(1-t)\epsilon+t x_{\mathrm{data}},\qquad t:0\rightarrow1.
$$

因此：

- `t=0` 是噪声；
- `t=1` 是数据；
- 时间网格为

$$
t_i=\frac{i}{50},\qquad i=0,\ldots,50;
$$

- 步长为

$$
h=t_{i+1}-t_i=\frac1{50}=0.02.
$$

## 3. 模型输出、速度和 CFG

JiT 网络输出 clean prediction。对于条件和无条件分支：

$$
\hat x_i^c=f_\theta(x_i,t_i,c),\qquad
\hat x_i^u=f_\theta(x_i,t_i,\varnothing).
$$

先分别转成速度：

$$
v_i^c=\frac{\hat x_i^c-x_i}{\max(1-t_i,0.05)},
\qquad
v_i^u=\frac{\hat x_i^u-x_i}{\max(1-t_i,0.05)}.
$$

然后做 CFG：

$$
v_i=v_i^u+w_{\mathrm{cfg}}(v_i^c-v_i^u).
$$

CFG 仅在以下条件成立时启用：

```python
cfg_active = (t < 1.0) & (t > 0.1)
```

区间外令 `cfg_scale=1.0`。CFG=1 时可以只计算 conditional 分支作为等价优化，但必须确认结果与上述公式一致。

实现一个统一函数：

```python
guided_velocity(x, t, labels, cfg_scale) -> (velocity, clean_prediction)
```

其中 `velocity` 是做完 CFG 后的原始速度。AB2 和 SG-prev 的历史缓存必须缓存这个值，而不是缓存经过 AB/SG 修正后的速度。

## 4. AB2

### 4.1 更新规则

第 0 步没有历史速度，使用 Euler：

$$
x_1=x_0+h v_0.
$$

从第 1 步开始使用 AB2：

$$
\bar v_i=\frac32v_i-\frac12v_{i-1},
$$

$$
x_{i+1}=x_i+h\bar v_i.
$$

### 4.2 缓存规则

每步更新后执行：

```python
v_prev = v_i
```

不得缓存 `bar_v_i`。

### 4.3 最后一步

最后一次模型调用发生在 `t=0.98`。继续使用 AB2 从 `0.98` 更新到 `1.0`，不要在 `t=1` 调用模型，也不需要把 AB2 强制改成 Euler。

## 5. CAB2

CAB2 使用 rectified coordinates。定义：

$$
\lambda_i=\frac{1-t_i}{t_i},\qquad
y_i=\frac{x_i}{t_i}.
$$

由于 `t=0` 时 `lambda` 和 `y` 奇异，不能从第 0 步直接使用 CAB 坐标。

### 5.1 启动规则

- 第 0 步 `t=0 -> 0.02`：在原始 `t` 坐标中使用 Euler。
- 第 1 步：使用原始 `t` 坐标中的 AB2。
- 收集到足够历史后，进入 variable-step CAB2。

### 5.2 CAB 使用的导数量

由 JiT 的 clean prediction 和截断后的速度构造：

$$
e_i=\hat x_i-v_i.
$$

这里的 `v_i` 是做完 CFG 后的速度；`hat_x_i` 也应是与其一致的 CFG clean prediction。可以直接先做 clean-pred CFG：

$$
\hat x_i^{\mathrm{cfg}}
=\hat x_i^u+w_{\mathrm{cfg}}(\hat x_i^c-\hat x_i^u),
$$

再计算

$$
v_i=\frac{\hat x_i^{\mathrm{cfg}}-x_i}{\max(1-t_i,0.05)},
\qquad
e_i=\hat x_i^{\mathrm{cfg}}-v_i.
$$

不要在最后几步用未经截断的 `(1-t)` 重新计算噪声，否则会破坏 JiT 官方的 `t_eps=0.05` 规则。

### 5.3 Variable-step CAB2

由于 JiT 使用 uniform-`t` 网格，而 `lambda=(1-t)/t` 不等间隔，必须使用 variable-step 公式。

定义：

$$
k_i=\lambda_{i+1}-\lambda_i,
\qquad
r_i=\frac{k_i}{k_{i-1}}.
$$

采样时 `lambda` 递减，因此 `k_i<0`，公式中不需要再手动添加负号。

CAB2 predictor：

$$
y_{i+1}^{p}
=y_i+k_i\left[
\left(1+\frac{r_i}{2}\right)e_i
-\frac{r_i}{2}e_{i-1}
\right].
$$

历史外推：

$$
r_{i-1}=\frac{k_{i-1}}{k_{i-2}},
$$

$$
e_i^{\mathrm{ext}}
=(1+r_{i-1})e_{i-1}-r_{i-1}e_{i-2}.
$$

defect correction：

$$
y_{i+1}
=y_{i+1}^{p}
+\theta k_i(e_i-e_i^{\mathrm{ext}}),
\qquad \theta=0.9.
$$

转回原坐标：

$$
x_{i+1}=t_{i+1}y_{i+1}.
$$

### 5.4 最后一步

CAB2 可以从 `t=0.98`、`lambda>0` 更新到 `t=1`、`lambda=0`。不要在 `t=1` 调用模型。最后一步继续使用 CAB2，不要因为 JiT 官方 Heun 最后使用 Euler 就自动把 CAB2 也改成 Euler。

## 6. SG

SG 在同一个状态 `x_i` 上，用真实时间和一个更噪的时间分别计算速度。

JiT 的方向是 `t: 0 -> 1`，因此更噪时间是更小的时间：

$$
t_i^-=\max(t_i-0.001,0).
$$

分别计算：

$$
v_i=v(x_i,t_i),
$$

$$
v_i^-=v(x_i,t_i^-).
$$

两个分支都必须独立应用相同的 CFG 规则。shifted 分支的速度分母使用自己的时间：

$$
v_i^-
=\frac{\hat x^{\mathrm{cfg}}(x_i,t_i^-)-x_i}
{\max(1-t_i^-,0.05)}.
$$

SG scale 固定为：

$$
\omega_{\mathrm{SG}}=1.0.
$$

组合速度：

$$
\tilde v_i
=v_i+\omega_{\mathrm{SG}}(v_i-v_i^-).
$$

用 Euler 更新：

$$
x_{i+1}=x_i+h\tilde v_i.
$$

在 `t=0` 时有 `t_i^-=t_i=0`，所以 SG 修正自然为零。最后一步在 `t=0.98` 计算两个分支后直接更新到 `t=1`，不要在终点调用模型。

## 7. SG-prev

SG-prev 复用上一个采样点的原始 CFG 速度：

$$
\tilde v_i
=v_i+\omega_{\mathrm{prev}}(v_i-v_{i-1}).
$$

### 7.1 Scale 的确定

SG 的时间偏移为：

$$
\delta_{\mathrm{SG}}=0.001.
$$

50-step SG-prev 的相邻采样点时间差为：

$$
\delta_{\mathrm{prev}}=0.02.
$$

使用局部近似

$$
v(t)-v(t-\delta)\approx
\delta\frac{\partial v}{\partial t},
$$

为了让 SG 和 SG-prev 的 correction 大小近似相同，令

$$
\omega_{\mathrm{prev}}\delta_{\mathrm{prev}}
=\omega_{\mathrm{SG}}\delta_{\mathrm{SG}}.
$$

因此：

$$
\omega_{\mathrm{prev}}
=1.0\times\frac{0.001}{0.02}
=0.05.
$$

本实验固定：

$$
\boxed{\omega_{\mathrm{prev}}=0.05}.
$$

### 7.2 更新和缓存

第 0 步没有历史值：

$$
\tilde v_0=v_0.
$$

之后：

$$
x_{i+1}=x_i+h\tilde v_i.
$$

缓存顺序应等价于：

```python
v_i = guided_velocity(x_i, t_i, labels, cfg_scale)

if v_prev is None:
    v_used = v_i
else:
    v_used = v_i + 0.05 * (v_i - v_prev)

x_next = x_i + h * v_used
v_prev = v_i
```

严禁执行 `v_prev = v_used`，否则 guidance 会递归积累，不再是 SG-prev。

## 8. JiT 终点截断

JiT 官方速度为：

$$
v(x,t)=\frac{\hat x-x}{\max(1-t,t_{\mathrm{eps}})},
\qquad t_{\mathrm{eps}}=0.05.
$$

50-step 网格的最后一次模型调用发生在：

$$
t_{49}=0.98.
$$

因此最后一次速度实际使用：

$$
v_{49}
=\frac{\hat x_{49}-x_{49}}
{\max(0.02,0.05)}
=\frac{\hat x_{49}-x_{49}}{0.05}.
$$

然后从 `t=0.98` 更新到 `t=1.0`。任何方法都不得在 `t=1` 调用模型。

JiT 官方 Heun 把最后一段改成 Euler，是因为 Heun 需要在终点再次计算速度。AB2、CAB2 和 SG-prev 不需要终点速度，所以不需要因为这个原因降阶。SG 的最后一段也只在 `t=0.98` 和 `t=0.979` 计算，然后 Euler 到终点。

## 9. ImageNet-10k 生成协议

### 9.1 标签

- 共 1000 类。
- 每类恰好生成 10 张，共 10,000 张。
- 预先生成并保存固定的 `(sample_id, class_id, seed)` 清单。
- 八组实验读取同一份清单，不能让各方法独立随机抽标签。

### 9.2 初始噪声

- 每个 `sample_id` 使用固定 seed 生成初始噪声。
- 初始噪声必须与分布式 world size、batch size 和数据分片方式无关。
- 推荐按全局 `sample_id` 派生 seed，而不是依赖每张 GPU 当前的 RNG 状态。
- 八组实验的同一个 `sample_id` 必须拥有逐元素一致的初始噪声。

### 9.3 输出

- 使用相同的反归一化、量化和 PNG 保存流程。
- 文件名包含固定宽度的全局 sample ID，例如 `000000.png`。
- 保存每组完整配置、Git commit、checkpoint 标识和实际 NFE。
- 生成结束后检查文件数、重复文件名、损坏图片和类别计数。

### 9.4 FID

- 计算 FID-10k。
- 所有方法使用同一份 ImageNet reference statistics。
- 在结果中明确写 `FID-10k`，不要直接与论文中的 FID-50k 混合比较。
- 若后续需要论文级最终结果，再对优胜配置补跑 50k。

## 10. 推荐命令行参数

建议新增以下参数，具体名称可按现有工程风格调整：

```text
--sampling_method {ab2,cab2,sg,sg_prev}
--num_sampling_steps 50
--num_images 10000
--cfg {1.0,3.0}
--interval_min 0.1
--interval_max 1.0
--t_eps 0.05
--cab_theta 0.9
--sg_shift 0.001
--sg_scale 1.0
--sg_prev_scale 0.05
--eval_seed 0
```

## 11. 实现验收检查

正式生成 10k 之前，至少完成以下检查：

1. **官方回归检查**：未修改的 Euler/Heun 在相同 seed 下仍能复现原输出。
2. **CFG=1 检查**：CFG=1 与只运行 conditional 分支的结果一致。
3. **AB2 启动检查**：第 0 步严格等于 Euler，从第 1 步才启用 AB2。
4. **缓存检查**：AB2 和 SG-prev 缓存的是原始 CFG 速度，不是修正后速度。
5. **SG 时间方向检查**：使用 `t-0.001`，不能写成 `t+0.001`。
6. **SG 分母检查**：shifted 分支使用 `max(1-t_shifted, 0.05)`。
7. **SG-prev scale 检查**：主实验为 `0.05`，不是 `1.0`。
8. **CAB 步长检查**：`k=lambda_next-lambda_current<0`，不要额外改变符号。
9. **终点检查**：日志中不得出现模型在 `t=1.0` 的 forward。
10. **最后分母检查**：`t=0.98` 时分母必须是 `0.05`。
11. **公平性检查**：八组的标签和初始噪声逐样本一致。
12. **小规模冒烟测试**：先为每种方法生成相同的 16 或 64 张图，确认没有 NaN、Inf、明显饱和或文件覆盖，再启动 10k。

## 12. 结果表模板

| 方法 | CFG | Steps | NFE | FID-10k | images/s | 总耗时 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| AB2 | 1 | 50 | 50 | | | | |
| AB2 | 3 | 50 | 50 | | | | |
| CAB2 | 1 | 50 | 50 | | | | `theta=0.9` |
| CAB2 | 3 | 50 | 50 | | | | `theta=0.9` |
| SG | 1 | 50 | 100 | | | | `shift=0.001, scale=1` |
| SG | 3 | 50 | 100 | | | | `shift=0.001, scale=1` |
| SG-prev | 1 | 50 | 50 | | | | `scale=0.05` |
| SG-prev | 3 | 50 | 50 | | | | `scale=0.05` |

## 13. 参考实现

- JiT 官方仓库：<https://github.com/LTH14/JiT>
- Self-Guidance 官方仓库：<https://github.com/maple-research-lab/Self-Guidance>
- Self-Guidance 论文：<https://arxiv.org/abs/2412.05827>
- CAB 官方仓库：<https://github.com/Anuska-Roy/CAB>
- CAB 论文：<https://arxiv.org/abs/2605.16736>


## 14. 本项目落地配置（2026-10-10）

- 独立脚本位于本目录；复用 `/mnt/data/LONG/hongxu/zjiaak/envs/fds/bin/python`。
- 正式每卡 batch=128，四方法×CFG1/3，共八个新组，一卡一组。
- 只读复用 `HDD/fds/runs/baseline8_20261007_cfg13/heun_cfg1` 和 `heun_cfg3` 的10k基线，不重新生成或重算旧指标。
- 新checkpoint使用 `HDD/checkpoints/jit-b-16/checkpoint-last.pth`，与旧Heun SHA256一致。
- 共享输入沿用 `HDD/fds/templates/baseline8_10k_seed0.npy`；每类10张、原sample ID和逐元素同噪声。
- 统一BF16/Flash/eager、FP32状态、原PNG round/clip；CFG1仍合批计算条件/无条件分支。
- FID/IS对齐旧baseline8：reference/generated统计float64、IS rng_seed=0、10 splits。
- CAB2第0步Euler、第1/2步AB2，i=3(t=.06)起使用variable-step CAB2，避免lambda0奇异历史步长。
- 正式输出 `HDD/cfg/runs/formal_<香港时间戳>`；烟测 `HDD/cfg/smoketest/smoke_<时间戳>`，每次新建目录。
- 旧Heun首个128 batch只做PNG像素回归核验；八个新组各128张、bs128、50步烟测。
- 用户已授权核验并停止take_gpu，烟测/正式生成/算分全部结束后由服务器tmux控制器恢复并核验PID。
- 运行入口与详细说明见本目录README.md及config.json；控制脚本只操作确认过的占卡PID，不停止其他程序。

## 15. 用户确认的补充对照（2026-10-11）

按 config_followup_20261011.json 运行八组，每组10k、batch128、CFG1/3：
Euler50（50 NFE）、Heun25（49 NFE）、SG25（50 NFE）、CAB2 theta=0/50（50 NFE）。
除步数和CAB theta外，沿用同checkpoint、噪声bank、标签、BF16、PNG和FID/IS配置。
CAB启动和截断、SG shift=.001/scale=1保持原值。每次整轮输出到同一个新run目录。
本轮GPU有其他任务但空闲显存充足，共享运行，不停止或恢复任何占卡进程；耗时单独标记。
先做八组各128张/bs128烟测，通过后正式生成并算分。


## 16. 100 NFE速度几何实验（2026-10-11）
用户确认交替Heun-flip；AB2/CAB2 theta=.9各100步，SG shift=.05/scale1各50步，Heun-flip50步/99NFE；各CFG1/3，共八组，每组10k、每卡bs512。按config_geometry_20261011.json，沿用同EMA1/噪声/标签/BF16/Flash/eager/FP32状态及评分配置。条件/无条件基础速度在CFG1和3均计算并逐样本记录L2/RMS、夹角、余弦、大小比及差值；SG两时间分支独立记录。记录原始历史、组合及实际更新速度、修正平行/垂直分量、与CFG方向关系；CAB另记e历史、predictor/defect与坐标映回贡献。零向量角度为缺失，按实际t汇总分位数。所有统计复用已有forward，无额外NFE；耗时含统计。无共同轨迹额外诊断。详情见README.md。17项CPU单元测试通过。每组满bs512烟测通过后正式运行；不结束其他用户的服务，不自动改batch，不覆盖旧结果。
