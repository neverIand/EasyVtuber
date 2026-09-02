# EasyVtuber 当前版与官方最新版整体性能比较（2026-09-02）

## 1. 结论先行

本轮停止继续修改运行时代码，只对当前项目和官方最新版做可重复的整体运行路径测试。结果如下：

- 当前版与官方版的固定检查帧、预处理图像和姿态结果保持一致，没有发现画面、透明度、颜色或动作精度变化。
- TensorRT 完成了 3600 帧稳态、900 帧热缓存回放和 300 帧强制无缓存测试。当前版稳态推理均值快约 13.7%，P95 快约 16.7%，强制无缓存均值快约 7.0%，热缓存端到端快约 84.7%。
- DirectML 已正确选择 RTX 4060，不再落到 Intel 核显。300 帧可比较窗口中，当前版稳态推理均值快约 23.2%，P95 快约 3.8%，热缓存端到端快约 84.2%。均值收益大于 P50/P95，主要来自 raw RAM 缓存命中不再做 Brotli 解压。
- iFacialMocap/姿态简化路径改善明显：当前版首次简化约 0.126 ms，官方版约 1977 ms；5000 帧批处理约 33.5 ms 对 245.2 ms，当前快约 86.3%。5000 帧归一化结果哈希一致。
- 当前版 raw RAM 缓存以空间换延迟。TensorRT 长测峰值工作集约 3.994 GiB，官方 Brotli 模式约 2.728 GiB，当前高约 46.4%。这直接支持后续在启动器中提供清晰的缓存容量与 raw/Brotli 模式选择。
- 这台机器当前的主要限制不是帧率，而是温度。当前与官方 DirectML 长测均达到 80°C 后被安全终止；当前 TensorRT 虽完整完成，但峰值温度已到 78°C。考虑到用户已怀疑显卡硬件状态，不应继续提高温度阈值或做更激进的压力测试。

因此，现阶段没有证据支持继续做低层性能改写。下一步应先处理散热/硬件稳定性，再决定是否继续 RIFE、SR、Spout 或 TensorRT runtime cache 等高复杂度方向。

## 2. 对照版本

| 项目 | 提交 | 说明 |
|---|---|---|
| 当前主项目 | `7a618796ff638a65218a74c4841a7a2fe452e802` | `codex/gpu-safety-90`，最后提交为 `Select the correct DirectML adapter` |
| 当前运行时 | `43d67b830b0e003c43acf39723c6733a3c9fe491` | `codex/trt-cache-safety`，最后提交为 `Disable unstable TensorRT runtime cache` |
| 官方主项目 | `f7dd2de4df93c878b0171f47346ff66414a863e6` | 2026-02-12，官方 `main` 当日最新状态，GitHub 页面显示 328 commits |
| 官方运行时子模块 | `07c302921e67b483b023a7db5c0fa96ac5fd8e6d` | 官方主项目固定的 `ezvtuber-rt` 提交，GitHub 页面显示 143 commits |

官方仓库：<https://github.com/yuyuyzl/EasyVtuber>

官方运行时：<https://github.com/yuyuyzl/ezvtuber-rt>

没有 push，没有创建 pull request，也没有修改官方工作树。

## 3. 测试范围与固定设置

### 3.1 当前运行设置

| 项目 | 设置 |
|---|---|
| 角色图 | `data/images/lambda_00.png` |
| 模型 | THA3 separable FP32 |
| 眉毛 | 关闭 |
| RIFE | 关闭 |
| 超分/SR | 关闭 |
| 帧率 | 30 FPS |
| 姿态简化 | Medium / `simplify=2` |
| RAM 缓存 | 2 GiB；当前版 `raw`，官方版原生 Brotli |
| VRAM 缓存 | 2 GiB |
| 输出后处理 | Debug/OpenCV 对应的 BGR 路径，扩展移动开启 |
| 输入 | 项目内 5000 帧 iFacialMocap 姿态流回放 |
| TensorRT runtime JIT cache | 当前版关闭；`.trt` 内容寻址引擎缓存保留 |
| GPU 持续占空目标 | 90% |
| 温度停止线 | 80°C |

### 3.2 覆盖的运行路径

测试覆盖：

1. 姿态简化模块导入、首次调用和 5000 帧批处理。
2. 透明像素清理与 BGRA 图像预处理。
3. 后端导入、DirectML 适配器发现与选择。
4. TensorRT/DirectML 核心初始化、`setImage` 和预热。
5. 30 FPS 有节拍稳态运行。
6. RAM 缓存命中回放。
7. 关闭帧缓存后的模型推理诊断。
8. Debug + 扩展移动后处理。
9. 最终 BGR 帧与原始模型输出的 SHA-256 检查点。
10. 进程工作集、GPU 利用率、显存、功率与温度。

不包含真实摄像头/iPhone 网络抖动、实际 OpenCV 窗口绘制、OBS/Spout/VirtualCam 接收、RIFE、SR、THA4 student 或启动器 UI 响应时间。它是完整核心数据路径基准，不是外设与显示系统的端到端录屏测试。

## 4. 测试矩阵

| 组合 | 用途 | 长测结果 |
|---|---|---|
| 当前 TensorRT | 当前配置主路径 | 完成 3600 + 900 + 300 帧 |
| 官方 TensorRT | 官方代码对照 | 完成 3600 + 900 + 300 帧 |
| 当前 DirectML Auto | 验证修复后的自动独显选择 | 长测在 80°C 停止；另保留 300 + 300 帧完整统计 |
| 官方 DirectML native | 官方默认启动行为 | 长测在 80°C 停止；另保留 300 + 300 帧完整统计 |
| 官方 DirectML same-device | 绕过官方无关的 CUDA eager import，固定同一 RTX/DXGI 设备 | 长测在 80°C 停止；300 + 300 + 100 帧补测完整完成 |

官方运行时会在每次 TensorRT 启动时重建五个引擎。为避免不可细分的高负载构建，本测试把同一 ONNX、同一 GPU、同一 TensorRT 版本下已验证的引擎字节注入官方 `build_engine` 返回值。官方代码仍执行临时写盘、反序列化、上下文创建、内存分配和推理，因此官方启动时间是偏有利的下界，不代表真实冷启动重建耗时。

官方原生包还会在 DirectML 启动时先初始化 CUDA。`native` 组合保留该行为；`same-device` 组合只隔离这段无关 eager import，再加载未修改的官方 `CoreORT`，用于确认相同 RTX 设备下的代码差异。

## 5. 安全流程

- 每个项目/后端组合使用全新的 Python 子进程，避免 DirectML、CUDA、模型缓存和导入顺序互相污染。
- 测试脚本通过 `nvml.dll` 直接采样，从未启动 `nvidia-smi`。每轮结束后检查 Python、`pythonw` 和 `nvidia-smi`，均没有残留进程。
- 每帧推理使用 90% 活跃时间占空限制器；30 FPS 节拍通常会提供更多空闲时间。
- 连续两次 0.5 秒 NVML 采样超过 90% 时立即停止；温度达到 80°C 时立即停止。
- 90% 是持续占空目标，不是显卡驱动层硬上限。当前 TensorRT 记录到一次 94% 瞬时峰值，官方 DirectML 长测记录到一次 100% 瞬时峰值；都没有连续两次超过 90%，但后者随后因温度停止。这说明软件节拍无法保证不可细分推理调用内的瞬时利用率绝不越过 90%。
- TensorRT 引擎构建被禁止；缓存不匹配会报错，不会自动重建。
- 组合之间进行 15 秒冷却；DirectML 补测之间提高到 60 秒。即使从 45°C 空闲温度起步，较长 DirectML 窗口仍会达到 80°C。

## 6. 可重复脚本与命令

测试脚本：`benchmarks/performance_compare.py`

脚本有两个入口：

- `matrix`：为每个组合创建隔离子进程，保存 JSON/日志并汇总比较。
- `worker`：只运行单一组合，主要由 `matrix` 调用。

可用 `--cases` 只选择部分组合；`--continue-on-error` 会保存温度/超时失败并继续剩余组合；`--case-timeout-seconds` 防止原生运行库永久卡住。每个组合使用独立 `TEMP/TMP`，当前 TensorRT 缓存目录通过 `EZVTB_TRT_CACHE_DIR` 显式固定，避免官方旧式临时缓存污染当前缓存迁移路径。

正式长测命令：

```powershell
$pythonPath = 'D:\_Projects\.Codex\projectless\2026-08-30\new-chat\EasyVtuber-runtime-v0.8.0\EasyVtuber\envs\python_embedded\python.exe'
$currentRoot = 'D:\_Projects\.Codex\projectless\2026-08-30\new-chat\EasyVtuber-updated'
$officialRoot = 'D:\_Projects\.Codex\projectless\2026-09-01\nvidia-smi\EasyVtuber-official-f7dd2de'

& $pythonPath -B "$currentRoot\benchmarks\performance_compare.py" matrix `
  --current-root $currentRoot `
  --official-root $officialRoot `
  --python $pythonPath `
  --model-root 'D:\_Projects\.Codex\projectless\2026-08-30\new-chat\EasyVtuber-runtime-v0.8.0\EasyVtuber\data\models' `
  --image-path "$currentRoot\data\images\lambda_00.png" `
  --pose-path "$currentRoot\ezvtuber-rt\test\data\pose_20fps.json" `
  --engine-cache 'C:\Users\90833\AppData\Local\EasyVtuber\trt-cache' `
  --output-dir "$currentRoot\benchmark_results\2026-09-02" `
  --continue-on-error `
  --stress-frames 3600 `
  --official-native-stress-frames 450 `
  --cache-replay-frames 900 `
  --cache-replay-span 300 `
  --no-cache-frames 300 `
  --fps 30 `
  --duty 90 `
  --max-gpu-utilization 90 `
  --max-temperature 80 `
  --cooldown-seconds 15 `
  --case-timeout-seconds 420
```

DirectML 可比较窗口使用同一命令，只增加以下选择并把输出放到 `directml-metrics-window`：

```powershell
--cases current-directml-auto official-directml-native official-directml-same-device `
--stress-frames 300 `
--official-native-stress-frames 300 `
--cache-replay-frames 300 `
--cache-replay-span 150 `
--no-cache-frames 100 `
--cooldown-seconds 60 `
--case-timeout-seconds 120
```

## 7. TensorRT 结果

| 指标 | 当前版 | 官方版 | 解读 |
|---|---:|---:|---|
| 运行时导入 | 1.180 ms | 283.890 ms | 当前延迟导入明显更轻 |
| 核心初始化 | 13.455 s | 12.719 s | 官方另有 1.413 s 的外部 90% 冷却；调整后当前略快 |
| 图像预处理 | 24.04 ms | 291.24 ms | 当前快约 91.7% |
| `setImage` | 191.89 ms | 213.10 ms | 当前略快 |
| 预热 | 320.71 ms | 329.28 ms | 基本相当 |
| 3600 帧稳态推理均值 | 14.192 ms | 16.437 ms | 当前快约 13.7% |
| 稳态推理 P95 / P99 | 15.956 / 17.053 ms | 19.151 / 20.819 ms | 当前尾延迟更低 |
| 稳态端到端均值 | 14.875 ms | 17.294 ms | 当前快约 14.0% |
| 33.33 ms 截止违约 | 11 / 3600 | 9 / 3600 | 都能稳定维持 30 FPS |
| 实际节拍 FPS | 29.999 | 29.999 | 相同 |
| 900 帧热缓存推理均值 | 0.0156 ms | 2.4399 ms | 当前 raw 命中不做 Brotli 解压 |
| 热缓存端到端均值 | 0.4664 ms | 3.0563 ms | 当前快约 84.7% |
| 300 帧强制无缓存均值 | 14.479 ms | 15.566 ms | 当前快约 7.0% |
| 强制无缓存 P95 | 14.591 ms | 16.434 ms | 当前更稳 |
| GPU 平均 / 瞬时峰值 | 37.63% / 94% | 35.33% / 85% | 当前平均略高；94% 仅单次采样 |
| 温度峰值 | 78°C | 76°C | 均未触发 80°C 停止，但当前余量很小 |
| 显存峰值 | 3.219 GiB | 3.223 GiB | 基本一致 |
| 进程峰值工作集 | 3.994 GiB | 2.728 GiB | 当前 raw 缓存高约 46.4% |

稳态阶段有 190 次 RAM 命中、3411 次未命中；热缓存阶段的 900 帧全部命中。当前版的两类收益可以分开看：强制无缓存快约 7% 来自推理/缓冲路径优化；热缓存快约 85% 来自 raw 无损缓存避免同步 Brotli 解压。

旧文档中的约 1.2 秒 TensorRT 初始化数据来自 runtime JIT cache 尚可用的历史状态。本轮当前配置明确关闭不稳定的 runtime cache，因此约 13.5 秒的上下文初始化才是现在应采用的基线。

## 8. DirectML 结果

三个组合都实际使用 RTX 4060：

- 当前 Auto 枚举为 `device 0 = Intel UHD`、`device 1 = RTX 4060`，自动选择 1。
- 官方 native 因先初始化 CUDA，枚举顺序为 `device 0 = RTX 4060`、`device 1 = Intel UHD`，官方默认 0 实际落到 RTX。
- 官方 same-device 隔离 eager CUDA 后恢复与当前相同枚举，并显式选择 1，也落到 RTX。

因此本表不是“RTX 对 Intel”，而是同一 RTX 上的代码路径比较。

| 指标 | 当前 Auto | 官方 native | 官方 same-device |
|---|---:|---:|---:|
| 运行时导入 | 50.19 ms | 319.21 ms | 50.29 ms |
| 核心初始化 | 2513.33 ms | 2481.91 ms | 2645.91 ms |
| 图像预处理 | 26.36 ms | 285.29 ms | 260.60 ms |
| `setImage` | 18.40 ms | 16.79 ms | 62.08 ms |
| 预热 | 37.58 ms | 39.75 ms | 177.90 ms |
| 300 帧稳态推理均值 | 9.578 ms | 12.468 ms | 13.056 ms |
| 稳态 P50 | 17.360 ms | 17.560 ms | 18.350 ms |
| 稳态 P95 | 19.148 ms | 19.909 ms | 22.120 ms |
| 稳态端到端均值 | 10.353 ms | 13.475 ms | 14.057 ms |
| 33.33 ms 截止违约 | 1 / 300 | 0 / 300 | 0 / 300 |
| 实际节拍 FPS | 29.980 | 29.993 | 29.997 |
| 300 帧热缓存推理均值 | 0.0127 ms | 2.0881 ms | 2.1896 ms |
| 热缓存端到端均值 | 0.4226 ms | 2.6761 ms | 2.7807 ms |
| 该补测最终状态 | 无缓存尾段到 80°C | 无缓存尾段到 80°C | 完整完成，峰值 76°C |

300 帧稳态包含 149 次命中和 152 次未命中，因此当前版均值比官方快约 23.2%，而 P50/P95 分别只快约 1.1%/3.8%。这不是矛盾：raw 缓存把约一半的命中帧降到近乎零开销，未命中的实际 DML 模型延迟仍大致在 17–20 ms 范围。截图中约 266 ms/帧、3.4 FPS 的问题在正确选择 RTX 后未复现。

## 9. iFacialMocap 与姿态简化

| 指标 | 当前版 | 官方版 | 结果 |
|---|---:|---:|---|
| 姿态简化模块导入 | 6.025 ms | 3.301 ms | 导入本身都很小 |
| 首次简化调用 | 0.126 ms | 1977.067 ms | 当前不再触发旧重依赖/JIT 链 |
| 5000 帧简化 | 33.494 ms | 245.167 ms | 当前快约 86.3% |
| 唯一简化姿态 | 4810 | 4810 | 相同 |
| 归一化零值哈希 | 相同 | 相同 | 包含 `+0.0/-0.0` 归一化后逐值一致 |

测试使用录制流回放，不包含手机网络延迟；它验证的是 iFacialMocap 数据进入模型前的解析后姿态简化热路径。已有单元/连续输入测试继续负责协议容错与缓冲复用。

## 10. 精度与视觉结果

| 路径 | 验证结果 |
|---|---|
| TensorRT 3600 帧稳态原始模型检查点 | 当前与官方 SHA-256 相同 |
| TensorRT 3600 帧最终 BGR 检查点 | 当前与官方 SHA-256 相同 |
| TensorRT 300 帧强制无缓存原始/最终检查点 | 当前与官方 SHA-256 相同 |
| DirectML 300 帧稳态原始/最终检查点 | 当前、官方 native、官方 same-device 三者相同 |
| DirectML 300 帧热缓存原始/最终检查点 | 三者相同 |
| 预处理图像 | 当前与官方相同 |
| 5000 帧姿态 | 归一化后相同 |

没有发现模型输出、颜色、alpha、扩展移动或动作数值变化。本轮未实际连接 Spout/OBS/VirtualCam，因此不能用这些哈希替代外部接收端的通道/alpha 验收。

## 11. 热稳定性与 90% 限制

主长测安全结果：

| 组合 | GPU 平均 / 峰值 | 温度峰值 | 结果 |
|---|---:|---:|---|
| 当前 TensorRT | 37.63% / 94% | 78°C | 完整完成 |
| 官方 TensorRT | 35.33% / 85% | 76°C | 完整完成 |
| 当前 DirectML | 43.01% / 89% | 80°C | 安全停止 |
| 官方 DirectML native | 25.90% / 100% | 81°C | 温度采样越过停止线后安全停止 |
| 官方 DirectML same-device | 40.03% / 73% | 80°C | 安全停止 |

DirectML 组合是顺序运行，虽然有冷却，起始温度仍不完全相同，所以这些数字不适合用来判断哪份代码“更热”。它们只能证明：在这台机器上，90% 持续占空目标仍不足以避免长时间 DirectML 达到 80°C。

在确认散热/硬件状况前，建议实际使用时优先采用 24 FPS，并把 GPU 占空目标从 90% 下调到 70–80% 观察；这会影响吞吐余量，但不改变模型精度或单帧图像。不要把温度阈值提高来换取测试完成。

## 12. 测试产物与临时缓存说明

保留的原始结果：

- `benchmark_results/2026-09-02/comparison.json`
- `benchmark_results/2026-09-02/current-tensorrt-standard.json/.log`
- `benchmark_results/2026-09-02/official-tensorrt-standard.json/.log`
- `benchmark_results/2026-09-02/current-directml-auto.json/.log`
- `benchmark_results/2026-09-02/official-directml-native.json/.log`
- `benchmark_results/2026-09-02/official-directml-same-device.json/.log`
- `benchmark_results/2026-09-02/directml-metrics-window/` 下的补测 JSON 与日志

烟雾测试早期暴露出一个测试隔离问题：官方原版会把五个无版本名引擎写到 `%TEMP%\ezvtuber_rt_engines`，当前版随后把它们识别为旧缓存并尝试迁移，导致缓存路径解析等待。正式脚本已通过每组合独立 `TEMP/TMP` 和显式 `EZVTB_TRT_CACHE_DIR` 修复，正式数据不受影响。

早期烟雾测试生成的以下目录仍存在，因为当前 Codex 沙箱没有权限删除用户临时目录：

`C:\Users\90833\AppData\Local\Temp\ezvtuber_rt_engines`

其中只有本轮官方烟雾测试在 2026-09-02 07:43 左右生成的五个 `.trt`，合计约 210 MiB。确认 EasyVtuber 已关闭后可安全删除整个目录。正式持久化缓存 `C:\Users\90833\AppData\Local\EasyVtuber\trt-cache` 没有被删除或替换。

## 13. 后续判断

1. 暂停继续做运行时性能优化；现有当前配置已经稳定满足 30 FPS 的计算预算。
2. 优先检查显卡散热、风扇、灰尘、硅脂/液金状态、电源模式和驱动稳定性。
3. 本报告完成后的启动器 v2 已把安全预设、FPS、持续占空目标以及 RAM/VRAM 用途、容量估算和 Raw/Brotli 取舍集中到“性能与安全”页；这项 UI 后续不改变本报告中的模型、像素或性能数据。
4. 保留 24 FPS、较低 GPU 占空目标作为这台机器的保守运行方案。
5. RIFE、SR、Spout、跨引擎融合和 runtime JIT cache 不在本轮继续实施；待硬件稳定后再按 `PERFORMANCE_OPTIMIZATION_TODO.md` 重新评估。
