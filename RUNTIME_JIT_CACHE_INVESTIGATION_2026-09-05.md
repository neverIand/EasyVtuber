# runtime JIT cache 调查（2026-09-05）

> **首轮只读调查归档。** 下文保留 GPU 验证获准之前的证据与判断；后续代码修复、短时实机验证和手动试用步骤见 [修复与验证报告](RUNTIME_JIT_CACHE_REPAIR_2026-09-05.md)。用户随后已确认散热问题解决并授权短时 GPU 验证。

目前应维持 runtime cache 默认关闭。历史回退有依据，但现有证据还不能把原生阻塞锁定为单一根因。最值得优先验证的是 PyCUDA 与 TensorRT-RTX 的 CUDA context 用法；“补一次 stream 同步”已经做过，而“换 EAGER 或升级版本就一定修好”均尚无实机证据。

本轮完成提交审计、实际 ONNX 形状与摘要检查、磁盘缓存身份检查、NVIDIA 官方资料核对，以及 51 项 CPU/mock 回归。新增只读审计脚本和结果；生产运行时代码、缓存文件与启动配置未修改。没有运行 GPU 推理、创建 CUDA context、反序列化 TensorRT 引擎或重建引擎。

## 1. 修改与回退过程

运行时子模块当前为 `43d67b830b0e003c43acf39723c6733a3c9fe491`（`codex/trt-cache-safety`）；调查开始时主项目为 `1a6b6cb`（`codex/gpu-safety-90`）。

| 提交 | 改动 | 与本问题的关系 |
|---|---|---|
| `914be3c`，9 月 1 日 | 缓存 `.trt` 引擎与 runtime JIT kernels | runtime cache 默认启用；context 创建后立刻保存，第一次异步 enqueue 后也立即保存 |
| `e96f942` | 稳定 GPU 缓存身份 | 修复键值随 CUDA 初始化顺序改变的问题 |
| `77c660c` | 减少重复缓存写入 | 序列化后比较字节，内容没变则跳过磁盘替换；仍然会调用原生 `serialize()` |
| `43d67b8`，9 月 2 日 | 默认关闭不稳定 runtime cache | 删除 context 创建后的立即保存；第一次推理完成并同步后再保存；runtime schema 从 1 升为 2；仅显式 opt-in 才启用 |

这次回退是隔离附加 runtime 缓存，内容寻址的 `.trt` 引擎缓存仍保留。更早的上游 `07c3029` 回退的是“并行加载引擎”，是另一件事；当前五个引擎仍串行构造。

历史记录包含两个阻塞位置：读取旧 cache 后 combiner 的 execution context 创建超过 90 秒；新 cache 的序列化也曾阻塞。当前同步修正并没有获得“完整新 cache 保存成功并跨进程重载稳定”的验收结果。上述阻塞来自[性能待办第 4.2 节](PERFORMANCE_OPTIMIZATION_TODO.md)，本轮没有重新复现，也没有取得可确认原生死锁原因的线程栈。

## 2. 本轮取得的实际证据

完整数据见 [runtime-cache-audit.json](benchmark_results/2026-09-05/runtime-cache-audit.json)。

| 项目 | 结果 |
|---|---|
| TensorRT-RTX | `1.3.0.35` |
| PyCUDA | `2025.1.2` |
| GPU / 当前驱动 | NVIDIA GeForce RTX 4060 Laptop GPU，CC 8.9 / `596.49` |
| 归档元数据时温度 | 49°C；NVML 单次只读采样，不是负载测试 |
| 五个 ONNX | SHA-256 全部与 9 月 2 日基准一致 |
| 五个内容寻址 `.trt` | 全部存在，当前身份重新计算的文件名全部命中 |
| runtime schema 1 | 五份全部存在，每份约 0.73–1.58 MB |
| runtime schema 2 | 对当前五个 ONNX/版本/GPU 身份计算出的文件全部不存在 |
| 本轮 CPU/mock 测试 | 51 项通过，0 失败 |

五个模型的所有实际输入维度都是固定整数：decomposer 为 `[512,512,4]`；其余图像张量与姿态张量均是固定 batch 1，没有符号维或 `-1`。代码中的 `is_dynamic_shape()` 检查首维是否为 `-1`，这些模型不会进入动态 profile 构建分支。

因此，现在设置 `EZVTB_TRT_RUNTIME_CACHE=1` 会走新 schema 的首次创建/保存路径，不能直接恢复旧测量中的 1–2 秒启动。旧 schema 文件缺乏内容完整性的原生验证，本轮仅确认身份、存在性和大小。

当前 `launcher.json` 保存的是 DirectML、30 FPS、80% 占空目标；这与 9 月 2 日 TensorRT 对照的设置不同。runtime JIT cache 不参与 DirectML 路径。

## 3. 原因判断与证据边界

### A. 优先排查：CUDA context 创建方式

[`ezvtb_rt/__init__.py`](ezvtuber-rt/ezvtb_rt/__init__.py) 第 42–43 行先 `cudaSetDevice(device_id)`，再导入 `pycuda.autoinit`。本机安装的 PyCUDA 源码显示，`autoinit` 通过 `make_default_context()` 调用 `Device.make_context()`，新建一个 context；`autoprimaryctx` 则 retain 并 push 设备的 primary context。

NVIDIA 的 [TensorRT-RTX 1.3 发布说明](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.3.html#known-issues) 明确建议与 PyCUDA 同用时选择 `pycuda.autoprimaryctx`，避免设备冲突。项目当前做法与该建议不一致，这是可定位到代码的具体问题，但还不能证明它就是两处 runtime-cache 阻塞的根因。

复测时还要保证 PyCUDA 与 `EZVTB_DEVICE_ID` 选择的是同一设备：PyCUDA 的自动选择读取 `CUDA_DEVICE` / `.cuda_device`，不会读取 `EZVTB_DEVICE_ID`。如果采用自动 primary-context 模块，应显式对齐设备选择；更直接的方式是对指定 `Device(device_id)` retain primary context，并管理 push/pop 生命周期。本机只有一个 CUDA GPU，此设备编号差异不是已证实的本机故障原因。

### B. 已修正：异步 enqueue 后过早序列化

旧实现第一次 `execute_async_v3()` 返回后立即调用 `serialize()`。NVIDIA 的[缓存工作流](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/work-with-runtime-cache.html) 在推理完成后再保存。当前 [`trt_engine.py`](ezvtuber-rt/ezvtb_rt/trt_engine.py) 第 221–226 行先同步提交推理的 stream，再保存；四项 engine mock 测试覆盖了默认关闭与保存顺序等行为。

这证明调用顺序的修正已在代码中，不能证明旧 blob 已损坏，也不能证明原生层稳定。`FakeStream`、`FakeContext` 与假的 runtime cache 不会模拟驱动、JIT 线程或原生锁。

### C. 相关资料，不能直接作为本机根因：动态形状后台编译

官方[动态形状说明](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/dynamic-shapes-advanced.html#setting-the-kernel-specialization-strategy) 将 LAZY 定义为后台编译；GPU stream 同步不能单独证明 CPU 后台编译完成。EAGER 会阻塞等待形状专用 kernel 编译，适合将来针对动态模型单独测试。

NVIDIA 在 [1.4 的 Fixed Issues](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.4.html#fixed-issues) 中记录了 Flux.1 独立 Python 脚本同时启用动态形状和 runtime cache 时的序列化错误修复。但本项目当前五个模型均为静态输入，不能据此认定此次 combiner 阻塞就是该 bug，也不能把 EAGER 当作当前配置的首选修复。

同理，`trt_engine.py` 中“动态形状尚未配置”的注释适用于通用封装的潜在动态输入，不能解释本次五个静态模型为什么在 context 创建阶段卡住。

### D. 稳定性缺口：原生阻塞无法靠 try/except 自动回退

目前 `load_runtime_cache()`、context 创建和 `save_runtime_cache()` 都在模型进程同步执行。Python 异常捕获只能处理返回的错误；如果原生函数不返回，后续禁用缓存、打印错误、冷却等逻辑都无法执行。成功反序列化的日志也不等于后续 context 可用。

因此，未来重新默认启用之前，需要在模型进程之外设启动阶段超时与 GPU 监测。缓存尝试失败/超时后由外部监督者结束本次子进程，并用同一 `.trt`、runtime cache 关闭的配置重试一次；不能在已经卡住的调用线程里尝试恢复。任何隔离/降级都只针对 runtime blob，避免意外删除并重建 `.trt`。

### E. 当前证据不支持的解释

- **引擎缓存失效或缺失**：五个当前 `.trt` 键全部命中，模型内容与基准一致。关闭 runtime cache 后历史稳态推理仍正常。
- **runtime 文件过大**：最大约 1.58 MB，远小于 1.3 发布说明提到需要留意序列化开销的 100 MB 级别；大小不能解释已记录的 90 秒级 context 阻塞。
- **当前仍误读旧 schema**：schema 2 的键与旧 schema 1 不同，本次重新计算已核实。旧 blob 存在不代表当前会读取它。
- **确定是驱动升级污染旧 cache**：项目 runtime 键没有驱动版本或 context 模式字段，但缺少旧 blob 创建时的驱动/模式证据。[1.6 发布说明](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.6.html) 加强了 runtime cache 兼容性检查；这提供了隔离升级复测方向，不能倒推本次根因已确认。

## 4. 建议的最小复测顺序

先解决[原性能待办中的散热/硬件确认要求](PERFORMANCE_OPTIMIZATION_TODO.md)，再按下面顺序做短时 GPU 实验。所有样本使用独立实验 cache 目录和已存在的同版本 `.trt`；任何引擎缺失直接退出，不允许自动构建。一次只运行一个实验进程，记录阶段起止、退出码、温度和输出摘要；温度上限不高于既有 80°C，持续占空目标不高于 90%，保留外部总超时。

| 步骤 | 改变量 | 要回答的问题 |
|---|---|---|
| 1 | 保持 1.3 / WHOLE_GRAPH_CAPTURE，只将 context 改为显式 primary；runtime cache 关闭 | 修正 context 本身能否维持启动、输出和退出正常？ |
| 2 | 同一配置，仅给历史卡住的 combiner 创建空的独立 runtime cache；一次推理同步后保存 | 新 cache 能否完整序列化？先验证最小单引擎 |
| 3 | 新进程加载步骤 2 的 blob，保持其余变量一致 | context 创建与推理能否重复成功？缓存是否真的缩短启动？ |
| 4 | 如果仍阻塞，只切换 CUDA Graph 开/关做小规模 A/B | 是否涉及 graph/runtime-cache 交互？不能同时换版本和 context |
| 5 | 单引擎通过后，扩展五个引擎并与 runtime-cache 关闭的固定输入输出逐字节比较 | 是否仅在完整管线/多 context 共存时失败？ |
| 6 | 仍失败时再建立隔离的新运行库环境 | 新版是否解决同一个最小复现？不得直接覆盖日用环境 |

历史 `autoinit` + 旧 blob 的行为已经记录；只有需要严格建立因果对照、且硬件状态允许时，才用副本追加该旧路径的有界复现。EAGER 只为另有动态输入的模型增加单独用例，不占当前静态 THA3 的首轮矩阵。

升级运行库意味着重建引擎：NVIDIA 的 [1.3 限制说明](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.3.html#limitations) 明确引擎不能跨运行库版本前向复用。新版本可能改变 kernel 选择与像素末位，因此必须保留 FP32、相同模型/输入，重新验证输出，不能只比较启动时间。

## 5. 收益与恢复条件

9 月 2 日关闭 runtime cache 后的核心初始化为 **13.455 秒**，来自[已归档原始数据](benchmark_results/2026-09-02/current-tensorrt-standard.json)。早期约 **1.247 秒**的数字是旧环境下成功使用缓存的历史结果，不是本轮测量或未来性能承诺。潜在收益主要是每次启动省约十余秒；当前没有证据表明恢复它会显著提高稳态 FPS。

恢复默认启用至少需要：空 cache 保存和后续多个新进程重载均完成；五引擎输出验证通过；正常退出无遗留进程；超时路径可以在外部自动回退且不重建 `.trt`；实验元数据包含运行库完整版本、GPU、驱动和 CUDA context 模式。本轮结论是调查与复测入口已经具体化，尚未达到重新启用的条件。

## 6. 可复现的本轮检查

从项目根目录运行只读审计（不会加载 TensorRT/PyCUDA 或修改缓存）：

```powershell
& '.\envs\python_embedded\python.exe' -B '.\benchmarks\audit_runtime_jit_cache.py' --project '.' --cache-dir 'C:/Users/90833/AppData/Local/EasyVtuber/trt-cache' --output '.\benchmark_results\2026-09-05\runtime-cache-audit.json'
```

脚本默认读取项目的 `data/models`，可用 `--model-root` 指定其他模型目录，并与现有 9 月 2 日基准的模型摘要对照。只读取自有 `.runtime.cache` 的大小和存在性，不把 opaque blob 的可读取性当作内容有效性证明。

CPU/mock 测试在 `ezvtuber-rt` 目录执行：

```powershell
& '..\envs\python_embedded\python.exe' -B -m unittest discover -s test -p 'test_*.py' -v
```

结果：`Ran 51 tests in 0.433s`，`OK`。`test_*.py` 的模式有意只选择现有 CPU/mock 集合，不运行命名为 `*_test.py` 的实机推理脚本。
