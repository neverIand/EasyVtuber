# runtime JIT cache 修复与短时验证（2026-09-05）

已完成 CUDA context 用法修正、启动超时与自动回退，并通过短时 GPU 验证。**实际模型进程从启动到 ready：关闭 runtime cache 为 14.974 秒，缓存命中的两次测量为 5.030 / 2.020 秒，首帧逐字节一致。** 用户随后确认启动器内停止再启动明显更快，并要求正式应用和提交。现在 A、B 默认开启带启动保护的 runtime cache，日常调试用 B；C 保留为设置相同的兼容入口。历史原生阻塞这次没有复现，旧 context + 旧 cache 的有界对照也成功，因此不能宣称已经找到了历史死锁根因。

用户已确认先前的散热/硬件问题解决，授权短时 GPU 验证。自动验证没有升级运行库、改变模型精度或重建引擎。在自动验证结束时，生产缓存目录里的五个内容寻址 `.trt` 和五个旧 runtime blob 最后修改时间仍为 9 月 2 日，schema 2 runtime 文件未创建；自动验证的新 cache 与引擎副本均位于实验目录。该文件状态是用户手动试用之前的快照。修改没有调整保存的 `launcher.json`。

前期提交历史、ONNX 形状与官方资料审计见[首轮调查归档](RUNTIME_JIT_CACHE_INVESTIGATION_2026-09-05.md)。完整结果索引、版本、GPU/驱动、输出核对和自动验证时的生产缓存文件状态见 [runtime-cache-repair-summary.json](benchmark_results/2026-09-05/runtime-cache-repair-summary.json)。改动基于主项目 `1a6b6cb`、运行时子模块 `43d67b8`；运行时修复提交为 `65048df`，主项目记录其新提交指针。

## 代码修改

- 运行时增加 [`cuda_primary.py`](ezvtuber-rt/ezvtb_rt/cuda_primary.py)，对 `EZVTB_DEVICE_ID` 指定设备 retain/push primary context，退出时清理并 detach；替换 `pycuda.autoinit` 创建独立 user context 的路径，保持后端惰性导入。这与 [NVIDIA TensorRT-RTX 1.3 的 PyCUDA 建议](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.3.html#known-issues) 一致，也避免 PyCUDA 的 `CUDA_DEVICE` / `.cuda_device` 选择与应用设备号不一致。它修正了具体兼容性问题，但本轮对照不支持将其认定为历史阻塞的唯一原因。
- [`trt_engine.py`](ezvtuber-rt/ezvtb_rt/trt_engine.py) 检查 `execute_async_v3()` 返回值；提交失败立即报错，不再尝试保存 cache。保留已有的“成功推理、stream 同步后首次保存”，并在 context 日志中标明模型名称。
- 主项目增加 [`model_startup.py`](src/utils/model_startup.py)。开启 runtime cache 的 TensorRT 启动由模型进程外的主进程等待 ready，默认每次 60 秒。失败/超时先 terminate、join，必要时 kill，再确认旧进程退出并释放其输出共享内存，之后只重试一次，关闭 runtime cache。旧进程无法退出时拒绝启动第二个 GPU worker。
- 两次有超时保护的尝试均要求已有有效引擎，禁止构建；后端导入失败明确报错，禁止静默转为 DirectML。首次 worker 只有遇到特定的 `EngineCacheRequiredError` 才发出缺少可用引擎信号：正式入口收到该信号后先回收旧 worker，再以 runtime cache 关闭的配置进入原有异步构建流程，保留启动器原有构建确认。普通异常、超时或第二次回退缺少引擎均不能开启构建。设置 `EZVTB_TRT_REQUIRE_ENGINE_CACHE=1` 的严格验证模式始终禁止构建。
- 新 ready 事件在模型、图像和 warm-up 完成后设置。主进程在选定最终模型 worker 后才绑定输出共享内存，避免回退后仍读取已释放的旧缓冲。
- `01A` 普通入口和 `01B` 调试入口默认设置 `EZVTB_TRT_RUNTIME_CACHE=1`；`01B` 使用 `-u` 及时输出日志。原先的 [`01C.Runtime-cache-trial.bat`](01C.Runtime-cache-trial.bat) 继续保留，现在采用相同设置并提示日常调试用 B。三个入口都尊重调用者已设置的变量，可通过 `EZVTB_TRT_RUNTIME_CACHE=0` 显式关闭缓存。SDK 的默认值仍为关闭，避免无外部监督的直接调用默认进入原生缓存路径。

启动超时覆盖初始化和首次保存；它不是持续运行期间的所有原生调用看门狗，也不自动隔离某一个坏 blob。若同一 blob 每次都导致超时，下一次启动仍会尝试它。更复杂的动态模型、RIFE/SR 与长时运行不在本轮实机验收范围内。首次需要构建引擎时走原有异步流程，不把合法的长构建当成 60 秒缓存启动超时。

## 实机结果

环境：TensorRT-RTX `1.3.0.35`、PyCUDA `2025.1.2`、RTX 4060 Laptop、驱动 `596.49`。模型为原有 THA3 separable FP32，五个 ONNX 摘要与 9 月 2 日基准一致，输入维度均固定。CUDA Graph 保持开启，没有同时更换运行库或切换 EAGER。

| 测量范围 | 关闭 runtime cache | 新 cache 保存 / 命中 | 核对 |
|---|---:|---:|---|
| combiner 原生 context，primary context | 4,235.000 ms | 首次 3,860.669 ms；命中 85.912 ms | 六个单引擎用例输入/输出相同 |
| 五引擎 `CoreTRT` 构造 | 15,799.683 ms，旧 user context 对照 | 首次 primary 17,935.477 ms；primary 重载 1,011.696 ms；应用 primary 初始化入口重载 989.419 ms | 五组管线的四个固定姿态输出全部相同 |
| 实际 `ModelClientProcess`，包含进程启动与 warm-up | 14.974 s | 两次命中 5.030 / 2.020 s | 三次共享内存首帧 RGBA 摘要相同，无回退 |

五个首次新 cache 均保存成功，序列化阶段分别约 21–44 ms，文件约 0.73–1.58 MB；随后已在多个新进程重载。管线测量和实际进程测量的计时边界不同，不能混为一个“整应用启动时间”；这些是短测样本，不能直接推出稳定的平均启动时间或稳态 FPS 提升。

另做了旧 schema 1 blob 副本的对照：combiner 在 primary / 原 user context 下均可完成；原 user context + 全部五个旧 blob 的构造为 1,208.271 ms，四姿态输出一致。这轮没有得到历史卡死的原生线程栈，仍不足以证明旧 blob 损坏、驱动变化或动态形状后台编译是根因。

实际模型进程的共同首帧 RGBA SHA-256：

```text
67829a4fc8bc596af1a54c942b171154d7e475244d0174d76d936a6f0098d73a
```

14 个有效 TensorRT GPU 用例全部完成，所有所属 worker 均退出，NVML 采样最高温度 56°C。探针在 GPU 低于 65°C 时开始，温度截止为 75°C，并有阶段/总超时和连续高利用率截止；使用 80% 占空目标。首次全管线 JIT 有一个 95% 利用率采样点，未连续触发截止；占空目标不等于单次原生调用的瞬时硬上限。

前两次实际进程探针 `client-load-1` / `client-load-2` 因沙箱重置子进程 PATH，找不到 TensorRT DLL 而自动使用了 DirectML。其 `status: ok` 仅表示当时输出检查完成，**已从 TensorRT 有效样本中排除**。CPU 环境探针确认该 PATH 差异后，经用户批准在沙箱外重测，得到上述三次 `client-trt-*` 结果。严格后端检查随之加入，避免以后误判。

## 回归与故障路径

- 最终正式应用阶段主项目 76 项、运行时 55 项测试通过，共 131 项；未运行需要 wx GUI 的 `test_launcher_ui.py`。之前实机阶段对应主项目 69 项、运行时 55 项；正式应用阶段没有重新制造 GPU 卡顿或构建引擎。
- 覆盖 primary context 设备选择、push 失败释放、正常清理与重复清理、惰性导入、enqueue 失败禁止保存、缺失/无效引擎禁止构建。
- 覆盖模型 worker 提前退出、启动超时、只回退一次、旧进程先退出再重试、必要时 kill、无法结束旧进程时拒绝重叠、spawn 失败释放共享内存，以及试用模式禁止隐式切换后端。
- 正式应用另覆盖特定缺引擎信号可进入原有构建流程、严格探针始终禁止构建、普通异常/超时不能开启构建、第二次回退不能追加构建，以及真实模型 worker 仅对指定异常发送缺引擎信号。
- Windows 真实子进程用 `ctypes.PyDLL(...).Sleep(60000)` 模拟原生函数占住 Python GIL；父进程在 2 秒截止后结束它，关闭 cache 的第二个 worker 正常 ready 并清理。
- 单独的外部探针看门狗也在约 1.188 秒终止同类阻塞；该用例预期状态就是 timeout，未调用 GPU。

运行时 CPU/mock 回归命令（在 `ezvtuber-rt` 目录）：

```powershell
& '..\envs\python_embedded\python.exe' -B -m unittest discover -s test -p 'test_*.py'
```

新增诊断脚本为 [单引擎/管线外部监督探针](benchmarks/probe_runtime_jit_cache.py)、[五引擎管线实现](benchmarks/probe_runtime_pipeline.py) 和[实际模型进程/首帧检查](benchmarks/verify_model_startup.py)。每个用例要求全新的 `--case-dir`，保留原始结果并防止覆盖；GPU 用例必须串行运行。引擎与 runtime 二进制副本通过实验目录 `.gitignore` 排除，JSON 证据保留。

## 日常入口与后续观察

用户在试用阶段已确认：通过 `01C` 选择 TensorRT 后，停止再启动明显更快；当时 `01B` 仍较慢，因为只有 `01C` 开启缓存。随后用户授权正式应用和提交，现已把相同的缓存与超时保护用于 A、B，C 也保持一致。以后普通启动用 A，需要调试输出用 B；C 只是保留的兼容入口。

真实动作输入与画面稳定性尚未收到明确反馈，可在日常使用 B 时继续观察：

1. 关闭已打开的旧启动器，再用 `01B.启动器（调试输出）.bat` 重新打开，让新环境开关生效。使用 **TensorRT、THA3 separable、FP32、30 FPS / 80% GPU 占空目标**；本轮 GPU 实测关闭 RIFE/SR。
2. 日志出现 `Loaded TensorRT runtime cache` 表示命中；`Inference backend: TensorRT` 确认后端。首次生成 runtime cache 仍需要 JIT；已有可用缓存的后续启动应明显更快。
3. 接入平时的动作输入，短用约 30 秒，观察眨眼、转头、透明度和预览。若出现 `runtime-cache startup failed`，记录该行与前面的最后一个模型/context/cache 日志；程序会先结束旧 worker，再禁用 runtime cache 重试一次。

正式入口包含启动超时保护；探针的 NVML 温度截止没有接入普通启动器。无需做长时间压力测试，也不要清理 TensorRT 缓存来制造冷启动。临时排障可在启动 B 的同一个命令行环境设置 `EZVTB_TRT_RUNTIME_CACHE=0`，三个入口都尊重这个显式关闭值。DirectML 不使用该缓存。

当前结论是：缓存加速与保护路径已取得可复现的短测证据，并在用户确认重启加速、授权正式应用后启用于启动器。历史间歇性阻塞的根因仍未确认。
