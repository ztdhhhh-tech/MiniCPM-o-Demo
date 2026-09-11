# MiniCPM-o 4.5 全双工音频通信时延测量方案

本文档用于指导 MiniCPM-o 4.5 在全双工音频通信场景下的架构级时延测量。目标不是只得到一个端到端耗时，而是把“模型发声卡顿”拆解到可定位、可复现、可优化的组件级路径中，区分首包慢、持续输出断流、上游输入堆积、播放端调度不足、以及跨单元 finalize 阻塞等不同问题。

## 0. 已落地实现与运行方式

当前仓库已包含采集器、模型/后端埋点、浏览器播放关联字段和 JSONL 汇总脚本。默认关闭诊断，不改变原有协议；启动全双工服务前设置：

```powershell
$env:MINICPM_LATENCY_MODE = "light"
$env:MINICPM_LATENCY_LOG_DIR = "logs/duplex_latency"
$env:MINICPM_LATENCY_SAMPLE_RATE = "1.0"
```

可选模式为 `off`、`light`、`gpu`、`profile`：

- `off`：关闭详细埋点；
- `light`：CPU wall time，适合持续运行；
- `gpu`：增加 CUDA Event，并报告 `gpu_sync_overhead_ms`；
- `profile`：对单元生成 Chrome trace，适合短时间定位。

采集结果按会话写入 `logs/duplex_latency/<session_id>.jsonl`。运行聚合脚本：

```powershell
python tools/duplex_latency_report.py logs/duplex_latency/<session_id>.jsonl `
  --output logs/duplex_latency/<session_id>.summary.json
```

服务端事件携带 `trace_id`、`unit_id`、`input_chunk_id` 和 `output_seq`；浏览器通过 `RealtimeSession.onLatency` 和 `RealtimeSession.onGap` 暴露播放块与 gap 事件。出现 gap 时，用 `trace_id/unit_id/output_seq` 回查同一 JSONL 中的 `backend.audio.done`、`backend.unit.done`、`transport.ws_send` 和模型 latency spans。

## 1. 背景与目标

MiniCPM-o 4.5 的全双工能力来自 Omni-Flow 式的时序组织：系统把视觉、听觉和输出控制 token 组织成连续的时间单元，在同一个 LLM 上完成“听”和“说”的决策，再通过轻量级语音 token 解码器和 token2wav 波形解码器输出音频。实际运行时，一个全双工单元大致经过：

```text
客户端采集音频/视频
  -> 网络发送
  -> 后端等待上一单元 finalize
  -> 音频/视觉预处理
  -> Whisper 音频编码 / SigLIP 视觉编码
  -> 投影与池化
  -> LLM prefill
  -> LLM generate 决策
  -> TTS 输入准备
  -> TTS token 生成
  -> token2wav 波形生成
  -> 网络返回音频帧
  -> 客户端解码、重采样、排队、播放
```

用户感知到的“发声卡顿”通常发生在最后的播放端，但根因可能在链路任意位置。例如 token2wav 生成速度慢、上一单元 finalize 等待过长、LLM/TTS 生成抖动、WebSocket 发送阻塞、客户端 AudioContext 缓冲耗尽，都会表现为声音断续。因此测量方案需要覆盖端到端链路，并能把一次播放 gap 反查到对应的后端组件耗时。

本方案的目标：

1. 建立统一 trace id，把输入音频块、模型时间单元、输出音频块和客户端播放事件关联起来。
2. 对模型核心组件分别测量 CPU wall time 和 GPU kernel time，避免 CUDA 异步执行导致的误判。
3. 直接测量“生成的真实 PCM 时长”和“生成间隔”，判断模型是否能持续供应音频。
4. 在客户端测量播放队列水位、gap、重采样耗时和实际排队时间。
5. 给出可执行的阶段化埋点与验收指标，支撑后续优化。

## 2. 需要测量的架构组件

结合论文架构和当前代码，建议按以下组件分层测量。

| 层级 | 组件 | 代码位置 | 主要风险 |
| --- | --- | --- | --- |
| 输入链路 | 客户端录音、编码、WebSocket 发送 | `static/duplex/lib/realtime-session.js` | 采集块间隔不稳、网络发送阻塞 |
| 后端接收 | WebSocket receive、media decode、session queue | `py_backend/server.py` | 输入排队、解码阻塞 |
| 单元依赖 | 等待上一单元 `duplex_finalize` | `py_backend/server.py` | finalize 在下一次 prefill 前形成隐藏阻塞 |
| 音频编码 | mel/Whisper encoder、MLP、AvgPool | `MiniCPMO45/modeling_minicpmo_unified.py` | 音频输入处理慢，导致听觉上下文滞后 |
| 视觉编码 | SigLIP VPM、Resampler | `MiniCPMO45/modeling_minicpmo_unified.py` | 视频输入导致 prefill 变慢 |
| LLM prefill | Qwen3 LLM 上下文写入 KV cache | `MiniCPMO45/modeling_minicpmo_unified.py` | 时间单元越长，prefill 越可能阻塞 |
| LLM generate | listen/speak 控制 token 与文本 token 生成 | `MiniCPMO45/modeling_minicpmo_unified.py` | 决策慢或 token 循环抖动 |
| TTS 输入准备 | text embedding + hidden projector | `MiniCPMO45/modeling_minicpmo_unified.py` | CPU/GPU 同步和张量拼接开销 |
| TTS token 生成 | Llama speech token decoder | `MiniCPMO45/modeling_minicpmo_unified.py` | 语音 token 供应不足 |
| 波形生成 | token2wav / flow matching decoder | `MiniCPMO45/modeling_minicpmo_unified.py` | 真实 PCM 时长不足或生成耗时大 |
| 输出链路 | base64/JSON/WebSocket send | `py_backend/server.py` | 后端算完但发送慢 |
| 播放端 | decode、resample、AudioContext schedule | `static/duplex/lib/audio-player.js` | 播放缓冲耗尽、重采样或调度延迟 |

其中需要特别注意：当前 `py_backend/server.py` 已经有部分 `prefill_ms`、`generate_ms`、`cost_llm_ms`、`cost_tts_ms`、`cost_token2wav_ms` 等指标，但这些指标还不足以解释所有卡顿，因为上一单元 finalize 等待、实际 PCM 时长、客户端播放水位和 WebSocket send 耗时没有完整串起来。

## 3. Trace ID 设计

每个全双工 session 需要建立四类 ID。

| ID | 含义 | 示例 |
| --- | --- | --- |
| `session_id` | 一次 WebSocket 会话 | `s_20260907_001` |
| `turn_id` | 一轮从用户输入到模型输出的逻辑轮次 | `turn_12` |
| `input_chunk_id` | 客户端上传的音频/视频块序号 | `in_000183` |
| `unit_id` | 模型内部 Omni-Flow 时间单元序号 | `unit_000183` |
| `output_seq` | 后端返回的音频块序号 | `out_000057` |

推荐在每条事件里携带：

```json
{
  "session_id": "s_20260907_001",
  "turn_id": 12,
  "input_chunk_id": 183,
  "unit_id": 183,
  "output_seq": 57,
  "parent_input_chunk_id": 183,
  "latest_input_sample_end": 2928000,
  "output_sample_start": 1344000,
  "output_sample_count": 24000,
  "output_sample_rate": 24000
}
```

`latest_input_sample_end` 和 `output_sample_*` 很关键。它们能判断模型说话时基于多旧的输入上下文，也能判断一次输出到底提供了多少毫秒的可播放音频。不要默认每次输出都是 1 秒，必须用实际 PCM 样本数计算：

```text
pcm_duration_ms = output_sample_count / output_sample_rate * 1000
```

## 4. 计时原则

### 4.1 CPU wall time

后端 CPU 侧统一使用 `time.perf_counter_ns()`，记录每个阶段的 `start_ns`、`end_ns` 和 `duration_ms`。客户端使用 `performance.now()`，并保留 `AudioContext.currentTime`。

不要直接把客户端时间戳和服务端时间戳相减来计算单向延迟。浏览器和后端机器不是同一个时钟域，最多通过 ping/pong 做粗略校准，并记录不确定范围。播放体验相关指标应优先使用客户端本地事件之间的差值，例如：

```text
音频帧到达客户端 -> decode 完成 -> resample 完成 -> schedule 到 AudioContext -> 预计播放时刻
```

### 4.2 GPU kernel time

PyTorch CUDA 默认异步执行，CPU 函数返回不代表 GPU 已完成。组件级 GPU 时间应使用 `torch.cuda.Event(enable_timing=True)`：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
result = component(...)
end.record()

# 在线路径不要每个组件立即 synchronize，否则测量会改变实时性能。
# 可把 event 保存到 trace 里，延迟 query；深度诊断时再同步。
```

在线测量建议：

1. 每个音频单元记录 CPU wall time。
2. 按采样率记录 GPU Event，例如每 20 个单元采 1 个。
3. 对疑似慢组件开启短时 profiler replay，而不是长期全量 profiler。
4. 对同一 CUDA stream 上的组件可以比较 event 区间；多 stream 重叠时不要把各组件 GPU 耗时简单相加。

### 4.3 不能只看组件耗时

模型是否会卡顿，核心取决于供应音频的速度：

```text
audio_supply_ratio = generated_pcm_duration_ms / wall_interval_between_audio_outputs_ms
```

如果每 1.15 秒才生成 1.00 秒音频，即使单个组件看起来不算慢，播放缓冲也会每秒损失约 150 ms，运行几轮后必然断流。缓冲只能推迟断流，不能消除持续赤字。

## 5. 后端埋点方案

### 5.1 `py_backend/server.py`

建议新增或扩展以下时段：

| 指标 | 起点 | 终点 | 用途 |
| --- | --- | --- | --- |
| `server_receive_ms` | WebSocket 消息进入 | JSON/media decode 完成 | 判断输入接收与解码开销 |
| `previous_finalize_wait_ms` | `_push_full_duplex` 开始等待 | `_wait_finalize()` 返回 | 暴露上一单元 finalize 的隐藏阻塞 |
| `prefill_wall_ms` | 调用 `backend.duplex_prefill` 前 | 返回后 | 当前单元输入写入耗时 |
| `generate_wall_ms` | 调用 `backend.duplex_generate` 前 | 返回后 | 当前单元输出生成总耗时 |
| `finalize_wall_ms` | 调用 `backend.duplex_finalize` 前 | 返回后 | 单元边界处理与 cache 维护耗时 |
| `ws_send_ms` | `send_json` 前 | 返回后 | 判断网络发送或 backpressure |
| `backend_unit_wall_ms` | 单元开始 | 所有同步生成完成 | 端到端后端耗时 |

当前 `_push_full_duplex` 会在正式计时前等待 `_wait_finalize()`，所以 `wall_clock_ms` 可能没有包含上一单元 finalize 的等待时间。这个字段应单独记录，否则会出现“服务端看起来不慢，但用户听到卡顿”的错觉。

### 5.2 `MiniCPMO45/modeling_minicpmo_unified.py`

建议在 `streaming_prefill`、`streaming_generate`、`finalize_unit` 周围补齐组件级事件。优先测这些阶段：

| 阶段 | 建议字段 |
| --- | --- |
| 音频预处理 | `audio_process_ms` |
| Whisper encoder | `audio_encoder_gpu_ms`、`audio_encoder_wall_ms` |
| audio projector + pooling | `audio_projector_pool_ms` |
| 视觉预处理 | `vision_process_ms` |
| SigLIP VPM | `vision_encoder_gpu_ms`、`vision_encoder_wall_ms` |
| Resampler | `vision_resampler_ms` |
| LLM prefill | `llm_prefill_gpu_ms`、`llm_prefill_wall_ms` |
| LLM generate | `llm_generate_gpu_ms`、`llm_generate_wall_ms`、`n_llm_tokens` |
| TTS 输入准备 | `tts_prep_ms` |
| TTS token 生成 | `tts_generate_gpu_ms`、`tts_generate_wall_ms`、`n_tts_tokens` |
| token2wav | `token2wav_gpu_ms`、`token2wav_wall_ms`、`pcm_duration_ms` |
| finalize | `finalize_gpu_ms`、`finalize_wall_ms` |

需要同时记录输入规模：

```json
{
  "audio_input_ms": 1000,
  "audio_tokens": 10,
  "image_count": 0,
  "image_slices": 0,
  "llm_context_len": 842,
  "kv_cache_len": 842,
  "n_llm_tokens": 4,
  "n_tts_tokens": 25,
  "pcm_duration_ms": 1000
}
```

如果没有这些规模字段，组件耗时很难比较。比如视觉路径偶发慢，可能只是这一单元带了更多图像切片；TTS 慢，可能是本次生成的 speech token 更多。

## 6. 客户端埋点方案

客户端重点不在“服务端花了多少时间”，而在播放队列是否被耗尽。建议在 `static/duplex/lib/audio-player.js` 和 `static/duplex/lib/realtime-session.js` 中记录：

| 指标 | 含义 |
| --- | --- |
| `response_recv_ms` | 音频 delta 到达浏览器时间 |
| `audio_decode_ms` | base64/Float32Array 解码耗时 |
| `resample_ms` | 24k 到实际 AudioContext sample rate 的重采样耗时 |
| `schedule_delay_ms` | 到达后距离计划播放时间的差值 |
| `playback_ahead_ms` | 已排队音频末尾距离当前 AudioContext 时间的余量 |
| `gap_ms` | `_nextTime < currentTime` 时的真实调度缺口 |
| `gap_count` | 播放期间 gap 次数 |
| `pdelay_ms` | 初始播放延迟 |
| `actual_sample_rate` | AudioContext 实际采样率 |
| `chunk_pcm_duration_ms` | 本块实际 PCM 时长 |

现有 `AudioPlayer` 已经有 `ahead`、`gapCount`、`totalShift`、`pdelay`、`onGap` 等基础能力，可以在此基础上把 `output_seq`、`unit_id` 和 `chunk_pcm_duration_ms` 加进去。

浏览器支持时，可使用 `AudioContext.getOutputTimestamp()` 估计 `contextTime` 与 `performanceTime` 的映射，从而更接近硬件输出时间。它适合定位浏览器播放端问题，但仍不能替代真实声卡回环测试。

## 7. 统一日志结构

建议每个 session 输出 JSONL，每行一个事件，便于后续用脚本聚合：

```json
{
  "event": "backend.generate.done",
  "session_id": "s_20260907_001",
  "turn_id": 12,
  "unit_id": 183,
  "output_seq": 57,
  "ts_ns": 1790000000000000,
  "duration_ms": 286.4,
  "metrics": {
    "llm_generate_wall_ms": 42.1,
    "tts_prep_ms": 3.8,
    "tts_generate_wall_ms": 71.5,
    "token2wav_wall_ms": 164.7,
    "n_llm_tokens": 4,
    "n_tts_tokens": 25,
    "pcm_duration_ms": 1000.0
  }
}
```

播放端 gap 事件应包含上游关联信息：

```json
{
  "event": "client.audio.gap",
  "session_id": "s_20260907_001",
  "turn_id": 12,
  "unit_id": 183,
  "output_seq": 57,
  "gap_ms": 137.2,
  "playback_ahead_ms": -137.2,
  "chunk_pcm_duration_ms": 1000.0,
  "response_recv_ms": 184234.5
}
```

当出现 gap 时，诊断脚本应自动拉取前后 3 个输出块对应的：

```text
previous_finalize_wait_ms
prefill_wall_ms
generate_wall_ms
llm_generate_wall_ms
tts_generate_wall_ms
token2wav_wall_ms
ws_send_ms
chunk_pcm_duration_ms
playback_ahead_ms
```

这样能直接判断是模型供应不足、发送阻塞、客户端调度问题，还是 finalize 在单元边界处堵住了下一次输入。

## 8. 分析指标

### 8.1 端到端指标

| 指标 | 定义 | 解释 |
| --- | --- | --- |
| `ttfs_ms` | 用户有效输入结束到第一段可播放音频到达或被 schedule 的时间 | 首次发声延迟 |
| `ttfa_ms` | 用户有效输入结束到 AudioContext 计划播放第一帧的时间 | 更接近用户听感 |
| `input_to_output_staleness_ms` | 输出生成时引用的最新输入样本距离当前时间的差 | 模型“听到”的上下文新鲜度 |
| `gap_rate` | gap 次数 / 输出块数 | 播放断流频率 |
| `gap_total_ms` | 所有 gap 持续时间之和 | 卡顿总量 |

### 8.2 实时供应指标

| 指标 | 定义 | 解释 |
| --- | --- | --- |
| `backend_rtf` | 后端总处理耗时 / 输入音频时长 | 是否能跟上输入 |
| `speech_supply_ratio` | 生成 PCM 时长 / 输出间隔 | 是否能持续发声 |
| `buffer_ahead_p50/p95/min` | 播放队列水位分位数 | 是否有断流风险 |
| `audio_deficit_ms` | 输出间隔 - 生成 PCM 时长 | 每块消耗多少缓冲 |

### 8.3 组件指标

组件报告不要只看平均值，应至少输出：

```text
P50 / P90 / P95 / P99 / max
```

并按以下维度拆分：

```text
冷启动 / warmup 后
纯音频 / 音频+视频
listen / speak
首个 speak chunk / 中间 chunk / 末尾 chunk
短对话 / 10-30 分钟长会话
有打断 / 无打断
```

不要把每个组件的 P99 相加当作端到端 P99。组件 P99 往往来自不同请求，直接相加会夸大最差路径。应对真实 trace 做端到端分位数，同时用组件分位数解释原因。

## 9. 卡顿归因规则

| 观测现象 | 可能根因 | 下一步 |
| --- | --- | --- |
| `previous_finalize_wait_ms` 高 | 上一单元 finalize 阻塞下一次 prefill | 分析 finalize 内部 cache 维护和边界 token feed |
| `token2wav_wall_ms` 高，`pcm_duration_ms` 正常 | 波形解码慢 | 降低 flow step、优化 token2wav、检查 GPU 占用 |
| `pcm_duration_ms` 小于预期 | TTS/token2wav 没有提供足够音频 | 检查 speech token 数、lookahead、flush 逻辑 |
| `speech_supply_ratio < 1` 且持续下降 | 后端供应速度低于播放消耗 | 优先优化最长组件或调整输出块策略 |
| `ws_send_ms` 高 | WebSocket backpressure 或网络问题 | 检查 payload 大小和客户端接收速度 |
| 客户端 `resample_ms` 高 | 浏览器重采样开销大 | 优化 resampler 或改输出采样率策略 |
| `gap_ms` 高但后端生成快 | 客户端排队/调度问题 | 检查 `AudioPlayer._nextTime`、初始 delay 和 schedule 策略 |
| 视频单元显著慢 | 视觉 encoder/resampler 占用 | 按图像帧率、分辨率、切片数拆分 |
| 运行越久越慢 | KV cache、日志、队列或内存累积 | 做 10-30 分钟 soak test 并看趋势 |

## 10. 实施阶段

### 阶段一：轻量埋点，先定位真实断点

优先完成：

1. 在后端补充 `previous_finalize_wait_ms`、`ws_send_ms`、`pcm_duration_ms`、`output_seq`。
2. 在客户端把 `output_seq`、`unit_id` 透传到 `AudioPlayer` 的 `onGap` 和 metrics。
3. 建立 JSONL trace 输出，支持按 session 保存。
4. 写一个离线分析脚本，输出 gap 附近的组件耗时表。

这一阶段不引入 profiler，也不强制 CUDA 同步，风险较低，能最快判断卡顿是后端供应不足还是播放端调度不足。

### 阶段二：GPU 组件测量

在疑似瓶颈组件周围加入采样式 CUDA Event：

```text
audio_encoder
vision_encoder
llm_prefill
llm_generate
tts_generate
token2wav
finalize
```

默认只采样一小部分单元，避免计时本身干扰实时交互。对高延迟 session，可保留最近 N 个 event 的 ring buffer，出现 gap 后把前后若干个单元标记为重点样本。

### 阶段三：短时 profiler replay

当阶段一、二确定某个组件高概率是瓶颈后，再对该组件做短时 profiler：

```text
warmup 5-10 个单元
profile 10-30 个单元
记录 kernel、CPU op、memory copy、synchronize 点
```

重点排查：

```text
`.item()`、`.cpu()`、日志打印导致的同步
小 kernel 过多
H2D / D2H copy
token 循环内重复构造张量
token2wav 是否和 LLM/TTS 抢同一 GPU stream
```

## 11. 测试矩阵

| 场景 | 目的 |
| --- | --- |
| 纯音频 1 分钟 | 基础持续发声能力 |
| 音频+视频 1 分钟 | 视觉路径对实时性的影响 |
| 冷启动首轮 | 模型 warmup、首包延迟 |
| warmup 后首轮 | 排除加载影响后的首包延迟 |
| 10-30 分钟长会话 | 缓冲、KV cache、内存和队列趋势 |
| 用户中途打断 | listen/speak 切换是否破坏播放或 finalize |
| 网络弱化模拟 | 区分模型瓶颈和传输瓶颈 |
| 不同播放延迟 | 评估 `playbackDelay` 对 gap 与交互延迟的权衡 |

测试时不要采用“等模型处理完再发送下一块音频”的闭环方式，因为这会掩盖输入堆积。客户端应按真实采集节奏持续发送音频块，后端测量队列和处理速度。

## 12. 验收标准建议

具体阈值需要按部署硬件校准，但建议先使用以下本地实验标准：

| 指标 | 目标 |
| --- | --- |
| 持续 speak 期间单次 starvation | 不超过 100 ms |
| gap rate | 小于 0.1% |
| `speech_supply_ratio` | 稳定大于 1.0 |
| `playback_ahead_ms` | 长会话中不持续下降 |
| 输入 backlog | 长会话中不持续增长 |
| 首次发声延迟 | 以当前基线为准，优化后不回退 |
| 打断响应 | 以当前基线为准，优化后不回退 |

如果 `playback_ahead_ms` 持续下降，即使暂时没有 gap，也应视为失败。它说明系统正在消耗缓冲，卡顿只是时间问题。

## 13. 后续优化方向

测量完成后，应按归因结果选择优化策略：

| 瓶颈 | 优化方向 |
| --- | --- |
| finalize 阻塞 | 缩短 finalize 内部操作；检查是否能把非依赖部分延后；减少边界 token feed 的同步点 |
| LLM generate 慢 | 减少 token 循环同步；合并张量构造；检查 logits 处理和日志开销 |
| TTS generate 慢 | 控制 speech token 数；优化 projector 与 TTS decoder；复用缓存 |
| token2wav 慢 | 调整 flow step；检查外部 decoder stream；评估更小输出块和 lookahead 策略 |
| 视觉路径慢 | 降低全双工视频帧率/分辨率/切片数；仅在必要单元处理视觉 |
| 客户端 gap | 调整初始播放延迟；改进 schedule 策略；优化重采样 |
| 网络发送慢 | 减小 payload；二进制帧替代 base64；拆分音频帧 |

优化时不要先随意缩短 Omni-Flow 时间单元或 speech token 预算。论文中 1 秒级时间单元是效果和实时性的折中，过度缩短可能牺牲模型理解和表达稳定性。应先用 trace 证明瓶颈位置，再做结构改动。

## 14. 推荐优先级

第一优先级是把三件事串起来：

```text
previous_finalize_wait_ms
真实 pcm_duration_ms
客户端 output_seq 级别的 gap/ahead
```

这三项能最快回答“到底是模型没及时产出声音，还是声音已经产出但客户端没顺利播放”。等这个闭环跑通后，再进入 GPU Event 和 profiler 级别的精细化定位。
