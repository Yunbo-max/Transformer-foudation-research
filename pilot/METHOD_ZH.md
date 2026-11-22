# Predicted-Skip 在 GDN-2 上做什么

这个分支保留的研究假设是：**上一次检索得到的答案，进入后续 token 流时，
可能被再次写入记忆；利用读取反馈，适当减小这类更新，能否减少干扰，同时
保留学习新关联的能力？** 这是值得验证的机制。当前完成了源码与数学审查，
尚没有本修订版本的训练结果来证明收益。

## 1. GDN-2 已有的反馈，与这里新增的反馈

这里采用 key-by-value 状态 $S\in\mathbb R^{K\times V}$。原始 GDN-2 更新为

$$
\bar S_t=D_tS_{t-1},\qquad
z_t=w_t\odot v_t-\bar S_t^\top(b_t\odot k_t),\qquad
S_t=\bar S_t+k_tz_t^\top.
$$

GDN-2 本身已经通过 erase 项读取旧状态，并有可学习的输入门。
不能把它描述成完全不知道当前记忆内容。这里额外比较的是

$$
r_t=S_c^\top q_{t-1},\qquad
\rho_t=\exp\{-\gamma_h\max[0,\cos(r_t,v_t)]\}.
$$

$S_c$ 是本 chunk 开始前的状态。它提供的是**前次 Query 的边界状态读取反馈**；
原始更新则针对**当前写入 Key**。当两者地址不同时，前者可能提示答案回写，
也可能把“新 Key 对应已有 Value”误判成可以少写。这两种可能都要测。

因此，GDN-2 是检验这个假设的底座，不是在数学上必须加入 Predicted-Skip。
本分支采用 channel-wise GDN-2 算子和小型匹配 decoder，并不是完整 NVIDIA
预训练架构的复现。算子依据见
[NVIDIA 固定版本源码](https://github.com/NVlabs/GatedDeltaNet-2/blob/de8476de0fecae08fc16f0bf3154900244c61c73/lit_gpt/gdn2_ops/chunk_gdn2.py)。

## 2. 为什么联合缩放 erase 和 write

joint 同时采用 $b'_t=\rho_tb_t$、$w'_t=\rho_tw_t$。对于未加额外投影的
raw 更新，在同一个输入状态下严格有

$$
S_t^{\rm joint}=\bar S_t+\rho_t(S_t^{\rm raw}-\bar S_t).
$$

这会缩小完整更新，保留 erase/write 的相对平衡。若只压低 write，
erase 仍按原强度执行，可能破坏已经正确的关联。这个结论说明联合缩放的
代数合理性；何时应该缩小更新，仍由近似的读取反馈判断。

当前默认 native 模式在缩放后应用 NVIDIA beta guard。该投影一般不是齐次的，
因此不能把上述 raw 恒等式直接当作 native 两条轨迹的等价关系。衰减仍然生效。
本 pilot 选择初始 $\gamma=0.25$，相似度为 1 时 $\rho\approx0.779$，
对应减小约 22.1% 的 raw active edit，并不是完全不写，也不是准确率下降 22.1%。

## 3. 相比 Vuk 的公开实现，哪些已经有，哪些是这里的集成

核对的是
[固定版本 mix.py](https://github.com/vukrosic/llm-research-paper-drafts/blob/d0dc052749a97061f6daa6907371bd0a2ec33405/03-linear-attention-skip-lookups/mix.py)。

| 内容 | Vuk 的公开实现 | 当前分支 |
|---|---|---|
| 前次 Query、chunk-start state、cosine gate | 已有 | 保留 |
| 同时缩放 erase/write | `--lpsE 1` 已有 | 作为 joint 实验臂保留 |
| 每层/头可学习强度、shuffled/key-read 对照 | 已有 | 保留并明确配置 |
| 底层 gate | `code e` 使用每头标量，erase 为 `beta * e` | 接入独立 key-channel erase、value-channel write 和 key-channel decay |
| gate 的 state 梯度 | chunked 路径通过边界状态反传 | 默认 detach 这个支路，可显式打开；主递推梯度保留 |
| 当前补充 | — | guard/精度/后端验证，以及按写入事件保存诊断 |

已有机制应准确归属。当前集成和验证是实际工作，但不能仅凭换底座声称新 gate
或已优于作者结果。作者的初始强度由 `--lps` 配置，默认关闭；0.25 是本 pilot
的设置，不是作者所有实验的统一默认值。

## 4. 如何判断这个方法有没有用

保留 baseline、write、joint、shuffled 和 keyread 五个实验臂。应同时观察旧事实
保留、同值新 Key 的学习、真正新事实的学习，以及计算成本。只看到平均写入量
下降，还不能判断好坏；少学新东西也可能让旧记忆显得更稳定。

本次补充的 `--diagnostics` 在 reference 评估结果中保存 `gate_events`，分别
记录 initial_fact、alias_fact、new_fact 及各类 answer 位置上的 rho、cosine、
读取/Value 范数、未抑制/实际更新范数和 guard 调整，并保留 sum/count。
这样可以检查“少写主要发生在哪里”，再与相应学习指标一起解释。

答案的评分 logit 在前一个位置，答案 token 自身的写入在后一个位置；诊断已
明确区分二者。事件标签只供评估使用，不输入门控。alias 当前是紧邻查询，
没有延迟 alias。事件也不是“必要写入”的真值标签：短卷积可能让关联信息
延续到后续 token。

现有数学推导和源码审查支持继续保留、验证这个方法。单元检查、CUDA parity
和训练收益仍需按 [LOCAL_AGENT_RUNBOOK.md](LOCAL_AGENT_RUNBOOK.md) 在目标环境
执行；当前生成的回归检查不能被记成已经通过的实验结果。
