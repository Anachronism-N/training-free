# Global CFG 与动态路由决策记录

> 日期：2026-07-20

## 1. 30 秒 Global CFG 消融

三组均使用相同 functional-memory retrieval 配置，只改变 few-step CFG：

| 方法 | Subject | Background | Aesthetic | Imaging | Motion | Dynamic |
|---|---:|---:|---:|---:|---:|---:|
| No CFG | **0.93993** | **0.93383** | **0.60116** | 0.63641 | **0.97242** | 0.97778 |
| Fixed CFG 3.0 | 0.90129 | 0.91949 | 0.54980 | 0.63130 | 0.94789 | **1.00000** |
| Dynamic CFG [1.5,3.5] | 0.91730 | 0.92851 | 0.59444 | **0.67497** | 0.95042 | 0.97778 |

Dynamic CFG 实际 scale：min `1.847`、max `3.5`、mean `2.037`。

## 2. 结论

1. 固定 CFG 3.0 明确退化 subject/background/aesthetic/imaging/motion；不保留。
2. Dynamic CFG 相比 fixed CFG 显著恢复多数维度，并将 imaging 提升到最高；说明 memory confidence
   可用于调节 guidance，但当前 scale 区间仍过强。
3. Dynamic CFG 仍明显不如 no-CFG 的 subject/background/aesthetic/motion，只在 imaging 上胜出。
4. 推理开销从约 128 秒/prompt 增至约 212–216 秒/prompt。

因此：

> Global dynamic CFG 是有效的诊断机制，但当前不满足质量/效率门槛，不进入主方法。

后续若保留，只允许使用更接近 1 的窄区间（如 `[1.0,1.8]`）进行一次小型复核；若仍不能超过
no-CFG，则彻底停止 CFG 方向。

## 3. 32-prompt 与 routing 的联合判断

32-prompt结果显示，confidence routing 的一致性收益仅约 `+0.001~0.002`，但 dynamic degree
下降 `0.0375`。Corrected 3-prompt routing 中，各方案各有局部最优，没有全面胜者。

因此动态功能信号是可测的，但当前“连续缩放 memory gate”没有转化为显著质量收益。

## 4. 主线收敛

主方法暂时收敛为：

1. bounded clean full-frame archive；
2. query-conditioned retrieval；
3. uncertainty abstention（absolute confidence + top1/top2 margin）；
4. query-drift/conflict suppression；
5. independent memory attention；
6. no CFG few-step generation。

动态 head 分析保留为：

- 解释和诊断工具；
- memory admission 的辅助信号；
- 不再声称已经优于 PF 的静态分类。

## 5. 下一关键实验

1. Correct vs wrong/shuffled/abstain memory，证明内容特异性；
2. 低 gate + hard abstention，检查 dynamic degree 是否恢复；
3. archive novelty/coverage compression，避免重复相似帧；
4. position-safe memory branch；
5. 只有以上机制显著优于 PF 后，才扩展到更多 prompt/seed。
