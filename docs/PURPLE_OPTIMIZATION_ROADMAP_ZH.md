# Verdant Purple 优化规划（新增）

## 1. 目标与定位

本规划面向 AgentBeats/AAA 赛道中的 **Purple 参评代理（Assessee）**，目标是让 Verdant 在以下维度达到可部署、可评测、可冲奖状态：

1. 稳定完成任务并通过 Green 评测。
2. 输出可验证、可追溯，便于答辩和复现实验。
3. 在“创新性”上有明确叙事和可展示功能。

## 2. 当前基线（已新增）

已在仓库中新增 Purple 基线实现：

1. `src/purple_agent.py`
2. `src/purple_service.py`
3. `config/purple_config.yaml`
4. `run_purple.py`

核心能力：

1. A2A RPC 接口：`tasks/send`、`tasks/get`、`tasks/cancel`
2. 双阶段验证（Dual-pass）
3. 确定性执行 Trace（deterministic trace）
4. 结构化 metrics/artifacts 输出

## 3. 优化路径（按优先级）

### P0（必须完成，决定可用性）

1. 真实环境域名映射  
   在 `config/config.yaml` 的 `environment.url_overrides` 填入 Render 域名映射，确保 Green 评测 URL 可达。
2. 端到端联调  
   Green 调 Purple 至少完成一次多任务评测并产出结果。
3. 失败可解释性  
   对失败任务保存 trace + verification reasons，用于复盘。

### P1（强烈建议，决定得分上限）

1. 任务策略模块化  
   按场景拆分策略（导航、表单、检索、摘要），避免单一模板输出。
2. 自适应重试策略  
   区分“暂时失败”与“确定失败”，减少无效重试。
3. 可靠性增强  
   引入 run-to-run 漂移监控（输出哈希、关键字段一致性）。

### P2（创新点，决定奖项竞争力）

1. Self-Critique 闭环  
   先生成答案，再执行批判检查并最小化修复。
2. Trace 压缩与证据指纹  
   对执行证据做哈希指纹，支持轻量验真。
3. 评测感知模式  
   根据 task metadata 动态切换“速度优先/稳健优先”模式。

## 4. 创新点设计（可展示）

建议在演示中强调三项：

1. **双验证门控（Dual-pass Gate）**  
   语义验证 + 结构验证，失败原因可直接读。
2. **确定性 Trace 机制**  
   同任务可复现关键步骤，便于审计。
3. **评测友好输出协议**  
   与 Green 指标天然对齐（success/efficiency/error recovery）。

## 5. 改造清单（工程落地）

### 已完成

1. Purple 执行器与服务化接口
2. RPC 任务生命周期管理
3. 中文规划文档（本文件）

### 下一步待完成

1. 补充最小化集成测试（Green->Purple）
2. 增加 Render 双服务 Blueprint（Green/Purple 各一）
3. 补充 README 的 Purple 运行和联调说明

## 6. 验收标准（部署前）

满足以下条件可认为达到“可部署状态”：

1. `/health` 正常，`/agent/info` 返回能力声明。
2. Green 同步评测 10+ 任务不崩溃。
3. 失败任务均有 trace 与明确错误原因。
4. 连续 3 次评测结果波动可解释（非随机飘忽）。

## 7. 部署建议（课堂导向）

1. Render 先部署 Purple（端口 8001），再部署 Green（端口 8080）。
2. Green 的评测请求指向 Purple Render URL。
3. 若后续接入 LLM，优先通过环境变量注入密钥，不写入仓库。
