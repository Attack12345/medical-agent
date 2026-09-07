# 智慧问诊 Agent（MedicalConsultationAgent）

面向"患者就医前咨询"场景的智慧问诊 Agent：自然语言提问（症状、科室、用药、就医指导），基于医疗知识图谱 + 向量检索双路召回，由 LangGraph 编排的 9 个单职责 Agent 协作回答。核心价值：**可信（回答强制溯源）、可控（确定性规则安全层）、可溯源（证据链）**。

## 架构

```
┌─ 交互层 ──────────────────────────────────────────────┐
│  FastAPI (REST + SSE)  │  聊天界面（单页）                │
└──────────────┬───────────────────────────────────────┘
┌──────────────▼──────── 智能体层（LangGraph StateGraph）──┐
│  intent_agent（意图识别）                                 │
│   ├─ 医疗问答 → symptom_agent（实体链接）                  │
│   │     ├─ 科室咨询 → department_agent                    │
│   │     ├─ 用药咨询 → drug_agent                          │
│   │     ├─ 知识查询 → medical_knowledge_agent（grounded） │
│   │     └─ 就医指导 → guide_agent                         │
│   │     → safety_agent（强制汇聚，纯规则）→ fusion_agent   │
│   └─ 通用对话 → chat_agent                                │
└──────────────┬───────────────────────────────────────┘
┌──────────────▼──────── 引擎层 ──────────────────────────┐
│  规则引擎（DSL+执行器+轨迹）   │  grounded-ReAct 强制检索    │
└──────────────┬───────────────────────────────────────┘
┌──────────────▼──────── 数据层 ──────────────────────────┐
│  MySQL（对话/评估）  Neo4j（8 类节点/12 类关系）  Qdrant     │
└───────────────────────────────────────────────────────┘
```

## 快速开始

```bash
# 0. 配置
cp .env.example .env    # 填 DASHSCOPE_API_KEY / MYSQL_PASSWORD

# 1. 建库（MySQL 5 张表）
mysql -uroot -p < sql/schema.sql

# 2. 数据接入（真实数据集下载失败时用 --generate 模拟兜底）
python backend/scripts/prepare_data.py

# 3. 建图（M2，Neo4j infra/docker-compose.yml）
python backend/scripts/build_graph.py

# 4. 向量库（M3，Qdrant infra/docker-compose.yml）
python backend/scripts/build_vector_db.py

# 5. 启动（M6）
uvicorn app.main:app --app-dir backend --port 8090

# 测试
python -m pytest backend/tests
```

## 里程碑状态

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M1 | 骨架+数据接入 | ✅ 完成（8765 疾病 / 19.6 万关系 / 5000 问答，pytest 11 用例绿） |
| M2 | 图谱建图 | ✅ 完成（8770 疾病 / 5988 症状 / 3798 药物 / 19.6 万关系，pytest 24 用例绿） |
| M3 | 检索+基线 | ✅ 完成（实体链接准确率 1.0 / 关系命中 0.96 / 证据覆盖 1.0） |
| M4 | 9 Agent 编排 | ✅ 完成（三类问题端到端 + 问诊 HITL interrupt，pytest 41 用例绿） |
| M5 | 安全层+证据链 | ✅ |
| M6 | API+前端 | ✅ |
| M7 | 金标集+门禁 | ✅ |


