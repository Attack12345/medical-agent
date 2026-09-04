# 知识图谱 Schema（锁定：8 类节点 / 11 类实际关系）

## 节点

| 标签 | 属性 | 数据来源 |
|---|---|---|
| Disease | name, aliases[], severity(轻/中/重/急症), summary | D1 diseases.json |
| Symptom | name, aliases[] | D1 症状（PRESENTS object） |
| Drug | name | D1 药物（TREATS subject） |
| Department | name | D1 科室（VISITS object） |
| Exam | name | D1 检查（REQUIRES_EXAM object） |
| Food | name | 模拟数据（AVOIDS_FOOD object，真实数据暂无） |
| Population | name | 模拟数据（AFFECTS object，真实数据暂无） |
| Hospital | name | 急症收治（ADMITS_TO object，暂无数据） |

说明：LOCATED_IN 用 Symptom.body_part 属性实现（不建 BodyPart 节点），因此实际关系 11 类。

## 关系（11 类，全部带 source 属性标注数据来源）

| 关系 | 语义 | 方向 | 现实数据量 |
|---|---|---|---|
| PRESENTS | 疾病表现为症状 | (Disease)-[:PRESENTS]->(Symptom) | 5.4 万 |
| TREATS | 药物治疗疾病 | (Drug)-[:TREATS]->(Disease) | 5.9 万 |
| VISITS | 症状就诊于科室 | (Symptom)-[:VISITS]->(Department) | 3.1 万 |
| DIAGNOSES | 检查诊断疾病 | (Exam)-[:DIAGNOSES]->(Disease) | 0（数据缺失） |
| AVOIDS_FOOD | 疾病禁忌食物 | (Disease)-[:AVOIDS_FOOD]->(Food) | 0 |
| AFFECTS | 疾病影响人群 | (Disease)-[:AFFECTS]->(Population) | 0 |
| CONTRAINDICATES | 药物禁忌（人群/疾病） | (Drug)-[:CONTRAINDICATES]->(Population\|Disease) | 0 |
| REQUIRES_EXAM | 疾病需检查 | (Disease)-[:REQUIRES_EXAM]->(Exam) | 3.9 万 |
| COMPLICATES | 疾病并发症 | (Disease)-[:COMPLICATES]->(Disease) | 1.2 万 |
| ACCOMPANIES | 症状伴随 | (Symptom)-[:ACCOMPANIES]->(Symptom) | 0 |
| ADMITS_TO | 急症收治 | (Disease)-[:ADMITS_TO]->(Hospital) | 0 |

关系语义约束（Text2Cypher prompt 用）：
- PRESENTS 只连 Disease→Symptom；TREATS 只连 Drug→Disease；VISITS 只连 Symptom→Department；
  REQUIRES_EXAM 只连 Disease→Exam；COMPLICATES 只连 Disease→Disease；其余关系按上表方向。
