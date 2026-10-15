半监督管线实施计划：

---

## 阶段 1：数据子集选择

- 从全量 recording 中选 4–6 个代表性文件
- 覆盖不同设备、不同小鼠、不同活动水平
- 用白名单模式训练 HSMM

---

## 阶段 2：候选行为单元发现

- 在子集上训练第一层 HSMM，得到 motion primitives
- 可选：在第一层 state 序列上训练第二层 Categorical HMM，得到更高阶 action
- 决定最终标注粒度：primitives 还是 actions

---

## 阶段 3：人工标注

- 为每个候选 state/action 抽取代表性视频片段
- 人工观看并标注语义
- 判断保留、合并、删除
- 输出 state/action 到 behavior 的映射表

---

## 阶段 4：构建行为本体

- 根据标注结果整理最终 behavior 类别列表
- 必须包含一个 "other" 类别
- 确定每个类别的名称和定义

---

## 阶段 5：生成训练数据

- 用映射表把 state/action 序列转换为 behavior 标签
- 过滤短片段和状态切换边界
- 得到 (latent, behavior) 训练样本

---

## 阶段 6：训练分类器

- 用轻量模型学习 latent 到 behavior 的映射
- 按 recording 划分训练/验证
- 处理类别不平衡

---

## 阶段 7：评估与迭代

- 计算每类 F1 和混淆矩阵
- 找出预测不确定的样本
- 人工复核、修正、扩展类别
- 重新训练直到稳定

---

## 阶段 8：部署

- 对新数据提取 latent
- 分类器预测每帧 behavior
- 可选时间平滑后处理

---

## 下一步

先确定两个问题：
1. 标注粒度选 primitives 还是 actions？
2. 是否有足够人力完成一轮人工标注？
