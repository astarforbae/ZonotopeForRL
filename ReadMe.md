# Zonotope

## 实验内容

核心思路：

-  起始阶段将状态空间划分成不同的zonotope
  - 需要`divide_tools`进行某种划分
  - zonotope 两个字段`center_vec`和`generate_matrix`
  - center_vec控制zonotope位置，generate_matrix控制zonotope的形状
  - 需要Kdtree或者rtree？进行存储
- 神经网络接受zonotope为输入，输出线性控制器系数
  - 神经网络actor-critic结构
  - 输出系数通过向量乘法运算

- 更新采用`ddpg`算法进行更新

  - 弄清楚ddpg原理
  - 实现策略梯度上升更新

  



## TODO

#### 核心

- [x] actor-critic

- [x] ddpg算法

- [x] zonotope求解 （updating）

- [ ] Kdtree需要能够判断zonotope


#### 工具

- [x] logger工具类 -> tensorboard
- [x] yaml 读取文件（updating）
- [ ] tensorboard包装类（updating）
- [ ] config / json类配置（updating）

#### misc
- [x] 统一requirement.txt
- [x] 工具类注释补全
- [ ] zonotope分出来

#### 测试

- [ ] pytest组件（updating)
- [ ] 设计实验
- [ ] 随机划分



### 实验

- [ ] 倒立摆（updating)


## 更新日志

### 8月14号更新

初始化仓库

### 8月21号更新

[update utils by yy6768 · Pull Request #1 · yy6768/ZonotopeForRL (github.com)](https://github.com/yy6768/ZonotopeForRL/pull/1)

### 8月30号更新

8月28号issue：

状态空间权重无法动态实现；zonotope仍然有none的情况

8月30号：

现在可以正常运行，但是离完整实现还差很多



### 9.4号更新：

- 阅读Tyler1+和A2I

- 更改core
- 更新test



### 9.10号更新

- 通过提高划分精度解决lineprog问题
- 撰写工具类logger
- 重装tensorflow，撰写tensorboard工具类



### 9.24号更新

- 更改为tensorboardX进行可视化
- 完善logger类
- fix bugs



### 10.2/10.10更新

- 训练
- 完成模型评估

- fix bugs

### 10.17更新 

- 10.14/10.17训练效果不好进行调整

### 10.17-10.26日更新

**工作内容：**

1. 在Pendulum环境上进行了消融实验。
2. 实验中移除了**抽象状态（zonotope）**。
3. 保留了使用DDPG生成的线性控制器策略。
4. 在多组超参数进行实验，实验效果不好

**实验结果：**

结果不理想，获得的奖励范围为-1600到-1000。

### 10.26-10.31日更新

**工作内容：**

1. 创建了新的环境。
2. 实验中添加了Ornstein-Uhlenbeck (OU) 噪声。
3. 引入了`Animator`以便于可视化。
4. 移除了抽象状态和线性控制器生成策略。
5. 对超参数进行了多次调整。

**实验结果：**

- 与上一周相比，训练结果有明显的好转趋势。
- 尽管如此，仍然出现了过拟合的情况。

week13-14**实验数据**：

| Experiment # | 抽象状态 | 线性控制器 | OU 噪声 | 最高奖励 | 最低奖励 | 注释        |
| ------------ | -------- | ---------- | ------- | -------- | -------- | ----------- |
| 1            | ❌        | ✅          | ❌       | -1600    | -1000    | Week 13实验 |
| 2            | ❌        | ❌          | ✅       | -xxxx    | -xxxx    | Week 14实验 |
| ...          | ...      | ...        | ...     | ...      | ...      | ...         |

下一步计划/建议：

1. **调试与优化**：尝试把调优的参数和模型放在抽象域上进行实验
2. **防止过拟合/正则化**：考虑引入dropout或其他正则化策略来防止过拟合
3. 实验数据调整

