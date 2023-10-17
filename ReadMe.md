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

- 8月14号更新

  初始化仓库

- 8月21号更新

  [update utils by yy6768 · Pull Request #1 · yy6768/ZonotopeForRL (github.com)](https://github.com/yy6768/ZonotopeForRL/pull/1)

- 8月30号更新

- 9月4日更新



8月28号issue：

状态空间权重无法动态实现；zonotope仍然有none的情况

8月30号：

现在可以正常运行，但是离完整实现还差很多



9.4号更新：

- 阅读Tyler1+和A2I

- 更改core
- 更新test



9.10号更新

- 通过提高划分精度解决lineprog问题
- 撰写工具类logger
- 重装tensorflow，撰写tensorboard工具类



9.24号更新

- 更改为tensorboardX进行可视化
- 完善logger类
- fix bugs



10.2/10.10更新

- 训练
- 完成模型评估

- fix bugs

10.17更新 

- 10.14/10.17训练效果不好进行调整