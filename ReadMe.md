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

#### 实验

- [ ] 实验接口
- [ ] 

#### 核心

- [x] actor-critic

- [x] ddpg算法

- [ ] zonotope求解 （updating）

- [ ] Kdtree需要能够判断zonotope


#### 工具

- [x] logger工具类
- [ ] yaml 读取文件（updating）
- [ ] tensorboard包装类
- [ ] config类配置

#### misc
- [ ] 统一requirement.txt
- [ ] 工具类注释补全
- [ ] zonotope分出来

#### 测试

- [ ] pytest组件（updating)
- [ ] 设计实验
- [ ] 随机划分


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

阅读Tyler1+和A2I，更改core，实现test