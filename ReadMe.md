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
  - 

  

8月21号更新：

[update utils by yy6768 · Pull Request #1 · yy6768/ZonotopeForRL (github.com)](https://github.com/yy6768/ZonotopeForRL/pull/1)

8月28号issue：

仍然无法正常运行

8月30号：

现在可以正常运行，但是离完整实现还差很多

## TODO

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


## 更新日志

- 8月14号更新
- 8月21号更新
- 8月30号更新