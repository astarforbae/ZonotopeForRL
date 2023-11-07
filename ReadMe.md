# ZonotopeForRL

## 目录
- [目录](https://github.com/yy6768/ZonotopeForRL/edit/master/ReadMe.md#%E7%9B%AE%E5%BD%95)
- [项目概述](https://github.com/yy6768/ZonotopeForRL/edit/master/ReadMe.md#%E9%A1%B9%E7%9B%AE%E6%A6%82%E8%BF%B0)
- [架构](https://github.com/yy6768/ZonotopeForRL/edit/master/ReadMe.md#%E6%9E%B6%E6%9E%84)
- [如何开始](https://github.com/yy6768/ZonotopeForRL/edit/master/ReadMe.md#%E6%9E%B6%E6%9E%84)
- [TODO](https://github.com/yy6768/ZonotopeForRL/edit/master/ReadMe.md#todo)

## 项目概述

本项目旨在使用抽象强化学习来实现可验证的强化学习。下图是我们的核心流程图：

![Core progress](https://typora-yy.oss-cn-hangzhou.aliyuncs.com/Typora-img/architection.png)

1. 环境状态`state`所在空间被称为环境状态空间`state_space`, 初始化时，核心模块`ZonotopeUtils`会将`state_space`划分为多个`zonotope`子空间，`state`输入`ZonotopeUtils`后查询`zonotope`集合，找到所属的`zonotope`，将(`state`,`center_vec`和`generator matrix`)组成`abs_state`
   - 注：zonotope是高维平行四边形，由关于zonotope的形式化定义请见下图：
     ![[Niklas Kochdumper, 2022] ](https://typora-yy.oss-cn-hangzhou.aliyuncs.com/Typora-img/image-20231107112301574.png)
2. 将`abs_state`输入DDPG算法训练的`Actor`和`Critic`中,最终输出动作`action`
   - `Actor`类似一个超网络，`Actor`类**主体是一个神经网络，输入一组抽象状态`abs_state`,输出一组线性控制器`LinearControl`的参数，输出的线性控制器，再与抽象状态进行点乘运算，输出动作**
   - `Critic`与普通的DDPG算法的Critic没有太大区别，主要功能判断`action`是否良好



## 架构

```
C:.
|   .gitignore                 # 丢弃无关文件
|   main.py 				   # 启动类
|   ReadMe.md                  # 主要文档
|   requirements.txt           # 依赖包
|   tensorboard.bat			   # 用于启动tensorboard的
|
+---core				# 核心包
|   |   agent.py               # 与环境直接交互的agent       
|   |   models.py			   # DDPG的Actor和Critic
|   |   reply_buffer.py        # Reply_Buffer 用于经验再现
|   |   train.py			   # 训练脚本
|   |   zonotope.py			   # Zonotope_utils类
|   |   __init__.py            
|   |
|   +---config
|   |       zonotope.yaml      # zonotope参数配置（TODO）
+---env 
|      cartpole_continuous.py  #连续状态空间的cartpole环境
|      __init__.py             
|
\---utils
    |   config.py              # 全局参数类
    |   logger.py			   # 日志类
    |   misc.py				   # 杂项
    |   read_yaml.py           # 读取yaml文件（TODO）
    |   torch_utils.py         # torch工具
    |   __init__.py
    |
```



## 如何开始

如果你想使用本项目，先：

```bat
git clone https://github.com/yy6768/ZonotopeForRL.git
```

使用pip：

```bat
pip install -r requirements.txt
```

使用conda:

```bat
conda install --file requirements.txt
```







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

- [x] 倒立摆Pendulum（updating)

