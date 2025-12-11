#### Task

论文：[面向智能网联汽车的车路协同感知技术及发展趋势.pdf](2025.9.7-论文搜集\面向智能网联汽车的车路协同感知技术及发展趋势.pdf) 

总结：[col.md](2025.9.7-论文搜集\col.md) 



#### Cooper

##### 文档

论文： [cooper.pdf](2025.9.22-cooper\cooper.pdf) ，(ICDCS 2019)

AI+个人修正详细理解：[cooper-reading](2025.9.22-cooper\reading.md) 

AI翻译：[cooper_chinese](2025.9.22-cooper\cooper_chinese.md) 



##### 简要理解

**第一次使用点云数据检测**，使用稀疏卷积方法，并且是原始级（使用原始点云数据）融合，设计了 SPOD 点云检测架构

验证了模型 信息传输量 在现有 车辆网络的带宽限制内，提出 ROI 方法压缩带宽

在 KITTI 数据集和自建 T&J 数据集上进行测试，效果很好，尤其是可以测出原先没有测出的车



##### 名词解释

SPOD：看论文的图一，分三块，Point clouds preprocessing & Voxel feature extractor -> Sparse Convolutional Middle Layers -> Region Proposal Network (RPN，就是检测头Detection Head)

ROI：region of interest，看图11，就是选感兴趣的区域，只发送这一区域的点云数据

Sparse Convolutional Middle Layers：由于体素化后的数据中绝大多数体素是空的，使用传统的3D卷积会造成巨大的计算浪费。该模块采用“稀疏卷积”技术，**仅在包含特征的非空体素上进行卷积运算**，从而高效地聚合局部特征，逐步构建出更高级、更抽象的场景特征图（Feature Maps）



#### Who2com

##### 文档

论文：  [Who2com.pdf](2025.9.23-Who2com\Who2com.pdf) 

AI+个人修正详细理解： [Who2com-reading](2025.9.23-Who2com\reading.md) 

AI翻译：[Who2com_chinese](2025.9.23-Who2com\Who2com_chinese.md) 



##### 简要理解

首次系统性提出在带宽受限下的多智能体协同感知与通信学习问题

设计了**三阶段握手通信**机制，并引入可学习的非对称压缩策略

- 三阶段握手通信：见论文图2
  - **Request**：目标智能体生成压缩后的请求消息（query），广播给其他正常智能体。
  - **Match**：正常智能体基于自己的观测生成键值（key），计算与请求消息的匹配分数（value注意力分数），并返回给目标智能体。
  - **Connect**：目标智能体根据匹配分数选择top-k个最佳智能体，从其获取高带宽特征图，并与自身信息融合完成感知任务。
- 非对称压缩：见论文图3
  - 智能体经过一个可学习矩阵G提取query和key，分数计算改成全局计算 $s_{ji}=\mu_j^TW_ak_i$ ，4个参数从左到右指分数、query，学习矩阵W，key。
  - 根据分数选取最佳正常智能体，目标和最佳智能体经过编码器E，拼接，再解码D得到融合图



#### When2com

##### 文档

论文：  [When2com.pdf](2025.9.26-When2com\When2com.pdf) 

AI+个人修正详细理解： [When2com-reading](2025.9.26-When2com\reading.md) 

AI翻译：[When2com_chinese](2025.9.26-When2com\When2com_chinese.md) 



##### 简要理解

该模型不使用全连接通信，而是采用类似于“握手”的机制来修剪无关的连接，过程主要分为三个阶段：

- **请求 (Request)**：智能体 $i$ 将其局部观测 $x_i$ 压缩为一个极小的查询向量 (Query) $\mu_i$ 和一个较大的键向量 (Key) $\kappa_i$ ，具体见公式1。

- **匹配 (Match)**：

  * 决定与谁通信 (Who)：计算智能体 $i$ 的 Query 与其他智能体 $j$ 的 Key 之间的相似度 $m_{i,j}$，具体见公式2（通用注意力），相似度信息会全部互相发送。如果匹配分数高，说明智能体 $j$ 能提供有用的信息。
    
* 决定何时通信 (When)：计算智能体自身的 Query 与自身的 Key 的相似度 $m_{i,i}$。如果自身匹配分数高，说明自身信息已足够，无需进行外部通信。

* **选择 (Select)**：利用前面计算的相似度，修剪掉低分数的连接，构建稀疏的通信图。
  - 通信图 $\bar{M}$ ：将相似度 $m_{i,j}$ 全部结果收集，取softmax，构造出匹配矩阵 $M$ (具体见公式5)，最后取一个阈值 $\delta$ ，$M$ 小于这个阈值 $\delta$ 的元素都置为0，得到新矩阵 $\bar{M}$ ，所得矩阵 $\bar{M}$ 可视为有向图的邻接矩阵，其元素表示何时通信，元素为零表示不与之通信。（可以看图4，很好理解）
* 根据通信图获取 自身 对 agent 的特征图 $f$ ，相似度作为系数，获取的所有特征图累加得到 $f^{all}$ ，与自身 $f_i$ 拼接，再解码D得到融合图



##### 与who2com区别

ai理解：[diff](2025.9.26-When2com\diff.md) 





#### FRLCP

##### 文档

论文：  [FRLCP.pdf](2025.11.25-FRLCP\FRLCP.pdf) 

AI+个人修正详细理解： [FRLCP-reading](2025.11.25-FRLCP\reading.md) 

AI翻译：[FRLCP_chinese](2025.11.25-FRLCP\FRLCP_chinese.md) 



##### 简要理解

本文提出了一个利用**深度强化学习（DRL）**的框架，具体是 **branching DQN** ，两两匹配车辆，分配资源块RB（一个车辆最多用一个RB），最大化车辆对接收到的传感信息的满意度。优化了很多情况。

- **车辆传感数据压缩**：使用区域四叉树压缩信息，参考论文的图2理解
  - 感应范围被递归分解为不同分辨率的正方形块，一个节点代表一个块，节点的儿子就是把这个节点代表的正方块切分成4块，4个子节点分别代表4块其中一块。
  - 每个块的状态为：占用、未占用或未知。覆盖情况：占用>未知>未占用
  - 车辆仅需传输特定的四叉树块，而非整个点云。块的选择也用 RL 学习
- **问题分解与 DRL 建模**
  - 为了解决复杂的联合优化问题，作者将其分解为两个层级的 RL 问题：
    - **路侧单元（RSU）层级：** 负责**车辆关联**（决定谁与谁配对）和**资源块（RB）分配**。
    - **车辆层级：** 负责**内容选择**，即决定传输哪些四叉树块给关联车辆。
  - **复杂度优化**：
    - 原因：RSU层级中，车辆配对可能情况是 $\prod_{n=1}^{\frac{N}{2}}(2n-1)$ ，资源块分配可能情况是 $K^N$ ，其中 $N$ 是车辆个数，$K$ 是资源块块数
    - 解决：使用 **branching DQN**，把配对情况的列举降为线性，具体见图3
    - 车辆配对：创立 $\lfloor \frac{N}{2} \rfloor$ 个 branches 分支，每 $i$ 个分支只有 $j_i=N-2i+1$ 个选择，假如选择 $t$ ，就是除开前面分支选择的车辆，选第一车辆，和 $t+1$ 车辆配对，具体例子见 图3 左边，枚举的可能情况降成 $\sum_{n=1}^{\frac{N}{2}}(2n-1)$ 
    - 资源块分配：创立 $\lfloor \frac{N}{2} \rfloor$ 个 branches 分支，一个branch视作一对车辆对，每辆车一个RB，一共两个RB，枚举 $C(k,2)$ 种可能
    - RB的第 j 个 branch 和 RSU 的第 j 个 branch配对
- **联邦强化学习 (Federated RL)：**
  * **目的：** 上面的训练是RSU独立和车辆独立的，引入RL为了加速训练并利用所有车辆的经验。
  * **机制：** 车辆在本地训练模型，并定期将参数上传给 RSU；RSU 聚合（平均）参数后广播回车辆，更新全局模型。具体见 Algorithm 2



#### MMW-RCSF

##### 文档

论文：  [MMW-RCSF.pdf](2025.11.29-MMW-RCSF\MMW-RCSF.pdf) 

AI+个人修正详细理解： [MMW-RCSF-reading](2025.11.29-MMW-RCSF\reading.md) 

AI翻译：[MMW-RCSF_chinese](2025.11.29-MMW-RCSF\MMW-RCSF_chinese.md) 



##### 简要理解

这篇论文提出了一种针对路侧异步**毫米波雷达**和**相机**的新型**时空同步**方法，旨在解决路侧不同传感器在时间和空间上缺乏统一基准的问题，从而实现更精确的**传感器数据融合**。

时间上对齐雷达和相机的时间，空间上都转换成世界坐标（如carla的坐标格式）

整个流程见图2，分以下五个步骤

* **数据预处理**：雷达和相机的采样率是不一样的，例如相机是25fps (40ms) ，雷达是20HZ (50ms) 。采用线性拟合，就是假设两个采样点之间变化率平均，雷达拟合相机采样率，如图3和公式1所示。
* **基于场景特征的预标定：** 取四个距离相近的车道线角点，依据已知的车道线标准等静态特征，计算出雷达坐标转世界坐标的转换矩阵各系数的值，用计算出的转换矩阵，把相机的像素坐标转成世界坐标
* **多目标跟踪与噪声估计：** 对两个传感器的数据进行独立跟踪，选一条连续轨迹点，用卡尔曼滤波估计各自的固有噪声（空间偏移和速度偏移），最后去噪。
* **虚拟检测线匹配：** 在道路上设置多条“虚拟检测线”，基于车辆通过的时间和车头时距（Time Headway）来匹配雷达和相机中的同一目标，从而**计算粗略的时间和空间偏差**（$\Delta T$ 和 $\Delta Y$）。如公式11所示。
* **时空同步优化模型：** 把雷达点云坐标转换成世界坐标，建立了一个包含12个参数（时间偏差、空间偏移、旋转角、图像缩放比例因子等）的非线性优化模型，通过最小化轨迹误差来微调同步结果。Loss如公式17所示（该公式都转成雷达坐标算Loss）



#### FPV-RCNN

##### 文档

论文：  [FPV-RCNN.pdf](2025.12.1-FPV-RCNN\FPV-RCNN.pdf) 

AI+个人修正详细理解： [FPV-RCNN-reading](2025.12.1-FPV-RCNN\reading.md) 

AI翻译：[FPV-RCNN_chinese](2025.12.1-FPV-RCNN\FPV-RCNN_chinese.md) 



##### 简要理解

这篇文章提出了一种名为 **FPV-RCNN**（Fusion PV-RCNN）的深度特征融合框架，作用如下

- 降低传输的数据量，同时不降低目标检测效果。
- 车辆有定位误差，校正了这个误差



全文核心见图2，图2注意只有到c部分才有数据共享是吗，b部分只是看着像共享，实际是无关的。



- **FPV-RCNN**：核心特征融合框架，提取出图二右下角的4个部分，下面一个一个讲解
  1. Sensor pose：传感器位姿，基本信息
  2. Proposals：锚框，用CIA-SSD方法提出，效果更好，便于后续锚框合并计算预测分数，这些 Proposals 随后在被用来筛选第4部分的关键点特征，只有位于这些 Proposals 内部的特征点会被选中 。
  3. **selected keypoints coordinates**：用于**坐标误差校正**，设计了一个基于最大一致性原则（Maximum Consensus Principle）的定位误差校正模块，利用环境特征（杆、墙、车辆）来修正车辆间的定位偏差，环境特征通过第4部分提取的3D关键点特征经过分类器分类，因为只用于坐标误差校正，因此只共享各类关键点的坐标。
  4. **selected features**：文章提出基于3D物体检测器 PV-RCNN 的框架，先同样提取出BEV特征图，通过Furthest Points Sampling (FPS) 方法先选取**关键点**，把与关键点有关的信息组合组成**关键点特征**，后续融合只用这个关键点特征，达到进一步精简信息的效果，过程见图3。
- **数据压缩**：提炼关键点特征是压缩，后续用**Draco编码**进一步压缩共享信息



#### Coopernaut

##### 文档

论文：  [Coopernaut.pdf](2025.12.2-Coopernaut\Coopernaut.pdf) 

AI+个人修正详细理解： [Coopernaut-reading](2025.12.2-Coopernaut\reading.md) 

AI翻译：[Coopernaut_chinese](2025.12.2-Coopernaut\Coopernaut_chinese.md) 



##### 简要理解

这篇文章介绍了 **COOPERNAUT**，一种用于网联车辆的端到端协作驾驶模型，作用如下

- 限制传输的数据量，模型**类似Transfomer**，应用于点云模型
- 介绍一种**训练模型的方法**，先根据全知视角的正确答案做决策，再逐渐转变为根据现实世界的有限信息推导的答案做决策，参数都拟合全知视角

解释：

- **COOPERNAUT**：见图2，核心特征融合框架，从左到右过程如下
  1. 点云提取，取 $N$ 个点云数据，一个点包含3个元素，用于记录坐标，
  2. **Point Encoder**：处理点云的编码器，保留关键空间信息，内部带Point Transformer
  3. 信息发送：CAV 发送信息，信息有两部分，Point Encoder 提取出的信息 和 点的坐标，点坐标用于坐标转换，因为 ego 和 CAV 点云坐标系不同。为了满足带宽限制，限制 CAV 发送的点的个数上限，限制 ego 可以接收信息的 CAV 的个数。
  4. 信息聚合：先聚合多辆CAV的信息，直接Voxel Pooling，最后和 ego 信息直接拼接
  5. **Rep Aggregator**：类似 Point Encoder 再聚合提炼点云信息
  6. Control Module：回归，作决策
- **训练方法**：见3.4节
  - Behavior Cloning：让模型由专家（完全正确的路线）做决策，参数拟合专家
  - DAgger：让模型按概率选择 专家做决策 或者 学生做决策（局限的视角，只能看到发送到自车的CAV信息），都拟合专家的正确决策，选择学生做决策的概率越来越高，见公式（5）
  - 先Behavior Cloning，再DAgger



#### CoBEVT

##### 文档

论文：  [CoBEVT.pdf](2025.12.3-CoBEVT\CoBEVT.pdf) 

AI+个人修正详细理解： [CoBEVT-reading](2025.12.3-CoBEVT\reading.md) 

AI翻译：[CoBEVT_chinese](2025.12.3-CoBEVT\CoBEVT_chinese.md) 



##### 简要理解

最重要是提出了融合轴向注意力 Fused Axial Attention (FAX) 机制，借助图2理解

FAX分两种Attention，Local Attention 和 Global Attention

- Local Attention：
  - 做法：把图按空间分块，按空间顺序组合块，理解块内关系，见 图2 的红色块
  - 作用：可以**理解细节**，理解一整块
- Global Attention：
  - 做法：把图按空间分块，取每块的第(x,y)个像素组成一组，理解组内关系，见 图2 的蓝色块
  - 作用：组是离散的，与全图每一块有关的，可以**理解整体**关系。如图2(b)，组的关系可以帮助不同传感器的图对正位置

造成两者效果不同的本质原因是，**数据排列不同了**，具体如何变的见 reading



CoBEVT各子块的理解，借助图3理解

- FuseBEVT：应用于融合，正常过 FAX 和 FFN 线性层
- SinBEVT：应用于单车提取信息，由FuseBEVT改进
  - 把 FAX 的 K,V 输入改成 原始的相机图像提取的特征图，因为在单车里有原始图像信息，比BEV信息的分辨率更高，最终效果更好
  - 把 FFN 改成 Conv 卷积，降低传输的图像大小，多尺度提取特征



#### V2XP-ASG

##### 文档

论文： [V2XP-ASG.pdf](2025.12.6-V2XP-ASG\V2XP-ASG.pdf) 

AI+个人修正详细理解： [V2XP-ASG-reading](2025.12.6-V2XP-ASG\reading)

AI翻译：[V2XP-ASG_chinese](2025.12.6-V2XP-ASG\V2XP-ASG_chinese.md) 



##### 简要理解

本文用 V2XP-ASG 方法生成对抗性（协作效果最差）环境，方法分为两个核心部分：

- **Adversarial Collaborator Search (ACS)** ：找出导致系统性能最差的**协作**组合
- **Adversarial Perturbation Search (APS)** ：找出导致系统性能最差的**位置扰动**组合

生成对抗性环境，把对抗性环境应用于模型训练，可以**提升模型训练效果**



过程：

- Adversarial Collaborator Search：
  - 环境：全部车配备雷达点云传感器，发送中间特征 $H_i \in R^{H \times W \times C}$ 给自车
  - 更新特征：根据所有的 $N$ 辆车的中间特征 $H$ 在ego坐标系下的位置 $(m,n)$ 的特征 $h_{m,n} \in R^{N \times C}$ ，用 self-attention 来捕获 ego 坐标系下同一空间位置不同智能体的注意力权重 $a_{m,n}$，见公式5，获取更新后的特征 $H^{'}$
  - **获取弱点**：
    - 单体重要性 $s_j$ ：根据公式 6 对注意力权重 $a_{m,n}$ 取全坐标平均，计算智能体 $j$ 的重要性 $s_j$ 
    - 单体弱点：定义重要性的倒数 $1/s_j$ 即为弱点
    - 组合弱点 $w_{\mathcal{I}}$ ：根据公式 7 ，计算每一个大小为 $k$ 的协同组合 $\mathcal{I}$ 的总弱点分数
  - 构建对抗性协作图：
    - 生成概率分布：根据公式 8 对组合弱点取softmax，得到采样概率分布 $p$ 。弱点越大的组合，采样概率越高
    - **选择 $\mathcal{I}^*$** ：根据概率 $p$ 不放回采样 $k_0$ 个组合，检测对抗性目标值，保留对抗性损失最低（攻击效果最好）的组合作为最终的对抗性协作组合 $\mathcal{I}^*$
    - 建图：最终使用这个选定的组合 $\mathcal{I}^{*}$ 构建对抗性协作图，仅允许这些被选中的“弱”智能体与自车共享信息。
- Adversarial Perturbation Search
  - 扰动定义：$\delta_i=(\delta x_i,\delta y_i, \delta \theta_i)$ ，前面是坐标扰动，后面是朝向角度扰动，多智能体扰动就是 $\Delta = \{ \delta_1,\delta_2,...,\delta_m \}$，本文取 $m=3$
  - 遮挡水平：
    - 定义：
      1. 内在遮挡分数：自身被其他某辆车遮挡，得一分
      2. 外在遮挡分数：自身遮挡其他车，根据遮挡的车的个数计分
    - 作用：选取最高的 $m$ 个遮挡水平的智能体**应用扰动**
    - 如此作用的原因：通过这种方式选出的车辆，更有可能处于视觉盲区边缘或导致盲区，微调它们最容易引起感知的混乱
  - 合法扰动集合 $Q$ ：限制扰动范围，均匀生成扰动，检查 $Q$ 去除非法样本（例如可能导致碰撞）
  - 生成候选扰动 $\Delta$ ：用黑盒搜索算法（如贝叶斯优化 BO、遗传算法 GA 或随机搜索 RS）根据历史数据 $\mathcal{D}$，“猜测”并生成一个新的潜在扰动 $\Delta$。**这个潜在扰动投影到 $Q$ 里最近的元素，用这个最近的元素更新覆盖 $\Delta$**，这么做可以快速选出合法扰动
  - **选择 $\Delta^*$** ：根据候选扰动 $\Delta$ 计算对抗损失，计入历史记录 $D$ ，最后选取对抗损失最低（攻击最好）对应的 $\Delta$ 作为 $\Delta^*$



#### V2X-ViT

##### 文档

论文： [V2X-ViT.pdf](2025.12.7-V2X-ViT\V2X-ViT.pdf) 

AI+个人修正详细理解： [V2X-ViT-reading](2025.12.7-V2X-ViT\reading)

AI翻译：[V2X-ViT_chinese](2025.12.7-V2X-ViT\V2X-ViT_chinese.md) 



##### 简要理解

提出V2X感知框架，环境有时延，位姿有错误，智能体异构（有车和路边设施）

- 特征融合以外的优化：
  - 元数据共享：部分数据大小很小，如位姿，这些元数据的传输是可以良好同步的，智能体可以互相共享元数据，具体是其他智能体接收到 ego 位姿后，讲自己的 LiDAR 点云**投影到自车坐标系**
  - 特征提取，压缩共享：使用 PointPillar ，提取的特征经**压缩通道维**后发送给 ego ，ego 内部解压
  - 时空校正：
    - 在 V2X 通信中，从数据采集到自车接收并处理数据存在时间差 $\Delta t$。
    - 全局错位（Global Misalignment）：自车在这段时间内移动了，导致坐标系变了。这个问题由 **STCM** 通过空间扭曲来修正。
    - 局部错位（Local Misalignment）：被检测的物体（如其他车辆）在这段时间内也移动了。使用 **Delay-aware Positional Encoding (DPE)** 修正，在reading.md 内部详细讲解 DPE 设计
- 中期特征融合方法**V2X-ViT**：对照图3理解，全文关键
  - 基于视觉 Transformer，有两个重要的注意力块，都是有多头注意力
  - **Heterogeneous multi-agent self-attention (HMSA)** ：图3 (b)
    - 设计原因：**车辆和基础设施**的传感器收集的数据特征是完全不同的，也就是**异构**的，而传统的 self-attention 把它们视为同质的
    - 作用：解决车辆和基础设施传感器收集的数据的异构性
    - 解决方法：**车辆和基础设施分别使用独立的线性层**，例如 Q,K,V 由 中期特征 $H_i$ 独立的线性层获取（ 公式 2,3,7 ）。由于Q，K可能是异构的，Attention 的分数计算修改为公式 3，根据QK是车辆还是基础设施，在QK间乘不同的可学习权重矩阵 $W^{m,ATT}_{\phi(e_{ij})}$；同样，V 的获取也要乘上可学习权重矩阵 $W^{m,MSG}_{\phi(e_{ij})}$ (公式 7 )
  - **Multi-scale window attention (MSwin)** ：图3 (c)
    - 设计：**多尺度窗口**，图3 (c) 很清晰地展示了，各窗口地结果用 Split-Attention 自适应融合各个窗口的信息
    - 大窗口：捕捉长距离视觉线索，能够“容忍”较大的定位偏差。
    - 小窗口：保留精细的局部上下文信息



#### MMVR

##### 文档

论文： [MMVR.pdf](2025.12.8-MMVR\MMVR.pdf) 

AI+个人修正详细理解： [MMVR-reading](2025.12.8-MMVR\reading)

AI翻译：[MMVR_chinese](2025.12.8-MMVR\MMVR_chinese.md) 



##### 简要理解

本文优化：

- Multi-Model Virtual-Real Fusion：优化LiDAR点云信息，通过RGB图给实例的点云加点
- Heterogeneous Graph Attention Network：同 V2X-ViT，解决智能体异构的中期融合



具体解释：

- **Multi-Model Virtual-Real Fusion**
  - 作用：在LiDAR点云中给 实例 添加虚拟点，让 实例 拥有的点相对密集，更容易被 PointPillars 捕捉
  - 方法：有RGB图，RGB 图像分辨率高、纹理丰富，先跑 2D 检测算法，获取掩码 $M_{x,b} $ （第x张RGB图第b个检测对象）。有点云信息，把点云映射到RGB图上，根据落入掩码的点生成虚拟点，具体生成方法见 2.A节 最后一段





#### DAIR-V2X

##### 文档

论文： [DAIR-V2X.pdf](2025.12.11-DAIR-V2X\DAIR-V2X.pdf) 

AI+个人修正详细理解： [MMVR-reading](2025.12.8-MMVR\reading)

AI翻译：[MMVR_chinese](2025.12.8-MMVR\MMVR_chinese.md) 



##### 简要理解

