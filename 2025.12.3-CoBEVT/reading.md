### 总结

**标题**: CoBEVT: Cooperative Bird's Eye View Semantic Segmentation with Sparse Transformers
**核心主题**: 基于稀疏 Transformer 的多智能体、多摄像头协作式鸟瞰图 (BEV) 语义分割框架。



#### 1. 研究背景与动机

* **单车局限性**: 传统的单智能体摄像头系统在复杂的交通场景中，容易受到遮挡和感知距离限制的影响。
* **V2V 的潜力**: 车对车 (V2V) 通信可以通过共享感知信息，显着提升感知性能和范围。
* **现有缺口**: 现有的 V2V 研究主要集中在 LiDAR 传感器上，而基于摄像头的协作感知尚未得到充分探索。



#### 2. 核心方法 (Methodology)

CoBEVT 是首个通用的多智能体、多摄像头感知框架，能够协作生成 BEV 地图预测。其核心组件包括：

* **融合轴向注意力 (FAX Attention)**:
    * 这是论文提出的核心创新模块，旨在高效融合特征。
    * **机制**: 结合了 **3D 局部窗口注意力** (用于捕捉局部细节和像素级对应) 和 **稀疏全局注意力** (用于捕捉长距离上下文依赖和全局语义)。
    * **优势**: 相比全注意力机制，具有更低的计算复杂度，同时保留了强大的表达能力。
* **SinBEVT (单智能体 Transformer)**:
    * 用于处理本车的摄像头数据，生成 BEV 特征。
    * 采用分层结构和 FAX 交叉注意力，相比传统的 CVT 方法，在保留细粒度图像细节和小物体检测上表现更好。
* **FuseBEVT (融合 Transformer)**:
    * 用于在 BEV 空间内融合来自自身和其他车辆（经压缩传输后）的特征。
    * 利用 FAX 自注意力模块，处理跨智能体的局部和全局交互。



#### 3. 实验结果

* **OPV2V 摄像头赛道**: CoBEVT 取得了最先进 (SOTA) 的性能。相比单智能体基线提升了 22.7%，相比领先的多智能体融合模型（如 DiscoNet, V2VNet）也有显著提升。
* **泛化能力**:
    * **LiDAR 任务**: 框架展示了跨模态的通用性，在 OPV2V 的 LiDAR 3D 检测任务上也达到了 SOTA。
    * **单车任务**: SinBEVT 在 nuScenes 数据集上的单车 BEV 分割任务中，以实时推理速度超越了之前的 SOTA 模型。
* **鲁棒性与效率**:
    * 在摄像头丢失（Sensor Dropout）的情况下仍能保持较高的安全性。
    * 对数据压缩不敏感，即使在 64 倍压缩率下也能保持高性能。
    * 实现了实时的推理速度。



#### 4. 结论

CoBEVT 提供了一个高效、灵活且高性能的解决方案。它证明了利用稀疏 Transformer 进行 V2V 协作感知可以显著提升自动驾驶在复杂环境下的感知能力，不仅适用于多摄像头系统，也能扩展到 LiDAR 系统和单智能体任务中。





### Fused Axial Attention (FAX) 理解

先回忆注意力输入是 $(b,n,d)$，（b:batch_size. n:序列长度, d:特征数）

batch_size无关，根据 n,d 内容输出



FAX分两种Attention，Local Attention 和 Global Attention

- Local Attention：
  - 做法：把图按空间分块，按空间顺序组合块，理解块内关系，见 图2 的红色块
  - 作用：理解细节，理解一个整体
- Global Attention：
  - 做法：把图按空间分块，取每块的第(x,y)个像素组成一组，理解组内关系，见 图2 的蓝色块
  - 作用：组是离散的，与全图每一块有关的，可以理解整体关系。如图2(b)，组的关系可以帮助不同传感器的图对正位置

造成两者效果不同的本质原因是，**数据排列不同了**

下面用数据流理解文中两种Attention



#### Local Attention

1. 窗口划分 (Window Partition) 把 $H\times W$ 的图切成 $P \times P$ 的小块。 
   -  操作：Reshape 
   -  Shape: $(N, H, W, C)$ $\rightarrow (N, \frac{H}{P}, P, \frac{W}{P}, P, C)$    
2. 维度重排 (Permute) —— 关键步骤 把代表“窗口位置”的维度放到一起，把代表“窗口内部内容”的维度放到一起。 
   -  操作：Permute / Transpose 
   -  Shape:  $(N, \frac{H}{P}, P, \frac{W}{P}, P, C) \rightarrow (\frac{H}{P}, \frac{W}{P}, N, P, P, C)$    
   -  这里我们将 $\frac{H}{P}$ 和 $\frac{W}{P}$ 移到了前面（作为 Batch 维度）。 
3. 准备 Attention 输入 (Flatten for Attention) 操作：
   - Reshape  $(\frac{H}{P}, \frac{W}{P}, N, P, P, C)$ $\rightarrow (\frac{H}{P}*\frac{W}{P}, NP^2, C)$    $\rightarrow (\text{Batch}, \text{Seq\_Len}, \text{Dim})$    
   -  $Batch = \frac{H}{P}*\frac{W}{P}$: 总共有 $\frac{H}{P}*\frac{W}{P}$ 个物理窗口（位置）。    
   -  $SeqLen = NP^2$ : 每个窗口里有 $N$ 个智能体 $\times$ $P \times P$ 个像素 组成的 Token。    
4. **含义**: 把图切成 $P \times P$ 块，再与 $N$ 拼，然后一整块作为一组



#### Global Attention

1. 网格划分 (Grid Partition) 同样把图切分，但在逻辑上我们准备进行间隔采样
   -  操作：Reshape 
   -  Shape: $(N, H, W, C)$ $\rightarrow (N, G ,\frac{H}{G}, G, \frac{W}{G},  C)$    
2. 维度重排 (Permute) —— 关键步骤 把代表“窗口位置”的维度放到一起，把代表“窗口内部内容”的维度放到一起。 
   -  操作：Permute / Transpose 
   -  Shape:  $(N, G ,\frac{H}{G}, G, \frac{W}{G},  C) \rightarrow  (\frac{H}{G},\frac{W}{G},N,G,G,C)$    
   -  这里我们将 $\frac{H}{P}$ 和 $\frac{W}{P}$ 移到了前面（作为 Batch 维度）。 
3. 准备 Attention 输入 (Flatten for Attention) 操作：
   - Reshape  $(\frac{H}{G},\frac{W}{G},N,G,G,C)$ $\rightarrow (\frac{H}{P}*\frac{W}{P}, NG^2, C)$    $\rightarrow (\text{Batch}, \text{Seq\_Len}, \text{Dim})$    
   -  $Batch = \frac{H}{G}*\frac{W}{G}$: 总共有 $\frac{H}{G}*\frac{W}{G}$ 个“采样相位”（比如所有 $(0,0)$ 点是一组，所有 $(0,1)$ 点是一组）。    
   -  $SeqLen = NP^2$ : 每个序列包含 $N$ 个智能体 $\times$ 全图 $G \times G$ 个稀疏采样点，组成token。
4. **含义**: 把图切成 $G \times G$ 块，再与 $N$ 拼，然后每块的 块内相对位置相同的的像素作为一组，比如块内像素位置是(0,1)视为一组



### SinBEVT的K,V

K (Key) 和 V (Value) 均来自 多视角相机特征 (Multi-view Camera Features) 。

具体来说，原始的相机图像（例如前、后、左、右四个视角的图像）首先经过一个相机编码器（如 ResNet34）提取特征，这些提取出的图像特征图（Image Features）被用作 Key 和 Value 。

这样做相比低分辨率的BEV图像作为K,V来做self attention效果更好