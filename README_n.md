Total3D的输出：```layout.mat，bdb_3d.mat，r_ex.mat```
```
layout.mat输出是8行3列，意为世界坐标系下，布局的八个顶点。

r_ex.mat输出是3行3列，意为外参矩阵。

bdb_3d.mat输出多个：
    basis为3行3列，意为一个物体的姿态在世界坐标系下的表示。
    coeffs为1行3列，意为距离物体中心点的三个轴向的偏移，可以形成立方框。
    centroid为1行3列，意为世界坐标系下，物体的三维中心位置。
```
符号含义：

$ori_i$：一个物体的物体坐标系在相机坐标系下的表示。

$loc_i$：一个物体三维中心点在相机坐标系下的表示。

$size_i$：一个物体立方框的大小。

$vis_i$：一个物体对应的图像特征，由Faster Rcnn提取。

表不表示成相机坐标系应该都可以，主要是坐标系要统一。

$\oplus$可能有多种解释

$s_i^{pose} = Embed(ori_i \oplus size_i)$：

$\qquad$
$Embed$是一个索引操作，那$ori_i \oplus size_i$的结果应该是一个整数，而其结果从效果上考虑是所检测物体的8个顶点在世界坐标系中的表示，而非整数。

$\qquad$
是否可以考虑，像VL-T5将$Box~coordinate$的四个顶点通过线形层升维那样，将8个顶点升维作为视觉信息的补充。公式变为

$\qquad$
$s_i^{pose}=Linear(ori_i \oplus size_i)$，此处的$\oplus$代表着去求立方框的8个顶点，先行层将8升维到2048

$s_i^v=FFN(s_i^{vis} \oplus s_i^{pose})$，此处$\oplus$代表逐像素相加

$s_{i,j}^e=FFN(loc_i \oplus loc_j)$，此处表示边信息，可能是由两个物体的向量夹角余弦值表示，然后$FFN()$将其从1维升至768维

$\gamma_{i,j}=\frac{e_{i,j}·exp(W_b(s_{j'}^v \oplus e_{i,j} \oplus s_t^v))}{\sum_{t=1}e_{i,t}·exp(W_b(s_{t'}^v \oplus e_{i,t} \oplus s_t^v))}$：

一个softmax的变形，旨在计算每一个边的贡献率，其中$e_{i,j},e_{i,t}$指向不明，可能是$s_{i,j}^e,s_{i,t}^e$。$s_t^v = s_{O_1}^v \oplus s_{O_2}^v$代表两个target目标，而两个目标是不需要指定的吗？

$loc_i \oplus loc_j = e_{i,j}$

$\gamma_{i,j}=\frac{e_{i,j}·exp(W_b(s_{j'}^v \oplus s^e_{i,j} \oplus s_t^v))}{\sum_{t=1}e_{i,t}·exp(W_b(s_{t'}^v \oplus s^e_{i,t} \oplus s_t^v))}$

## 一张图片要生成几个图(G)
根据输入的36目标框，判断置信度和目标间距离后生成一个全图，在全图中遍历两两节点，分别作为subject和object，其余节点作为surrounding。对每个图的质量进行评分？然后送进GCN还是先全送进GCN，然后对输出评分？还是全部的图都用上，每个图选个子图，每个子图都输出一个结果，对结果进行评分？