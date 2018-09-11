---
title: Opencv SIFT特征
author: 5477
layout: post
music-id: 404610
---
31861287  
22743825  
首先展示图像金字塔结构，
 <span class="image right"><img width = "300" height = "400" src="{{ 'assets/images/sift.JPG' | relative_url }}" alt="123" /></span>
 
Octave1中，包含6个图像，size和原始图像是一样的，其中上层是高斯滤波的参数，$\sigma$,$k\sigma$,$k^2\sigma$,
之后高斯平滑后的两幅图像相减得到DOG space(difference of gaussian) 中的图像，在Octave2中，Gaussian space 
图像中，size为Octave1的一半，下采样产生，高斯参数为$2\sigma$,$2k\sigma$,$2k^2\sigma$,Octave3类似。
此外Octave2中的初始图像由Octave1中倒数第3张图像下采样而来，

Octave的层数$n=log_2{min(M,N)}-t$, $t \in [0,log_2{min(M,N)}]$, M,N为图像原始size,n为塔顶图像最小维度的log值。
尺度空间的尺度系数$k=2^{\frac{1}{s}}$,s为每塔的层数,k值有待进一步考究。。。。

一幅图像的尺度空间函数$ L(x,y,\sigma ) $ = $ G(x,y,\sigma )$ $ * $ $I(x,y) $,高斯函数和图像的卷积产生。
其中$G(x,y,\sigma)=\frac{1}{2 \pi \sigma ^2 e^{\frac{-(x^2+y^2)}{2 \sigma ^2}} }$.
$D(x,y,\sigma)=(G(x,y,k \sigma)-G(x,y,\sigma))*I(x,y)=L(x,y,k \sigma)-L(x,y,\sigma)$.

<span class="image right"><img width = "100" height = "100" src="{{ 'assets/images/gauss.jpg' | relative_url }}" alt="error image path " /></span>
 用高斯差分函数去近似尺度不变的高斯拉普拉斯函数$\sigma ^2 \triangledown ^2 G $.
$\sigma \triangledown ^2 G=\frac{\partial G}{\partial \sigma} \approx \frac{G(x,y,k \sigma)-G(x,y,\sigma)}
{k \sigma -\sigma}$,so,$G(x,y,k \sigma)-G(x,y, \sigma) \approx (k-1) \sigma ^2 \triangledown ^2 G $

 
### 极值检测：
<span class="image right"><img width = "300" height = "300" src="{{ 'assets/images/extremumdetect.jpg' | relative_url }}" alt="error image path " /></span> 

中间的检测点和它同尺度的8个相邻点和上下相邻尺度对应的9×2个点共26个点比较，以确保在尺度空间和二维图像空间都检测到极值点,为了在每组中检测S个尺度的极值点，
则DOG金字塔每组需S+2层图像，而DOG金字塔由高斯金字塔相邻两层相减得到，则高斯金字塔每组需S+3层图像，实际计算时S在3到5之间。

### 限制一些点 或者说关键点定位
通过拟和三维二次函数以精确确定关键点的位置和尺度（达到亚像素精度），同时去除低对比度的关键点和不稳定的边缘响应点(因为DoG算子会产生较强的边缘响应)，
以增强匹配稳定性、提高抗噪声能力，在这里使用近似Harris Corner检测器。利用DoG函数在尺度空间的Taylor展开式(拟合函数)为  
$D(x)=D+ \frac {\partial D^T}{\partial x} x+ \frac{1}{2}x^T \frac{\partial ^2 D}{\partial x ^2} x$  
$X=(x,y,\sigma)$后求导等于0，  
得到极值点，$\hat{x} =-\frac{\partial^2 D^{-1}}{\partial x ^2} \frac{\partial D}{\partial x}$.

$D(\hat{x})=D+\frac{1}{2} \frac{\partial D ^T}{\partial x} \hat{x}$.  

$\hat{x}=(x,y,\sigma)^T$,代表相对插值中心的偏移量，当它在任一维度上的偏移量大于0.5时（即x或y或$ \sigma $），意味着插值中心已经偏移到它的邻近点上，
所以必须改变当前关键点的位置。同时在新的位置上反复插值直到收敛,也有可能超出所设定的迭代次数或者超出图像边界的范围，
此时这样的点应该删除.如果$|D(\hat{x})| \ge 0.03$则该特征点就保存，否则就删掉.   
### 消除边缘效应
一个定义不好的高斯差分算子的极值在横跨边缘的地方有较大的主曲率，而在垂直边缘的方向有较小的主曲率。
主曲率通过一个2×2 的Hessian矩阵H求出:  
$$H=
 \begin{bmatrix}
   D_{xx} & D_{xy} \\
   D_{xy} & D_{yy}
  \end{bmatrix}  
$$  
$Tr(H)=D_{xx}+D_{yy}= \alpha + \beta $  
$Det(H)=D_{xx}D_{yy}-(D_{xy})^2= \alpha \beta$  
设 $\alpha= \gamma \beta$,  
$\frac{Tr(H)^2}{Det(H)}=\frac{(\alpha +\beta)^2}{\alpha \beta}=\frac{(r+1)^2}{r}$  
(r + 1)2/r的值在两个特征值相等的时候最小，随着r的增大而增大，因此，为了检测主曲率是否在某域值r下，只需检测
$\frac{Tr(H)^2}{Det(H)}< \frac{(r+1)^2}{r}$,论文中r=10.

###确定方向  
每幅图中的特征点，为每个特征点计算一个方向，依照这个方向做进一步的计算， 利用关键点邻域像素的梯度方向分布特性
为每个关键点指定方向参数，使算子具备旋转不变性。
Lowe建议描述子使用在关键点尺度空间内$4*4$的窗口中计算的8个方向的梯度信息，共$4 * 4 * 8 =128$维向量表征。  
关键点尺度坐标为 $\sigma (o,s)$ $ = \sigma_0 $ $ 2 ^{o+ \frac{s}{S}} $ ,而且 $ o \in [0,1,2,... O-1] $ , $\sigma_0$为基准层尺度，o为
octave的索引,s为组内索引，O为Octave个数，S为Octave内层数,  
取$k=2^{\frac{1}{s}}$,在构建高斯金字塔时，组内每层的尺度坐标按如下公式计算  
$\sigma(s)= \sqrt {(k^s-\sigma_0)^2-(k^{s-1} \sigma_0)^2}$  
$\sigma_0 $为初始尺度，论文中$\sigma_0=1.6,S=3 $,s为组内的层索引，不同组相同层的组内尺度坐标$\sigma (s)$相同。  
组内下一层图像是由前一层图像按进行高斯模糊所得，上式用于一次生成组内不同尺度的高斯图像。  
而在计算组内某一层图像的尺度时，直接使用如下公式进行计算：
$ \sigma \\_ oct(s)=\sigma _0 2 ^{\frac{s}{S}}$,$s \in [0,1,2,S+2]$ ,该组内尺度在方向分配和特征描述时确定采样窗口的大小.


$m(x,y)=\sqrt {(L(x+1,y)-L(x-1,y))^2+(L(x,y+1)-L(x,y-1))^2}$  
$\theta (x,y)=atan2(\frac{L(x,y+1)-L(x,y-1)}{L(x+1,y)-L(x-1,y)})$
L为关键点所在的尺度空间值，按Lowe的建议，梯度的模值m(x,y)按$\sigma =1.5 \sigma \\_oct $的高斯分布加成，
按尺度采样的$3 \sigma $原则，邻域窗口半径为$3 *1.5 \sigma \\_oct $.  

    
在完成关键点的梯度计算后，使用直方图统计邻域内像素的梯度和方向。梯度直方图将0~360度的方向范围分为36个柱(bins)，其中每柱10度。
如图所示，直方图的峰值方向代表了关键点的主方向，(为简化，图中只画了八个方向的直方图)
<span class="image right"><img width = "400" height = "200" src="{{ 'assets/images/keypoint.jpg' | relative_url }}" alt="error image path " /></span> 
方向直方图的峰值则代表了该特征点处邻域梯度的方向，以直方图中最大值作为该关键点的主方向。为了增强匹配的鲁棒性，
只保留峰值大于主方向峰值80％的方向作为该关键点的辅方向。因此，对于同一梯度值的多个峰值的关键点位置，
在相同位置和尺度将会有多个关键点被创建但方向不同。仅有15％的关键点被赋予多个方向，但可以明显的提高关键点匹配的稳定性。实际编程实现中，
就是把该关键点复制成多份关键点，并将方向值分别赋给这些复制后的关键点，并且，离散的梯度方向直方图要进行插值拟合处理，
来求得更精确的方向角度值.
### 描述子

SIFT描述子是关键点邻域高斯图像梯度统计结果的一种表示。通过对关键点周围图像区域分块，计算块内梯度直方图，生成具有独特性的向量，这个向量是该区域图像信息的一种抽象，具有唯一性。
##确定计算描述子所需的图像区域
特征描述子与特征点所在的尺度有关，因此，对梯度的求取应在特征点对应的高斯图像上进行。将关键点附近的邻域划分为
$d*d$(Lowe建议d=4)个子区域，每个子区域做为一个种子点，每个种子点有8个方向。每个子区域的大小与关键点方向分配时相同，即每个区域有个$3 \sigma \\_ oct$子像素，
为每个子区域分配边长为$3 \sigma \\_ oct$的矩形区域进行采样(个子像素实际用边长为$ \sqrt {3 \sigma \\_ oct}$的矩形区域即可包含，但$3 \sigma \\_ oct \ge 6 \sigma_0 $不大，
为了简化计算取其边长为$3 \sigma \\_ oct$，并且采样点宜多不宜少)。考虑到实际计算时，需要采用双线性插值，所需图像窗口边长为$3 \sigma \\_ oct x (d+1)$。
在考虑到旋转因素(方便下一步将坐标轴旋转到关键点的方向)，如图所示，实际计算所需的图像区域半径为：  
$radius=\frac{3 \sigma \\_ oct x \sqrt{2}x (d+1)}{2}$,结果四舍五入取整。    
<span class="image right"><img width = "100" height = "100" src="{{ 'assets/images/rotate.jpg' | relative_url }}" alt="error image path " /></span>   
## 将坐标轴旋转为关键点的方向，以确保旋转不变性
<span class="image right"><img width = "100" height = "100" src="{{ 'assets/images/rotate2.jpg' | relative_url }}" alt="error image path " /></span>    
旋转后邻域内采样点的新坐标为：  
$$
 \begin{bmatrix}
   x^'  \\
   y^'
  \end{bmatrix}  
$$  =
$$
 \begin{bmatrix}
   cos \theta & -sin \theta  \\
   sin \theta & cos \theta
  \end{bmatrix}  
$$  
$$
 \begin{bmatrix}
   x  \\
   y
  \end{bmatrix}  
$$  $x,y \in [-radius,radius]$  
##将邻域内的采样点分配到对应的子区域内，将子区域内的梯度值分配到8个方向上，计算其权值。
旋转后的采样点坐标在半径为radius的圆内被分配到的$ dxd $子区域，计算影响子区域的采样点的梯度和方向，分配到8个方向上。

旋转后的采样点$(x^',y^')$落在子区域的下标为:  
$$
 \begin{bmatrix}
   x^'  \\
   y^'
  \end{bmatrix}  
$$  = \frac{1}{3 \sigma \\_ oct}
$$
 \begin{bmatrix}
   x^'  \\
   y^'
  \end{bmatrix}  
$$  +\frac{d}{2}  
Lowe建议子区域的像素的梯度大小按$\sigma =0.5d $的高斯加权计算，即
$w=m(a+x,b+y)*e ^ - \frac{ (x^')^2+(y^')^2}{2x(0.5d)^2}$,a,b为关键点在高斯金字塔图像中的位置坐标。  
##插值计算每个种子点八个方向的梯度。



