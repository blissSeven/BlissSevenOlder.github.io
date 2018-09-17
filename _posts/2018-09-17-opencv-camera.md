---
title: Opencv 立体几何
author: 5477
layout: post
music-id: 404610
---

# 立体几何

## 相机模型

![](/assets/images/0917/计算机视觉框架.JPG)



### 射影空间

欧式空间与射影空间相比，射影空间引入了无穷远点。

在射影空间中用齐次坐标表示：

1. 非齐次--->齐次：

   	$(x,y)  \Rightarrow \begin{bmatrix}  x\\y\\1 \end{bmatrix} \quad  $ $(x,y,z)  \Rightarrow \begin{bmatrix}  x\\y\\z\\1 \end{bmatrix}$

   opencv中实现为

   ```c++
   //src	Input vector of N-dimensional points.
   //dst	Output vector of N+1-dimensional points.
   void cv::convertPointsToHomogeneous (InputArray  	src,OutputArray  	dst ) 	
   ```

2. 齐次--->非齐次

   $\begin{bmatrix} x\\y\\w \end{bmatrix}  \Rightarrow (x/w,y/w)  $  $\begin{bmatrix} x\\y\\z\\w \end{bmatrix} \Rightarrow (x/w,y/w,z/w)$

   在opencv中实现为

   ```c++
   /*
   src	Input vector of N-dimensional points.
   dst	Output vector of N-1-dimensional points.
   */
   void cv::convertPointsFromHomogeneous 	( 	InputArray  	src,
   		OutputArray  	dst 
   	) 		
   ```

   **齐次坐标在相差一个尺度时等价。无穷远点坐标为** $\begin {bmatrix} x\\y\\z\\0 \end{bmatrix} $



   ![欧式空间](/assets/images/0917/射影空间.JPG)



将三维坐标（在相机坐标系下的坐标）投影到二维图像平面上$\vec{q} =M * \vec{Q} $

$$  \vec{q}=\begin{bmatrix} x\\y\\w \end{bmatrix},M= \begin{bmatrix} f_x &0&c_x\\0&f_y&c_y\\ 0&0&1\end{bmatrix},\vec{Q}=\begin{bmatrix} X\\Y\\Z \end{bmatrix}$$

而将三维坐标（世界坐标系下）投影到二维图像平面上时，需考虑坐标系的变换，需要多增加一个旋转矩阵R和平移矩阵T。

![](/assets/images/0917/3dto2d.JPG)

在学习opencv3 page=565中给出了类似表达：

$\vec{P_0}$为3D点在世界坐标系下坐标，$\vec{P_c}$为该点在相机坐标系下的坐标，旋转矩阵$R=R_x(\psi)*R_y(\varphi)*R_z(\theta)$,为世界坐标系到相机坐标系，$\vec{T}=origin_{object}-origin_{camera}$:

$\vec{P_c}=R*(\vec{P_0}-\vec{T})$

$\vec{T}$前的负号表示与t相反。

![](/assets/images/0917/相机模型.JPG)



### Rodrigues变换

3x3的旋转矩阵可以用一个三维向量$\vec{r}=[r_x,r_y,r_z]$表示，向量方向代表旋转的方向，向量模代表旋转的角度。

$R=cos \theta *I_3+(1-cos \theta)*\vec{r}*\vec{r}^T+sin\theta \begin{bmatrix} 0&-r_z&r_y\\r_z&0&-r_x\\r_y&r_x&0\end{bmatrix}$

$$sin \theta \begin{bmatrix} 0 &-r_z&r_y\\r_z&0&-r_x\\r_y&r_x&0\end{bmatrix}=\frac{R-R^T}{2}$$

opencv中实现，Rodrigues可以双向变换，jacobian参数用于solvePnp,calibrateCamera内部优化。

```c++
void cv::Rodrigues 	( 	InputArray  	src,
		OutputArray  	dst,
		OutputArray  	jacobian = noArray() 
	) 	
```

###  透镜畸变  

1. 径向畸变：

   $x_{corrected}=x(1+k_1 r^2+k_2r^4+k_3r^6)$

   $y_{corrected}=y(1+k_1r^2+k_2r^4+k_3r^6)$

2. 切向畸变

   $x_{corrected}=x+[2p_1xy+p_2(r^2+2x^2)]$

   $y_{corrected}=y+[p_1(r^2+2y^2)+2p_2xy]$

# 单目相机标定

计算的参数为M R T，所以对应点应为世界坐标系下的3D点坐标和图像坐标系下的坐标。

要计算的参数有M（内参矩阵）的$f_x,f_y,c_x,c_y$4个，R（旋转矩阵）可用三个角度表示，3个参数，T有三个$x,y,z$表示平移，共10个参数。

对于一个视图，要求解10个参数，而不同视图的内参矩阵是一样的。

一个视图固定8个参数？？？

至少2个视图。

## 棋盘标定

opencv中实现：

当找到所有的角点时才返回TRUE。

| bool cv::findChessboardCorners                               | (    | InputArray    | *image*,                                                     |
| ------------------------------------------------------------ | ---- | ------------- | ------------------------------------------------------------ |
|                                                              |      | size          | *patternSize*,                                               |
| *patternSize* 为棋盘中角点的size，ie:Size(9,6),一行有9个角点，共6行。 |      | [OutputArray] | *corners*,                                                   |
|                                                              |      | int           | *flags* = `CALIB_CB_ADAPTIVE_THRESH+CALIB_CB_NORMALIZE_IMAGE` |
|                                                              | )    |               |                                                              |

当用圆网格标定时，找角点用bool $ cv::findCircleGrid$.

可在此举出上进一步亚像素化角点位置，$cv::cornerSubPix()$



| void cv::drawChessboardCorners | (    | InputOutArray | *image*,          |
| :----------------------------- | ---- | ------------- | ----------------- |
| 可将角点画出                   |      | Size          | *patternSize*,    |
|                                |      | InputArray    | *corners*,        |
|                                |      | bool          | *patternWasFound* |
|                                | )    |               |                   |

## 单应矩阵

$M=\begin{bmatrix} f_x&0&c_x\\ 0&f_y&c_y\\0&0&1\end{bmatrix},W=[R,\vec{t}]$

$\vec{q} =s*M*W*\vec{Q} \Rightarrow$$  \begin{bmatrix} x\\y\\1 \end{bmatrix}=s*M*[\vec{r_1},\vec{r_2},\vec{r_3} ,\vec{t}]* \begin{bmatrix} X\\Y\\0\\1  \end{bmatrix}=s*M*[\vec{r_1},\vec{r_2},\vec{t}]*\begin{bmatrix} X\\Y\\1  \end{bmatrix}$

这里3D点世界坐标中的Z可以人为设置为0。

单应矩阵$H=s*M*[\vec{r_1},\vec{r_2},\vec{t}]$,$\vec{q}=s*H*\vec{Q'}$,s为比例因子，齐次坐标在相差一个比例时等价。

*opencv实现*,这里src dst 均为二维坐标。

| [Mat]cv::findHomography   | (    | [InputArray]  | *srcPoints*, CV_32FC2/vector<Point2f> |
| ------------------------- | ---- | ------------- | ------------------------------------- |
|                           |      | [InputArray]  | *dstPoints*, CV_32FC2/vector<Point2f> |
| *method*                  |      | [OutputArray] | *mask*,                               |
| 0：考虑所有点的重投影误差 |      | int           | *method* = `0`,                       |
| 1：RANSAC                 |      | double        | *ransacReprojThreshold* = `3`         |
| 2：LMEDS                  | )    |               |                                       |

## 相机标定

参数数量讨论：

1. K个图象由于x,y坐标，提供2K个方程/约束。
2. 忽略畸变参数，K个视图有4个内在参数+6K个外参数（3个旋转+3个平移）
3. $2*N*K \ge 6K+4 \Rightarrow  (N-3)K \ge2$
4. 但一幅图像最多提供4个点的信息，无论发现多少角点。
5. 所以N>=2，一般上 至少10张7x8或更大图像。

opencv实现,计算 M，R， T,根据flag可设置参数初值。

| double cv::calibrateCamera | (        | [InputArrayOfArrays]  | *objectPoints*,                                              |
| -------------------------- | -------- | --------------------- | ------------------------------------------------------------ |
|                            |          | [InputArrayOfArrays]  | *imagePoints*,                                               |
|                            | 图像size | [Size]                | *imageSize*,                                                 |
|                            |          | [InputOutputArray]    | *cameraMatrix*,                                              |
|                            |          | [InputOutputArray]    | *distCoeffs*, vector of 4,5,8 ocefficients                   |
|                            | 旋转矩阵 | [OutputArrayOfArrays] | *rvecs*,                                                     |
|                            | 平移矩阵 | [OutputArrayOfArrays] | *tvecs*,                                                     |
|                            |          | int                   | *flags* = `0`,                                               |
|                            |          | [TermCriteria]        | *criteria* = `TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, DBL_EPSILON)` |
|                            | )        |                       |                                                              |

solvePnP计算外参数，R,T

```
bool cv::solvePnP 	( 	InputArray  	objectPoints,
		InputArray  	imagePoints,
		InputArray  	cameraMatrix,
		InputArray  	distCoeffs,
		OutputArray  	rvec,
		OutputArray  	tvec,
		bool  	useExtrinsicGuess = false,
		int  	flags = SOLVEPNP_ITERATIVE 
	) 	
```

其健壮版本为：

```
bool cv::solvePnPRansac 	( 	InputArray  	objectPoints,
		InputArray  	imagePoints,
		InputArray  	cameraMatrix,
		InputArray  	distCoeffs,
		OutputArray  	rvec,
		OutputArray  	tvec,
		bool  	useExtrinsicGuess = false,
		int  	iterationsCount = 100,
		float  	reprojectionError = 8.0,
		double  	confidence = 0.99,
		OutputArray  	inliers = noArray(),
		int  	flags = SOLVEPNP_ITERATIVE 
	) 	
```

## 相机矫正

矫正时指定输入图像中每个像素在输出图像中的位置，即map。

像素原坐标下$[x,y]$,map后坐标$[x',y']$map表示方式,

1. CV_32FC2  noArray()   x',y'=map(x,y) map矩阵中(x,y)的值即为新的坐标值

2. CV_32FC1  CV_32FC1  $x'=xmap(i,j)  \quad y '=ymap(i,j)$,x_map y_map分别为两个map矩阵

3. CV_16SC2 noArray()  同1，但速度快

4. CV_16SC2 CV_16UC1 同3，如果需要更精确值，内插信息在CV_16UC1中

   可用下相互转换。

   ```c++
   void cv::convertMaps 	( 	InputArray  	map1,
   		InputArray  	map2,
   		OutputArray  	dstmap1,
   		OutputArray  	dstmap2,
   		int  	dstmap1type,
   		bool  	nninterpolation = false 
   	) 	
   	
   ```

计算map，返回map1,map2。

```c++
void cv::initUndistortRectifyMap 	( 	InputArray  	cameraMatrix,
		InputArray  	distCoeffs,
		InputArray  	R,
		InputArray  	newCameraMatrix,
		Size  	size,
		int  	m1type,
		OutputArray  	map1,
		OutputArray  	map2 
	) 	
```

newCameraMatrix在处理立体图像时，用到，将在矫正前将图像更改为在不同相机的不同内在参数下的图像。

矫正图像：

```c++
void cv::remap 	( 	InputArray  	src,
		OutputArray  	dst,
		InputArray  	map1,
		InputArray  	map2,
		int  	interpolation,
		int  	borderMode = BORDER_CONSTANT,
		const Scalar &  	borderValue = Scalar() 
	) 	
```

直接矫正一幅图像,将不再计算map，仅针对一幅图像直接矫正。

```c++
void cv::undistort 	( 	InputArray  	src,
		OutputArray  	dst,
		InputArray  	cameraMatrix,
		InputArray  	distCoeffs,
		InputArray  	newCameraMatrix = noArray() 
	) 	
```

稀疏矫正，针对一些点进行矫正不再是一幅图像。

```
void cv::undistortPoints 	( 	InputArray  	src,
		OutputArray  	dst,
		InputArray  	cameraMatrix,
		InputArray  	distCoeffs,
		InputArray  	R = noArray(),
		InputArray  	P = noArray() 
	) 	
```

输入点(u,v):
$$
x'' \Leftarrow (u-c_x)/f_x \\
y'' \Leftarrow (v-c_y)/f_y \\
(x',y')=undistort(x'',y'',distCoeffs)  \\
[X,Y,W]=R*[x',y',1]^T  \\
x \Leftarrow X/W\\
Y \Leftarrow Y/W \\
if\quad P\quad is\quad noArray()\\
u' \Leftarrow xf'_x+c'_x\\
v' \Leftarrow yf'_y+c'_y
$$




# 参考文献

1.计算机视觉课件

2.学习opencv3中文版

