# Graphics-and-Image-processing-platform

该软件分为两部分：图形处理部分和图像处理部分，下面分别进行讲解。

# 图形部分

## 2D 图形

当用户点击左上角的 2D 按钮后，即可对 2D 图形进行处理。

### 选择要绘制的图形

用户可通过蓝色标题“2D Operation”下的第一个下拉框选择想要绘制的图形，如图所示

* 三角形  
 ![](https://ws2.sinaimg.cn/large/006tNc79gy1fn4gfcfe2yj31kw0timzf.jpg)

* 矩形
  ![](https://ws4.sinaimg.cn/large/006tNc79gy1fn4gfc182kj31kw0ti76i.jpg)

### 图形变换

用户可通过蓝色标题“2D Operation”下的四个点选框使图形进行变换，其四个点选框分别为

1. x 方向的平移  
  ![](https://ws2.sinaimg.cn/large/006tNc79gy1fn4gfblje6j31kw0titax.jpg) 
2. y 方向的平移  
  ![](https://ws1.sinaimg.cn/large/006tNc79gy1fn4gfb6gm9j31kw0titax.jpg)
3. 旋转  
  ![](https://ws3.sinaimg.cn/large/006tNc79gy1fn4gfastarj31kw0tijtl.jpg)
4. 缩放  
  ![](https://ws1.sinaimg.cn/large/006tNc79gy1fn4gfaew0cj31kw0tidi2.jpg)

### 设置裁剪窗口

用户可通过蓝色标题“Please choose a clipping algorthm”下的三个按钮选择是否需要裁剪或者选择 Cohen-Sutherland 裁剪算法或者 Liang-Barsky 裁剪算法进行裁剪。由于无论是 CS 算法还是 LB 算法，只要裁剪窗口的位置和大小一定，那么裁剪后的图形也是一样的，因此这里只展示一张图片。

![](https://ws1.sinaimg.cn/large/006tNc79gy1fn4gfa0243j31kw0ti76g.jpg)

如图所示，即为被边长为 15 的裁剪窗口 裁剪后的三角形。

用户还可以通过上述三个按钮下面的点选框来自定义裁剪窗口的大小，如图所示

![](https://ws1.sinaimg.cn/large/006tNc79gy1fn4gf9nmbjj31kw0tidi0.jpg)

上图为被边长为 12.2 的裁剪窗口裁剪时的三角形。

### 设置颜色

用户可通过左下角的三个滑尺来自定义图形的颜色，三个滑尺由上到下分别代表 R、G、B 三个通道，颜色效果如图所示

![](https://ws2.sinaimg.cn/large/006tNc79gy1fn4gf970pbj31kw0timzh.jpg)

## 3D 图形

当用户点击左上角的 3D 按钮后，即可对 3D 图形进行处理。

### 选择要绘制的图形

用户可通过蓝色标题“3D Operation”下的第一个下拉框选择想要绘制的图形，如图所示

* 棱锥
  ![](https://ws4.sinaimg.cn/large/006tNc79gy1fn4gf8t0w2j31kw0ti0uz.jpg)
* 茶壶
  ![](https://ws2.sinaimg.cn/large/006tNc79gy1fn4gf8fafgj31kw0tidjx.jpg)

### 图形变换

用户可通过蓝色标题“3D Operation”下的三个点选框使图形进行变换，其三个点选框分别为

1. 图形距离
  ![](https://ws2.sinaimg.cn/large/006tNc79gy1fn4gf7xmw5j31kw0tiwnv.jpg)
  ![](https://ws3.sinaimg.cn/large/006tNc79gy1fn4gf74qmsj31kw0titb9.jpg)
2. 绕 x 轴旋转
  ![](https://ws2.sinaimg.cn/large/006tNc79gy1fn4gf6ljlbj31kw0tijvk.jpg)
3. 绕 y 轴旋转  
  ![](https://ws2.sinaimg.cn/large/006tNc79gy1fn4gf64gocj31kw0tiq72.jpg)

用户还可通过上述三个点选框下面的下拉框选择渲染方式，除了默认的线渲染外，还有两种，分别为

1. 点渲染  
  ![](https://ws4.sinaimg.cn/large/006tNc79gy1fn4gjmqpmcj31kw0tigpk.jpg)
2. 面渲染  
  ![](https://ws4.sinaimg.cn/large/006tNc79gy1fn4gjmafbij31kw0tiack.jpg)

### 设置裁剪平面

用户可通过蓝色标题“Please choose a clipping algorthm”下的三个按钮选择是否需要裁剪或者采用两个裁剪平面中的任意一种，分别如下图所示

1. 裁剪平面为 (1, 0, 0)
![](https://ws2.sinaimg.cn/large/006tNc79gy1fn4gjlwg31j31kw0tigov.jpg)
2. 裁剪平面为 (1, 1, 0)
 ![](https://ws1.sinaimg.cn/large/006tNc79gy1fn4gjlewoxj31kw0ti41u.jpg)

> 为了充分体现裁剪效果，因此视角向 x 轴正方向做了略微移动。

### 设置颜色

和 2D 图形的颜色设置方法相同，只展示一张图片

![](https://ws4.sinaimg.cn/large/006tNc79gy1fn4gjkznubj31kw0tigq6.jpg)

# 图像部分

图像部分的操作流程是

1. 打开一张 RGB 图像
2. 转化为灰度图并显示直方图
3. 直方图均衡化并显示直方图
4. 傅里叶变换并进行巴特沃斯低通滤波
5. 傅里叶反变换
6. 保存所有图像

下面分别讲解

## 打开图像

点击右上角的“1. Open File...”按钮即可打开用户选择的图像

![](https://ws3.sinaimg.cn/large/006tNc79gy1fn4fgkby28j30xc0q8dmz.jpg)

## 转化为灰度图

点击右侧的“rgb2gray”按钮即可将原图转化为灰度图并显示直方图

![](https://ws4.sinaimg.cn/large/006tNc79gy1fn4fhs2390j30xc0q87dv.jpg)

![](https://ws1.sinaimg.cn/large/006tNc79gy1fn4fhxafo1j30sg0fg0sl.jpg)

## 直方图均衡化

点击右侧的“Equalize Hist”按钮即可将灰度图进行直方图均衡化处理并显示处理后的图像及相应的直方图

![](https://ws2.sinaimg.cn/large/006tNc79gy1fn4fj1wfgdj30xc0q8qcp.jpg)

![](https://ws4.sinaimg.cn/large/006tNc79gy1fn4fj5jd37j30sg0fgq2u.jpg)

## 傅里叶变换并进行巴特沃斯低通滤波

点击右侧“DFT”按钮即可对灰度图进行傅里叶变换，使其从空域图像转换为频域图像，然后进行巴特沃斯低通滤波

1. 频域图像
![](https://ws1.sinaimg.cn/large/006tNc79gy1fn4fku512cj30xc0q8wvy.jpg)

2. 滤波后的频域图像
![](https://ws3.sinaimg.cn/large/006tNc79gy1fn4fl7fljij30xc0q87lx.jpg)

## 傅里叶反变换

点击右侧“IDFT”即可对滤波后的频域图像进行傅里叶反变换操作，使其从频域图像转换为空域图像，一边观察滤波效果

![](https://ws3.sinaimg.cn/large/006tNc79gy1fn4fn7r59pj30xc0q8dm3.jpg)

## 保存

点击右上角的“Save File...”按钮即可保存所有图像

![](https://ws1.sinaimg.cn/large/006tNc79gy1fn4ft47otsj31kw0zkth4.jpg)