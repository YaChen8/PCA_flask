## 基于PCA的人脸识别
主成分分析（Principal components analysis，PCA）是一种通过降维技术把多个变量化为少数几个主成分的统计方法，通过消除数据的相关性，找到一个空间，使得各个类别的数据在该空间上能够很好地分离，是最重要的特征提取方法之一。

本次实验通过PCA来实现人脸的识别功能，并使用Flask框架将最后的功能部署到网页上

## 环境依赖
python 3.8

Flask 2.0.2

## 运行步骤
1. pip install -U Flask  //安装Flask框架环境
2. 启动程序
    run flask_run.py
3. 打开网页进行测试
   URL=http://192.168.199.181:5000/testpca

## 目录结构描述
├── README.md                   // help

├── meanImage.png               // 测试集求出的均值脸

├── diffImage.png               // 测试脸与均值脸之差

├── RigenFace                   // 测试集的特征脸图像

│   ├── 0.jpg

│   ├── 1.jpg

│   …

│   └── 399.jpg            

├── mycode                      //代码

│   ├── FaceDB_orl              //训练数据集，共40x9=360张

│   │   ├── 001

│   │   ├── 002

│   │   …

│   │   └── 040

│   ├── FaceDB_orl_test         //测试数据集，共40x1=40张

│   │   ├── 001

│   │   ├── 002

│   │   …

│   │   └── 040

│   ├── static                  //网页的静态文件

│   │   ├── css                 //css文件

│   │   ├── FaceDB_orl          //在网页展示检测结果

│   │   ├── img                 //html的img资源

│   │   ├── js                  //js文件

│   │   └── uploads             //上传测试脸图片

│   ├── templates  

│   │   ├── hello.html          //功能测试页面

│   │   ├── index.html          //主页面

│   │   └── next.html           //检测结果跳转页面

│   ├── flask_run.py            //Flask框架路由

│   └── pca_feature.py          //PCA算法
