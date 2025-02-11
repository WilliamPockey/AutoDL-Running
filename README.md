# AutoDL-Running
本博客介绍了AutoDL的使用方法和使用时遇到的一些问题

​
## 前言

在深度学习项目中，为应对个人电脑算力不足，我们可以选择租用云平台显卡进行模型训练等工作。本文章主要记录云平台AutoDL使用方法与实战项目遇到的问题。

## AutoDL平台显卡租用流程

在平台登录后，先充值一定数额，然后进入算力市场

![image](https://github.com/user-attachments/assets/7868c879-77d0-487a-b72e-cd9112a3236a)


按照自己的需求选择显卡，如rtx 3090，点击可租的卡

![image](https://github.com/user-attachments/assets/f8ad518f-0b6d-4818-bfa2-eff8ce3c1bbc)


选择想要的配置和项目需要的镜像，小项目建议按量计费，配置以Miniconda为例

![image](https://github.com/user-attachments/assets/655b7d7a-ec2f-4ac7-9ca5-0b3f1fcb9539)


创建后点击个人主页--》左侧栏的容器实例--》右侧的开机(也可以点击更多-->以无卡模式启动来配置环境(按量计费时省钱))

![image](https://github.com/user-attachments/assets/c62c3707-ad56-45c5-9483-a751a4b8185e)


然后点击JupyterLab进入服务器主界面即可

![image](https://github.com/user-attachments/assets/72babf98-2af4-4370-b826-c6994a91bc42)


点击文件夹上的上传按钮可将项目上传至AutoDL服务器，点击右侧页面的cmd可进入控制台
另外autodl-tmp文件夹是数据盘，建议将项目存储在此

点击AutoPanel可以进入服务器监控面板
![image](https://github.com/user-attachments/assets/59e6d015-a679-48dd-bcef-e33ae6701d1b)


可以在这监控硬件性能释放，也可以点击实用工具进行pip/conda换源或清理包
![image](https://github.com/user-attachments/assets/f981b4d5-4e16-4810-900b-675207f4ca0d)



## AutoDL解决conda install卡在collecting package meta或solving envieonments

使用conda install卡在collecting package meta或solving envieonments是因为conda版本较老，需要更新conda版本，并且建议替换成mamba从而加速包下载

方法：在控制台中输入conda update -n base conda 更新conda到最新版本。然后执行：conda update --all 。然后下载mamba替换conda：conda install mamba -n base -c conda-forge
之后的conda命令都用mamba进行替换即可。

## 给AutoDL服务器设置代理

AutoDL在下载外网的包时非常卡顿。如果需要下载github或者huggingface的资源。可使用官方的学术加速。方法：控制台输入source /etc/network_turbo
如果需要取消则输入unset http_proxy && unset https_proxy

但如果需要下载其他网站资源，请首先自备代理软件Clash。本处参考：https://github.com/VocabVictor/clash-for-AutoDL?tab=readme-ov-file

方法：1.在控制台输入git clone https://github.com/VocabVictor/clash-for-AutoDL.git

2.在控制台输入cd clash-for-AutoDL
cp .env.example .env
vim .env(使用vi编辑器修改文件)

3.进入你使用的代理网站，找到订阅链接并复制
![image](https://github.com/user-attachments/assets/62d135f1-398c-402e-b08d-2a5943e46ab8)

4.(此处是为不懂vi编辑器的方便操作)点击(或方向键移动至)第二行的CLASH_URL右侧第一个引号，然后黏贴，使你复制的连接位于两个引号之间，然后键盘输入:wq退出即可
![image](https://github.com/user-attachments/assets/accb1ab2-6f0a-4077-83a9-01b8fafe75fc)



5.在同个目录下输入apt-get update
apt-get install lsof

6.运行启动脚本

source ./start.sh

看到下面的输出只要有“网络连接测试成功”即实现了代理

配置文件已存在，无需下载。
配置文件格式正确，无需转换。

正在启动Clash服务...
服务启动成功！                                             [  OK  ]

Clash 控制面板访问地址: http://<your_ip>:6006/ui
Secret: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

已添加代理函数到 .bashrc。
请执行以下命令启动系统代理: proxy_on
若要临时关闭系统代理，请执行: proxy_off
若需要彻底删除，请调用: shutdown_system

[√] 系统代理已启用
正在测试网络连接...
网络连接测试成功。

7.如果未出现“网络连接测试成功”且你的输出中有envsubst: command not found

需要在控制台输入sudo apt-get update
sudo apt-get install gettext

然后再次输入source ./start.sh即可

原因:隔壁gpu.pro出现envsubst: command not found的解决方法 · Issue #37 · VocabVictor/clash-for-AutoDL

## (个人向)运行时报错"undefined symbol: iJIT_NotifyEvent"解决方法

原因：mkl包太新，而pytorch是基于老版本mkl写的

方法：mamba install mkl=2024.0

## (个人向)解决torch-geometric报错的问题

### 1.报错"torch_geometric AttributeError: 'builtin_function_or_method' object has no attribute 'default'"

原因：torch-geometric版本太高

方法：pip install torch-geometric==1.7.2

### 2.在解决1后，运行时报错RuntimeError: object has no attribute sparse_csc_tensor

原因：虽然torch-geometric版本降低了，但其依赖的torch_cluster、torch_sparse、torch_scatter、torch_spline的版本仍然与pytorch、python以及cuda版本不兼容

方法：进入data.pyg.org/whl/torch-1.9.1+cu111.html

将网址上的1.9.1改成自己的pytorch版本，111改成自己的cuda版本

根据自己的python版本，将四个包进行下载(以python3.8为例，网页中的cp38表示python3.8)

![image](https://github.com/user-attachments/assets/5ac2bea7-1a59-4d9f-a6d5-0a03bdef7844)


下载好后将其上传至当前文件夹，然后在控制台输入pip install torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl，即可重新安装适配当前版本的torch_spline包，其他三个包以此类推。然后运行程序即可。

​
