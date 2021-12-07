# lstm
用于lstm预测股票模型

##########下载沪深300历史数据######
https://blog.csdn.net/a19990412/article/details/82733291


###########虚拟环境################
激活可以运行代码的虚拟环境
conda activate xxx

对于conda安装的包使用下面命令生成yml文件：
conda env export > xxx.yml

在目标电脑使用下面明亮克隆conda安装的包：
conda env create -f  xxx.yml

对于pip安装的包使用如下命令生成txt文件
pip freeze > xxx.txt

在目标电脑使用下面命令进行克隆
pip install -r xxx.txt

对pip和conda安装的包都克隆完就可以跑代码啦！！！
conda创建新环境
conda create -n env_name

conda克隆环境
conda create -n new_env --clone old_env


###############安装tushar################
pip install  -i https://pypi.doubanio.com/simple/  --trusted-host pypi.doubanio.com    --target=c:\users\dk\anaconda3\envs\lstm\lib\site-packages  tushare
