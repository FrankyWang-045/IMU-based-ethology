# 文件说明
preprocess.py           数据预处理
model.py                VAE模型配置


train.py                模型训练入口，每次训练会生成一个exp文件夹
training_dataset.py     训练数据集准备
embed.py                使用当前训练好的模型嵌入latent

evaluate.py             VAE重建质量评估



config.yaml             参数调试入口
utils.py                读取config参数的接口



config路径：



# 运行流程
1. 准备raw_dir中的csv文件和output_dir空文件夹 

- **raw_dir** 经过对齐和上采样到100Hz的csv原始数据
- **output_dir** 所有输出所在根目录

2. 预处理与训练

终端中依次输入
- cd /Your/Script/Path (当前代码仓库所在路径)
- python preprocess.py
- python train.py 
- python embed.py 
