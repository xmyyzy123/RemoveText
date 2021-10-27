# 监控图片文字去除
**流程：文字检测(PaddleOCR) -> 掩膜(Mask) -> 图像修复(Image Inpainting)**

****
快速开始：
[部署](https://github.com/xmyyzy123/RemoveText/tree/release-1.0)

****
## 目录

- [监控图片文字去除](#监控图片文字去除)
  - [目录](#目录)
  - [1. 效果：](#1-效果)
  - [2. 环境配置：](#2-环境配置)
    - [2.1 PaddleOCR:](#21-paddleocr)
    - [2.2 lama:](#22-lama)
  - [3. 运行：](#3-运行)
    - [3.1 下载文字检测模型：](#31-下载文字检测模型)
      - [3.1.1. 复制模型：](#311-复制模型)
    - [3.2 生成掩膜：](#32-生成掩膜)
      - [3.2.1 修改`gen_mask.py`：](#321-修改gen_maskpy)
      - [3.2.2 运行：](#322-运行)
    - [3.3 下载图像修复模型：](#33-下载图像修复模型)
    - [3.4 修改图片格式:](#34-修改图片格式)
    - [3.4 修改代码:](#34-修改代码)
    - [3.4 去除文字：](#34-去除文字)
  - [4. 训练：](#4-训练)
    - [4.1 训练PaddleOCR检测模块：](#41-训练paddleocr检测模块)
      - [4.1.1 (可选)数据格式转换(VOC->paddleOCR)：](#411-可选数据格式转换voc-paddleocr)
        - [4.1.1.1 文件目录：](#4111-文件目录)
        - [4.1.1.2 修改路径:](#4112-修改路径)
        - [4.1.1.3 转换数据集格式：](#4113-转换数据集格式)
      - [4.1.2 修改数据集路径：](#412-修改数据集路径)
      - [4.1.3 训练：](#413-训练)
      - [4.1.4 模型导出：](#414-模型导出)
    - [4.2 训练图像修复模块：](#42-训练图像修复模块)
  - [参考：](#参考)

****
## 1. 效果：

![原图](img/Landscapes.jpg "原图") | ![去文字](img/Landscapes_mask_inpainted.jpg "去文字")
---|---

`说明：因为隐私问题，没用监控图像展示。训练的模型是在自己的数据集上微调的(检测左上角和右下角的文字),可参考 **4.训练** 训练自己的文字检测模型`

****
## 2. 环境配置：
```
git clone --recurse-submodules https://github.com/xmyyzy123/RemoveText.git
```
### 2.1 PaddleOCR:
```
conda create -n RemoveText python=3.6
conda activate RemoveText
conda install paddlepaddle-gpu==2.1.3 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
pip install "paddleocr>=2.0.1" -i https://mirror.baidu.com/pypi/simple
```
如有问题参考：https://github.com/xmyyzy123/PaddleOCR/blob/release/2.3/README_ch.md
### 2.2 lama:
```
cd lama
conda install pytorch=1.9.1 torchvision=0.2.2 torchaudio cudatoolkit=10.2 -c pytorch -y
pip install -r requirements.txt -i https://mirrors.ustc.edu.cn/pypi/web/simple
```
如有问题参考：https://github.com/xmyyzy123/lama/blob/main/README.md

****
## 3. 运行：
### 3.1 下载文字检测模型：
#### 3.1.1. 复制模型：
解压后复制到`RemoveText`文件夹下   
1. 基于[ResNet50_vd_ssld_v2](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_v2_pretrained.pdparams)训练的适用于监控场景的文字推理模型：
链接：https://pan.baidu.com/s/1nWZL31WJvu6ON0_qdroHvA 提取码：02fb 
2. 通用场景的文字检测可以下载(DB算法的)[通用推理模型](https://github.com/xmyyzy123/PaddleOCR/blob/507129207b854b333d575839b8ce4cfa296a1411/doc/doc_ch/models_list.md)
### 3.2 生成掩膜：
#### 3.2.1 修改`gen_mask.py`：
```
original_img_path = '/path/to/your/images/folder' # 要去文字的图片的路径
ocr_det_model_dir = './det_r50_vd_inference' # 推理模型文件夹路径
result_img_path = '/path/to/RemoveText/lama/original_and_mask' # 需指定到lama文件夹内，会自动生成此文件夹
```
#### 3.2.2 运行：
```
cd RemoveText
python gen_mask.py
```
### 3.3 下载图像修复模型：
解压后将`big-lama`放入`lama`文件夹下：
链接：https://pan.baidu.com/s/1fiN6gcqc65UhRG70P7RHNg 提取码：7ox2 
### 3.4 修改图片格式:
修改`lama/configs/prediction/default.yaml`：
指定图片格式 `image_suffix`, e.g. `.png` or `.jpg`
### 3.4 修改代码:
修改`lama/saicinpainting/evaluation/data.py`：
61行
```
self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
```
将`.png`改为`.jpg`
### 3.4 去除文字：
```
cd lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=.
python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/original_and_mask outdir=$(pwd)/output
```
处理结果在`lama/output`文件夹下

****
## 4. 训练：
如果模型效果达不到你的要求，可重新训练。
### 4.1 训练PaddleOCR检测模块：
#### 4.1.1 (可选)数据格式转换(VOC->paddleOCR)：
因为开始的时候是使用labelImg标注的文字框，所以需要将labelImg格式VOC格式的数据转换为PaddleOCR格式的数据格式。如果已经使用Paddle标注的话，可以跳过这一步。   
`说明：因为只需要检测文字的位置，所以转换的时候统一将文字内容都设置成了"test"`
##### 4.1.1.1 文件目录：
将labelImg标注的`.jpg` 和 `.xml`放在同一个文件夹下
##### 4.1.1.2 修改路径:
打开`voc2paddleocr.py`:
```
# 绝对路径
xml_img_path = 'path/to/your/xml jpg/folder' # 指定为你的图片和xml所在的文件夹路径
paddleocr_root_path = 'path/to/your/generating/paddleocr/format/folder'# 指定为要生成PaddleOCR格式的数据路径，会自动生成此文件夹
```
##### 4.1.1.3 转换数据集格式：
运行`voc2paddleocr.py`，转换数据集格式：
```
python voc2paddleocr.py
```
#### 4.1.2 修改数据集路径：
修改`PaddleOCR/configs/det/myconfig_db.yml`中训练集和验证集路径为上一步3中生成的PaddleOCR格式的数据集路径：   
**说明：使用此配置文件是因为算法必须使用DB算法**   
```
Train:
  dataset:
    name: SimpleDataSet
    data_dir: /home/datasets/paddleocrFormat # 数据集路径
    label_file_list:
      - /home/datasets/paddleocrFormat/train_label.txt # 标签
    ratio_list: [1.0]
```
```
Eval:
  dataset:
    name: SimpleDataSet
    data_dir: /home/datasets/paddleocrFormat # 数据集路径
    label_file_list:
      - /home/datasets/paddleocrFormat/val_label.txt # 标签
```
#### 4.1.3 训练：
```
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/myconfig_db.yml -o Global.pretrained_model=./pretrain_models/ResNet50_vd_ssld_pretrained/ResNet50_vd_ssld_v2_pretrained
```
模型最后路径到模型名称(不加.pdparams后缀)   
预训练[ResNet50_vd_ssld_v2](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_v2_pretrained.pdparams)模型可以在[PaddleClas repo 主页中找到下载链接](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.0/README_cn.md#resnet%E5%8F%8A%E5%85%B6vd%E7%B3%BB%E5%88%97)   
网盘也提供了一份：链接：https://pan.baidu.com/s/1mAo86VtUEPR_rasW0Uf1Nw 提取码：5tkh 
#### 4.1.4 模型导出：
```
python3 tools/export_model.py -c configs/det/myconfig_db.yml -o Global.pretrained_model="./output/det_r50_vd/best_accuracy" Global.save_inference_dir="./output/det_r50_vd_inference/"
```
如有问题参考：https://github.com/xmyyzy123/PaddleOCR/blob/release/2.3/doc/doc_ch/detection.md

### 4.2 训练图像修复模块：
参考：[lama](https://github.com/saic-mdal/lama)

****
## 参考：
* https://github.com/PaddlePaddle/PaddleOCR
* https://github.com/saic-mdal/lama
