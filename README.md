# 监控图片文字去除
**流程：文字检测(PaddleOCR) -> 掩膜(Mask) -> 图像修复(Image Inpainting)**
## 1. 效果：
## 2. 环境：
```
git clone 
```
**1. PaddleOCR:**
```
conda create -n RemoveText python=3.6
conda activate RemoveText
conda install paddlepaddle-gpu==2.1.3 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
pip install "paddleocr>=2.0.1" -i https://mirror.baidu.com/pypi/simple
```
如有问题参考：https://github.com/xmyyzy123/PaddleOCR/blob/release/2.3/README_ch.md
**2. lama:**
```
cd lama
conda install pytorch=1.9.1 torchvision=0.2.2 torchaudio cudatoolkit=10.2 -c pytorch -y
pip install -r requirements.txt -i https://mirrors.ustc.edu.cn/pypi/web/simple
```
## 3. 运行：
### 3.1 下载文字检测模型：

### 3.1 生成掩膜：
```
cd RemoveText
python gen_mask.py
```
### 3.2 下载模型：

解压后将big-lama放入lama文件夹下
### 3.3 修改`lama/configs/prediction/default.yaml`:
指定图片格式 `image_suffix`, e.g. `.png` or `.jpg`
### 3.4 修改`lama/saicinpainting/evaluation/data.py`:
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
## 3. 训练：
如果模型效果达不到你的要求，可重新训练。
### 3.1 训练PaddleOCR检测模块：
#### 3.1.1 (可选)数据格式转换(VOC->paddleOCR)：
因为开始的时候是使用labelImg标注的文字框，所以需要将labelImg格式VOC格式的数据转换为PaddleOCR格式的数据格式。如果已经使用Paddle标注的话，可以跳过这一步。   
`说明：因为只需要检测文字的位置，所以转换的时候统一将文字内容都设置成了"test"`
1. 将labelImg标注的`.jpg` 和 `.xml`放在同一个文件夹下
2. 打开`voc2paddleocr.py`:
```
# 绝对路径
xml_img_path = 'path/to/your/xml jpg/folder' # 指定为你的图片和xml所在的文件夹路径
paddleocr_root_path = 'path/to/your/generating/paddleocr/format/folder'# 指定为要生成PaddleOCR格式的数据路径，会自动生成此文件夹
```
3. 运行`voc2paddleocr.py`，转换数据集格式：
```
python voc2paddleocr.py
```
4. 修改`PaddleOCR/configs/det/myconfig_db.yml`中训练集和验证集路径为上一步3中生成的PaddleOCR格式的数据集路径：
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
    data_dir: /home/datasets/paddleocrFormat # 数据集路径paddleocrFormat
    label_file_list:
      - /home/datasets/paddleocrFormat/val_label.txt # 标签
```
5. 训练：
```
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/myconfig_db.yml -o Global.pretrained_model=./pretrain_models/ResNet50_vd_ssld_pretrained/ResNet50_vd_ssld_v2_pretrained
```
 模型最后路径到模型名称(不加.pdparams后缀)
6. 模型导出：
```
python3 tools/export_model.py -c configs/det/myconfig_db.yml -o Global.pretrained_model="./output/det_r50_vd/best_accuracy" Global.save_inference_dir="./output/det_r50_vd_inference/"
```
如有问题参考：https://github.com/xmyyzy123/PaddleOCR/blob/release/2.3/doc/doc_ch/detection.md

## 参考：
* https://github.com/PaddlePaddle/PaddleOCR
* https://github.com/saic-mdal/lama