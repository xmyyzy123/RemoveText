# 监控图片去文字部署
conda打包的代码和环境：链接：https://pan.baidu.com/s/11FvurBSLTSdoqFH8DjBcnA 提取码：i2zc 
## 目录结构
```
.
├── ./ch_ppocr_mobile_v2.0_cls_infer                                # 分类模型，PaddleOCR初始化需要，实际并未使用
│   ├── ./ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams
│   ├── ./ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams.info
│   └── ./ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel
├── ./ch_ppocr_server_v2.0_det_infer                                # 检测模型，PaddleOCR初始化需要，官方通用场景模型
│   ├── ./ch_ppocr_server_v2.0_det_infer/inference.pdiparams
│   ├── ./ch_ppocr_server_v2.0_det_infer/inference.pdiparams.info
│   └── ./ch_ppocr_server_v2.0_det_infer/inference.pdmodel
├── ./ch_PP-OCRv2_rec_infer                                         # 识别模型，PaddleOCR初始化需要，实际并未使用
│   ├── ./ch_PP-OCRv2_rec_infer/inference.pdiparams
│   ├── ./ch_PP-OCRv2_rec_infer/inference.pdiparams.info
│   └── ./ch_PP-OCRv2_rec_infer/inference.pdmodel
├── ./det_r50_vd_inference                                          # 检测模型，PaddleOCR初始化需要，基于自己数据集微调的模型
│   ├── ./det_r50_vd_inference/inference.pdiparams
│   ├── ./det_r50_vd_inference/inference.pdiparams.info
│   └── ./det_r50_vd_inference/inference.pdmodel
├── ./environment.yml                                               # 环境包
├── ./envs                                                          # 已经打包好的conda环境
├── ./lama                                                          # lama图像修复算法
├── ./gen_mask.py                                                   # 检测位置生成mask
├── ./inpaint.py                                                    # 调用lama算法
├── ./interface.py                                                  # 界面
```
## 运行
```
cd RemoveText
source envs/bin/activate # 激活环境
python interface.py
```
