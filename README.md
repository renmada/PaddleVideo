# 【飞桨特色模型挑战赛】Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition

---
## 内容
- [轻量化方法介绍](#轻量化方法介绍)
- [模型精度](#模型精度)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [自动化测试脚本](#自动化测试脚本)

## 轻量化方法介绍
本repo通过精简模型结构、数据增强、模型蒸馏和最小化转静态模型的方法，在inference模型小于10M条件下，在NTU-RGB+D数据集，joint模态，X-sub评测标准，Top1 acc达到 *90.87* 
- 精简模型结构：去掉backbone的l10(第十层TCN_GCN_unit)
- 数据增强：使用mixup方法
- 模型蒸馏：使用dml方法，初始化权重为89.93的Paddlevideo精度CTRGCN的模型
- 最小化导出：使用nn.Sequential定义的模型导出静态图模型尺寸小于nn.Layer定义的导出
## 模型精度
| split | modality |       Top-1       |                                                   checkpoints                                                   |
| :----: | :----: |:-----------------:|:---------------------------------------------------------------------------------------------------------------:|
| cross-subject | joint | 90.95467329025269 | [CTRGCNLiteJoint_best.pdparams](https://pan.baidu.com/s/12PiDBq0psb7tDcwjcl2d8w?pwd=sqbt) |
### 模型和日志
[下载地址](https://pan.baidu.com/s/12PiDBq0psb7tDcwjcl2d8w?pwd=sqbt)
- CTRGCN_ntucs_joint_dml.pdparams：初始化的权重，转至PaddleVideo训练好的[模型](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/CTRGCN_ntucs_joint.pdparams)
- train.log： 日志
- CTRGCNLiteJoint.pd*： inference_model文件
## 数据准备

NTU-RGBD数据下载及准备请参考[NTU-RGBD数据准备](docs/zh-CN/dataset/ntu-rgbd.md)

## 模型训练

### NTU-RGBD数据集训练

- NTU-RGBD数据集单卡训练，启动命令如下：

```bash
# joint modality
python main.py --validate -c configs/recognition/ctrgcn/ctrgcn_lite_ntucs_joint_dml.yaml \
-w data/CTRGCN_ntucs_joint_dml.pdparams
```
## 模型测试

### NTU-RGB+D数据集模型测试

- 模型测试的启动命令如下：

```bash
# joint modality
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_lite_ntucs_joint_dml.yaml \
-w output/CTRGCNLiteJoint/CTRGCNLiteJoint_best.pdparams  
```

## 模型推理

### 导出inference模型

```bash
python3.7 tools/minimal_export_model.py -c configs/recognition/ctrgcn/ctrgcn_lite_ntucs_joint_dml.yaml \
                                -p output/CTRGCNLiteJoint/CTRGCNLiteJoint_best.pdparams \
                                -o inference/CTRGCNLite
```
上述命令将生成预测所需的模型结构文件`CTRGCNLiteJoint.pdmodel`和模型权重文件`CTRGCNLiteJoint.pdiparams`。模型大小为9.9M

- 各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/example_NTU-RGB-D_sketeton.npy \
                           --config configs/recognition/ctrgcn/ctrgcn_lite_ntucs_joint_dml.yaml \
                           --model_file inference/CTRGCNLite/CTRGCNLiteJoint.pdmodel \
                           --params_file inference/CTRGCNLite/CTRGCNLiteJoint.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

输出示例如下:

```
Current video file: data/example_NTU-RGB-D_sketeton.npy
        top-1 class: [58]
        top-1 score: [0.37179583]
```

可以看到，使用在NTU-RGBD数据集上训练好的ST-GCN模型对`data/example_NTU-RGB-D_sketeton.npy`进行预测，输出的top1类别id为`58`，置信度为0.37179583。


### 对比普通的export

```bash
python3.7 tools/export_model.py -c configs/recognition/ctrgcn/ctrgcn_lite_ntucs_joint_dml.yaml \
                                -p output/CTRGCNLiteJoint/CTRGCNLiteJoint_best.pdparams \
                                -o inference/CTRGCNLite_norm_export

python3.7 tools/predict.py --input_file data/example_NTU-RGB-D_sketeton.npy \
                           --config configs/recognition/ctrgcn/ctrgcn_lite_ntucs_joint_dml.yaml \
                           --model_file inference/CTRGCNLite_norm_export/CTRGCNLiteJoint.pdmodel \
                           --params_file inference/CTRGCNLite_norm_export/CTRGCNLiteJoint.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```
输出结果
```
Current video file: data/example_NTU-RGB-D_sketeton.npy
        top-1 class: [58]
        top-1 score: [0.37179583]
```
模型大小14M,输出结果与最小化导出相同

## 自动化测试脚本
详细日志在test_tipc/output

TIPC: test_tipc/README.md

首先安装auto_log，需要进行安装，安装方式如下：

```shell
pip install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
```
进行TIPC：
1. 需完成前面的[数据准备](#数据准备)
2. 运行测试
```shell
bash test_tipc/test_train_inference_python.sh test_tipc/configs/CTRGCNLite/train_infer_python.txt 'lite_train_lite_infer'
```
TIPC结果：
如果运行成功，在终端中会显示下面的内容，具体的日志也会输出到test_tipc/output/文件夹中的文件中。
```
Run successfully with command - CTRGCNLiteJoint - python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_lite_ntucs_joint_dml.yaml --seed 1234 -o DATASET.train.file_path="data/ntu-rgb-d/xsub/train_data.npy" -o DATASET.train.label_path="data/ntu-rgb-d/xsub/train_label.pkl" -o DATASET.valid.file_path="data/ntu-rgb-d/xsub/val_data.npy" -o DATASET.valid.label_path="data/ntu-rgb-d/xsub/val_label.pkl" -o DATASET.test.file_path="data/ntu-rgb-d/xsub/val_data.npy" -o DATASET.test.label_path="data/ntu-rgb-d/xsub/val_label.pkl" --max_iters=30      -o output_dir=./test_tipc/output/CTRGCNLiteJoint/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1 -o epochs=1   -o DATASET.batch_size=32     > ./test_tipc/output/CTRGCNLiteJoint/lite_train_lite_infer/train.log 2>&1 - ./test_tipc/output/CTRGCNLiteJoint/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1.log 
Run successfully with command - CTRGCNLiteJoint - python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_lite_ntucs_joint_dml.yaml -o DATASET.train.file_path="data/ntu-rgb-d/xsub/train_data.npy" -o DATASET.train.label_path="data/ntu-rgb-d/xsub/train_label.pkl" -o DATASET.valid.file_path="data/ntu-rgb-d/xsub/val_data.npy" -o DATASET.valid.label_path="data/ntu-rgb-d/xsub/val_label.pkl" -o DATASET.test.file_path="data/ntu-rgb-d/xsub/val_data.npy" -o DATASET.test.label_path="data/ntu-rgb-d/xsub/val_label.pkl"   -w=./test_tipc/output/CTRGCNLiteJoint/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1/CTRGCNLiteJoint_epoch_00001.pdparams > ./test_tipc/output/CTRGCNLiteJoint/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1_eval.log 2>&1  - ./test_tipc/output/CTRGCNLiteJoint/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1_eval.log 
Run successfully with command - CTRGCNLiteJoint - python3.7 tools/minimal_export_model.py -c configs/recognition/ctrgcn/ctrgcn_lite_ntucs_joint_dml.yaml --save_name inference -p=./test_tipc/output/CTRGCNLiteJoint/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1/CTRGCNLiteJoint_epoch_00001.pdparams -o=./test_tipc/output/CTRGCNLiteJoint/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1 > ./test_tipc/output/CTRGCNLiteJoint/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1_export.log 2>&1  - ./test_tipc/output/CTRGCNLiteJoint/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1_export.log 
Run successfully with command - CTRGCNLiteJoint - python3.7 tools/predict.py --config configs/recognition/ctrgcn/ctrgcn_lite_ntucs_joint_dml.yaml --use_gpu=False --enable_mkldnn=False --cpu_threads=1 --model_file=./test_tipc/output/CTRGCNLiteJoint/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1/inference.pdmodel --batch_size=1 --input_file=./data/example_NTU-RGB-D_sketeton.npy --enable_benchmark=True --precision=fp32 --params_file=./test_tipc/output/CTRGCNLiteJoint/lite_train_lite_infer/norm_train_gpus_0_autocast_null_nodes_1/inference.pdiparams > ./test_tipc/output/CTRGCNLiteJoint/lite_train_lite_infer/python_infer_cpu_gpus_0_usemkldnn_False_threads_1_precision_fp32_batchsize_1.log 2>&1  - ./test_tipc/output/CTRGCNLiteJoint/lite_train_lite_infer/python_infer_cpu_gpus_0_usemkldnn_False_threads_1_precision_fp32_batchsize_1.log 
```