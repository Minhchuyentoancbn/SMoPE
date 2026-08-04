# SMoPE 复现进展报告

## 1. 项目目标

本项目围绕论文 **One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning (SMoPE)** 开展复现与后续优化。当前阶段的目标是：

1. 跑通官方代码。
2. 建立稳定可复现的实验环境。
3. 完成三套主数据集 `CIFAR-100 / CUB-200 / ImageNet-R` 的正式复现。

---

## 2. 已完成工作

### 2.1 论文与代码梳理

已完成对论文和官方仓库的初步分析，明确了核心设定：

- Backbone：`ViT-B/16`
- Prompt length：`Np = 25`
- Top-K experts：`K = 5`
- 数据集：`CIFAR-100`、`CUB-200`、`ImageNet-R`
- 主要指标：`FAA`、`CAA`

仓库地址：

- [Minhchuyentoancbn/SMoPE](https://github.com/Minhchuyentoancbn/SMoPE)

本地目录：

- [SMoPE](/Users/lujiayi/Desktop/summer/SMoPE)

### 2.2 本地环境与数据准备

已完成：

1. 克隆官方仓库并建立开发分支 `mps-setup`
2. 使用 Python `3.11.15` 创建本地虚拟环境
3. 安装关键依赖：`torch`、`torchvision`、`timm`、`scikit-learn`、`xlsxwriter` 等
4. 下载并校验 `CIFAR-100`
5. 配置 `sup1k` 预训练权重自动下载

相关文件：

- [MPS_SETUP.md](/Users/lujiayi/Desktop/summer/SMoPE/MPS_SETUP.md)
- [data/cifar-100-python](/Users/lujiayi/Desktop/summer/SMoPE/data/cifar-100-python)
- [pretrained/vit_base_patch16_224_augreg2_in21k_ft_in1k.bin](/Users/lujiayi/Desktop/summer/SMoPE/pretrained/vit_base_patch16_224_augreg2_in21k_ft_in1k.bin)

### 2.3 本地 MPS 适配

原始仓库默认面向 CUDA，无法直接在 MacBook 上运行。已完成以下兼容性改造：

1. 将设备逻辑改为支持 `cuda / mps / cpu`
2. 将大量 `.cuda()` 改为设备无关写法
3. 修复新版 PyTorch 下的 `prompt_loss` 原地操作问题
4. 修复 `MPS` 不支持 `float64` 的问题
5. 修复极小规模实验下调度器边界问题

主要改动文件：

- [learners/default.py](/Users/lujiayi/Desktop/summer/SMoPE/learners/default.py)
- [learners/prompt.py](/Users/lujiayi/Desktop/summer/SMoPE/learners/prompt.py)
- [trainer.py](/Users/lujiayi/Desktop/summer/SMoPE/trainer.py)
- [run.py](/Users/lujiayi/Desktop/summer/SMoPE/run.py)
- [models/vit.py](/Users/lujiayi/Desktop/summer/SMoPE/models/vit.py)
- [models/zoo.py](/Users/lujiayi/Desktop/summer/SMoPE/models/zoo.py)
- [utils/schedulers.py](/Users/lujiayi/Desktop/summer/SMoPE/utils/schedulers.py)

### 2.4 本地验证结果

已在本地 `MPS` 上完成一次 tiny 闭环实验，验证通过以下流程：

1. 训练
2. 保存 checkpoint
3. 加载 checkpoint
4. 验证
5. 导出结果

实验输出：

- [outputs/cifar-100/mps-tiny](/Users/lujiayi/Desktop/summer/SMoPE/outputs/cifar-100/mps-tiny)

结果：

- `Val Acc = 98.5`
- `FAA = 98.5`
- `CAA = 98.5`

结论：

- 本地 `MPS` 可用于调试和小规模验证
- 但正式 `10-task` 复现实验速度过慢，不适合作为主实验平台

---

## 3. 平台切换

### 3.1 放弃本地主平台的原因

虽然本地 `MPS` 已经可以运行，但正式配置在 MacBook 上耗时过长。与此同时，学校 GPU 平台存在容器到期后环境难保留的问题，不适合长期复现实验。

因此，正式复现平台切换为 **AutoDL A100**。

### 3.2 AutoDL 当前状态

在 AutoDL 上，已确认：

1. `cuda:0` 正常启用
2. 训练可正常进入 task 级别循环
3. checkpoint 保存与验证流程正常

同时，对 [experiments/cifar-100.sh](/Users/lujiayi/Desktop/summer/SMoPE/experiments/cifar-100.sh) 做了增强：

1. 自动生成时间戳输出目录
2. 自动保存环境快照
3. 自动保存 `pip freeze`
4. 自动保存训练前后 `nvidia-smi`
5. 自动打包结果目录

---

## 4. 当前正式实验进展

### 4.1 CIFAR-100 10-task

当前主实验已完成：

- 数据集：`CIFAR-100`
- 设置：`10-task`
- 平台：`AutoDL A100`
- repeat：`2`
- 策略：`OnePrompt` / `SMoPE`

最终结果如下：

- `Averaged accuracy for task 10 = 89.03`
- `Total time = 1:36:49`
- `FAA mean = 89.115`
- `FAA std = 0.085`
- `CAA mean = 92.14498809523809`
- `CAA std = 0.08344841269840231`
- `FR mean = 4.05`
- `FR std = 0.09444444444444276`

说明：

- 这里的 `acc` 汇总值与论文口径下的 `FAA` 对齐
- `nvidia-smi` 的训练前后快照已自动保存，便于后续补充显存占用统计

实验已自动写入：

- `results.xlsx`
- `run_info.txt`
- `pip_freeze.txt`
- `nvidia_smi_start.txt`
- `nvidia_smi_end.txt`
- `one-prompt.tar.gz`

这说明 `CIFAR-100 10-task` 的正式复现链路已经闭环，结果可用于后续报告撰写和论文对照。

### 4.2 CUB-200 10-task

第二个主复现实验也已完成：

- 数据集：`CUB-200`
- 设置：`10-task`
- 平台：`AutoDL A100`
- repeat：`2`
- 策略：`OnePrompt` / `SMoPE`

最终结果如下：

- `Averaged accuracy for task 10 = 86.88251318897214`
- `Total time = 0:21:00`
- `FAA mean = 86.9848604775273`
- `FAA std = 0.1023472885551655`
- `CAA mean = 91.30822712744879`
- `CAA std = 0.8689670386353541`
- `FR mean = 4.664100333593253`
- `FR std = 0.36506372605679793`

实验已自动写入：

- `results.xlsx`
- `run_info.txt`
- `pip_freeze.txt`
- `nvidia_smi_start.txt`
- `nvidia_smi_end.txt`
- `one-prompt.tar.gz`

这说明 `CUB-200 10-task` 也已经完成正式复现，可以与论文结果直接对照。

### 4.3 ImageNet-R 10-task

第三个主复现实验也已完成：

- 数据集：`ImageNet-R`
- 设置：`10-task`
- 平台：`AutoDL A100`
- repeat：`2`
- 策略：`OnePrompt` / `SMoPE`

最终结果如下：

- `Averaged accuracy for task 10 = 79.0426513416979`
- `Total time = 1:04:28`
- `FAA mean = 79.0426513416979`
- `FAA std = 0.01967114577668383`
- `CAA mean = 83.57033509624657`
- `CAA std = 0.8504049703577508`
- `FR mean = 5.236530659864623`
- `FR std = 0.40845365910502096`

实验已自动写入：

- `results.xlsx`
- `run_info.txt`
- `pip_freeze.txt`
- `nvidia_smi_start.txt`
- `nvidia_smi_end.txt`
- `one-prompt.tar.gz`

这说明 `ImageNet-R 10-task` 也已经完成正式复现，三套主数据集已经全部跑通。

---

## 5. 三数据集结果总表

下表汇总了三套正式复现实验结果，并与论文 Table 1 的 5-run 平均结果对照。

| Dataset | Our FAA | Paper FAA | Delta FAA | Our CAA | Paper CAA | Delta CAA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-100 | 89.1150 | 89.23 | -0.1150 | 92.1450 | 93.67 | -1.5250 |
| CUB-200 | 86.9849 | 87.43 | -0.4451 | 91.3082 | 91.11 | +0.1982 |
| ImageNet-R | 79.0427 | 79.32 | -0.2773 | 83.5703 | 84.39 | -0.8197 |

论文对应结果来源：

- [One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning](https://arxiv.org/pdf/2509.24483.pdf)

论文 Table 1 中的关键基线结果为：

- CIFAR-100: `FAA 89.23`, `CAA 93.67`
- CUB-200: `FAA 87.43`, `CAA 91.11`
- ImageNet-R: `FAA 79.32`, `CAA 84.39`

---

## 6. 阶段结论

截至目前，可以明确得出：

1. 代码已经从“无法运行”推进到“可正式复现”
2. 本地 `MPS` 环境已经完成验证，适合调试但不适合正式长跑
3. AutoDL A100 已成为当前主实验平台
4. `CIFAR-100 10-task` 基础复现已经完成
5. `CUB-200 10-task` 也已经完成正式复现
6. `ImageNet-R 10-task` 也已经完成正式复现
7. 三套主数据集结果已经可以用于论文对照和课程总结

---

## 7. 差异分析

整体来看，三套数据的复现结果都与论文处于同一量级，说明主流程、数据划分和训练策略总体正确。

关键差异有三点：

1. 论文的对照表是 `5 runs` 平均，而当前正式复现只做了 `repeat=2`，统计稳定性不如论文完整设置。
2. `CIFAR-100` 的 `FAA` 很接近论文，但 `CAA` 偏低更明显，说明中后期任务上的累计平均表现还有小幅差距。
3. `CUB-200` 和 `ImageNet-R` 的 `FAA` 都非常接近论文，且 `ImageNet-R` 的波动更小，说明这套配置在细粒度和风格变化数据上的复现相对稳定。

从结果偏差上看：

- `CIFAR-100` 更像是“主结果接近，但后期累计稳定性略弱”。
- `CUB-200` 基本达到论文水平，`CAA` 甚至略高于论文。
- `ImageNet-R` 的 `FAA` 接近论文，但 `CAA` 仍有一定差距，提示 task-adaptive prediction 或后期校准环节可能还存在实现/超参敏感性。

此外，还要注意一点：

- 论文正式结果是按 `A100`、`5 runs`、作者完整设置报告的
- 你的复现结果来自当前环境和 `repeat=2`
- 因此这些差异更适合作为“复现误差与实现敏感性分析”，不应直接解释为方法优劣变化

---

## 8. 下一步计划

1. 整理三套数据集的复现实验曲线和论文差异分析图
2. 将这份进展报告扩展成课程正式报告
3. 如需优化，再选一套最稳定的数据集做小范围调参实验
4. 最终打包提交结果、日志、环境说明与实验记录

---

## 9. 关键产物

### 代码与文档

- [MPS_SETUP.md](/Users/lujiayi/Desktop/summer/SMoPE/MPS_SETUP.md)
- [REPORT_MPS_PROGRESS.md](/Users/lujiayi/Desktop/summer/SMoPE/REPORT_MPS_PROGRESS.md)

### 配置文件

- [configs/cifar-100_prompt_smope_mps_tiny.yaml](/Users/lujiayi/Desktop/summer/SMoPE/configs/cifar-100_prompt_smope_mps_tiny.yaml)
- [configs/cifar-100_prompt_smope_mps_full.yaml](/Users/lujiayi/Desktop/summer/SMoPE/configs/cifar-100_prompt_smope_mps_full.yaml)

### 实验相关

- [experiments/cifar-100.sh](/Users/lujiayi/Desktop/summer/SMoPE/experiments/cifar-100.sh)
- [outputs/cifar-100/mps-tiny](/Users/lujiayi/Desktop/summer/SMoPE/outputs/cifar-100/mps-tiny)

---

## 10. 总结

目前项目已经完成了最难的工程阶段：环境、数据、设备兼容、训练闭环和平台迁移。现在三套主数据集都已跑通，后续重点转向结果整理、论文对比和课程报告定稿。整体来看，这次复现已经具备完整交付条件。
