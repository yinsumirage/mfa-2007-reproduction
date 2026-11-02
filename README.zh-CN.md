# MFA 2007：CPU 算法复现

[English README](README.md)

本仓库根据 Xu 等人 2007 年的 [MFA 论文](https://doi.org/10.1109/TIP.2007.906769)重写 Python 实现，包含 MFA-1、MFA-2、Matrix-based MFA、MBA/KMBA、BDA/KBDA 以及步态周期平均和序列匹配。使用 CPU，无需 GPU。这是 AI 辅助开发、经过数值检查的独立教学实现。

**已经完成核心算法实现与替代数据实验；没有完整复现原文 USF/Corel22K 的结果表。**

## 运行检查

使用 Python 3.12，从仓库根目录运行：

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s code -p 'test_*.py' -v
python code/check_paper_contract.py
```

Windows 建环境用 `py -3.12 -m venv .venv`，激活 `.venv\Scripts\Activate.ps1` 后运行相同的 pip 和检查命令。28 项检查与定义核查不需要下载数据。

## 复跑实验

```sh
python code/run_experiment.py
python code/prepare_data.py
python code/run_all.py
python code/visual_comparison.py
python demo/build_demo.py
```

第一个是小型 Digits 实验。完整分类、检索实验需要 `prepare_data.py` 联网下载和核验 Olivetti、固定 commit 的 Corel1K 镜像。重跑会更新结果，建议另用一份检出目录保留当前快照。仓库不附带原始图片、课程资料或论文 PDF。

直接用浏览器打开 [demo/index.html](demo/index.html) 可以切换四种方法、点开图片和最近邻；演示数据已经内嵌，无需联网。

## 固定的原文定义

- k1：每个训练样本选多少个同类近邻。
- k2：每个类别对其余所有类别合并选取的最近跨类样本对数，最终作无向并集。
- 分类固定 k1=3、k2=50，seed 为16/17/18，同用1-NN。
- MFA-1、MFA-2 是原文的两种求解方式：GED 近似与直接 Trace Ratio 迭代，共用同一套图规则。
- MMFA 在本仓库指 Matrix-based MFA。shared-PCA 是单独的预处理诊断。

详细设置见[原文定义](docs/PAPER_DEFINITIONS.md)、[实现约定](code/implementation-notes.md)及[实验报告](docs/EXPERIMENTS.md)。第三方参考只作可选工程对照，不替代本仓库的原文定义。

数值文件与当前已检查的核心实现一致，公开导出只调整图像输出位置和不依赖本地 PDF 的核查入口。来源、AI 辅助开发和导出范围见[记录](docs/PROVENANCE.md)。
