"""Titanic 生存预测项目。

模块划分::

    config        —— 路径、特征集合、交叉验证协议与超参数
    data          —— 原始数据读取与特征工程
    preprocessing —— 折内（fold-wise）缺失值填充、编码与标准化
    model         —— FT-Transformer 与 EMA
    train         —— FT-Transformer 的重复交叉验证训练
    baselines     —— 梯度提升 / 线性等互补模型
    ensemble      —— OOF 加权融合与阈值选择
    main          —— 一键训练 + 推理 + 生成报告
"""

__version__ = "2.0.0"
