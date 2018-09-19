## Rank Re-identification

### 原理
- 三张图两两之间的相似度大小决定了rank时备选图片的顺序
- 可以根据相似度计算两对图的排序概率，两图相似度越高，排序概率越大
- 通过回归两对图间的排序概率，对图像分类器做更好的训练

### 模型

![](img/rank_model.png)

- 基础网络：ResNet50
- 输入：根据基础网络单模型计算相似度得到一个rank表，按表中的相似度选择一张待匹配图片A，备选图片B和C
- 输出：使用ranknet公式计算AB排序高于AC的概率，回归根据rank表计算得到的实际排序概率

### 硬件
- TITANX单卡（通过CUDA_VISIBLE_DEVICES指定使用哪块GPU）

### 代码说明
- baseline：ResNet52基础模型
  - [evaluate.py](baseline/evaluate.py)
    - extract_feature: 在测试集上计算特征
    - similarity_matrix: 根据特征计算特征之间的余弦相似度（使用GPU矩阵运算进行加速）
    - 在测试集上，调用test_predict计算rank表
    - 在训练集上，调用train_predict计算rank表
    - 使用map_rank_eval在Market1501上计算rank acc和map
    - 使用grid_result_eval在GRID上计算rank acc
  - [train.py](baseline/train.py)
    - 使用源训练集数据预训练ResNet50基础网络
- pair: 双图二分类模型预训练
  - [pair_train.py](pair/pair_train.py)：双图预训练
    - pair_generator: 数据生成器，根据标签选择正样本和负样本
    - pair_model: 搭建二分类双输入模型
  - [eval](pretrian/eval.py)：各种模型的测试都在这里
    - 加载对应的模型
    - 调用baseline/evaluate.py中与数据集对应的函数进行测试

- transfer: 根据rank表选择样本对模型进行增量训练
  - [simple_rank_transfer.py](transfer/pair_transfer.py): 三图ranknet增量训练
    - triplet_generator_by_rank_list：与二分类相同的方法，选择三张（两对）图，同时计算排序概率作为回归标签
    - rank_transfer_model：三图输入，一个排序概率loss


#### Reference
- [RankNet算法原理和实现](http://x-algo.cn/index.php/2016/07/31/ranknet-algorithm-principle-and-realization/)