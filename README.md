# TFusion
CVPR2018: Unsupervised Cross-dataset Person Re-identification by Transfer Learning of Spatio-temporal Patterns

![TFusion架构](https://upload-images.jianshu.io/upload_images/1828517-e12da67722080fdf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- We present a novel method to learn  pedestrians' spatio-temporal patterns in  unlabeled target datsets by transferring the visual classifier from the source dataset. The algorithm does not require any prior knowledge about the spatial distribution of cameras nor any assumption about how people move in the target environment.

- We propose a Bayesian fusion model, which  combines the  spatio-temporal patterns learned and the visual features to achieve  high performance of person Re-ID in the unlabeled target datasets.

- We propose a learning-to-rank based  mutual promotion procedure, which uses the fusion classifier to teach the weaker visual classifier by the ranking results on unlabeled dataset. This mutual learning mechanism can be applied to many domain adaptation problems.

**This code is ONLY released for academic use.**

## How to use
We split TFusion into two components:

- rank-reid
  - Framework: Keras and Tensorflow
  - Training Resnet based Siamese network on source dataset
  - Learning to rank on target dataset
- TrackViz
  - Dependencies: Some traditional libraries, including numpy, pickle, matplotlib, seaborn
  - Building spatial temporal model with visual classification results
  - Bayesian Fusion

Components communicate by ranking results. We use this results for visualization and logical analysis in our experiments, thus we save them on file system in TrackViz/data. 

Written and tested in python2.

### Dataset
#### Download
 - [CUHK01](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
 - [VIPeR](https://vision.soe.ucsc.edu/node/178)
 - [Market-1501](http://www.liangzheng.org/Project/project_reid.html)
 - [GRID](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_qmul_underground_reid.html)


#### Pre-process
- CUHK01

we only use CUHK01 as source dataset, so we use all images for pretrain, place all images in a directory.

- VIPeR

the same as CUHK01.

- GRID as Source dataset

we use all labeled images in GRID for pretraining as source dataset, so place all labeled images in a directory, for example "grid_label"

- Market-1501 
  - download
  - rename training directory to 'train', rename probe directory to 'probe', renmae gallery directory to 'test'

- GRID as Target Dataset
  - follow dataset instruction, split the dataset to ten cross-validation sets
  - in each cross-validation set, rename training directory to 'train', rename probe directory to 'probe', renmae gallery directory to 'test'
  - you can also refer to 'TrackViz/data/grid' for more details about GRID cross validation.

Finally, your data will look like this:


```bash
Market-1501
├── probe
│   ├── 0003_c1s6_015971_00.jpg
│   ├── 0003_c3s3_064744_00.jpg
│   ├── 0003_c4s6_015641_00.jpg
│   ├── 0003_c5s3_065187_00.jpg
│   └── 0003_c6s3_088392_00.jpg
├── test
│   ├── 0003_c1s6_015971_02.jpg
│   ├── 0003_c1s6_015996_02.jpg
│   ├── 0003_c4s6_015716_03.jpg
│   ├── 0003_c5s3_065187_01.jpg
│   ├── 0003_c6s3_088392_04.jpg
│   └── 0003_c6s3_088442_04.jpg
└── train
    ├── 0002_c1s1_000451_03.jpg
    ├── 0002_c1s1_000551_01.jpg
    ├── 0002_c1s1_000776_01.jpg
    ├── 0002_c1s1_000801_01.jpg
    ├── 0002_c1s1_069056_02.jpg
    └── 0002_c6s1_073451_02.jpg

```

```bash
grid_train_probe_gallery
├── cross0
│   ├── probe
│   │   ├── 0002_1_25008_169_19_94_224.jpeg
│   │   ├── 0003_1_25008_57_44_97_265.jpeg
│   │   ├── 0004_1_25072_204_72_106_277.jpeg
│   │   └── 0005_1_25120_210_22_84_215.jpeg
│   ├── test
│   │   ├── 0000_1_25698_101_16_87_246.jpeg
│   │   ├── 0000_1_26113_116_13_72_212.jpeg
│   │   ├── 0000_1_26207_113_25_69_172.jpeg
│   │   └── gallery.txt
│   └── train
│       ├── 0001_1_25004_107_32_106_221.jpeg
│       ├── 0001_2_25023_116_134_128_330.jpeg
│       ├── 0009_1_25208_126_19_71_215.jpeg
│       ├── 0009_2_25226_176_72_87_246.jpeg
│       └── 0248_5_33193_101_100_90_308.jpeg
├── cross1
├── cross2
├── cross3
├── cross4
├── cross5
├── cross6
├── cross7
├── cross8
└── cross9

```

Place all datasets in the same directory, like this:

```bash
dataset
├── cuhk01
├── grid_train_probe_gallery
├── Market-1501
└── source
```

#### Configuration
- Pretrain Config: Modify all path containing '/home/cwh' appearing in rank-reid/pretrain/pair_train.py  to your corresponding path.
- Fusion Config 
  - Modify all path containing '/home/cwh' appearing in TrackViz/ctrl/transfer.py  to your corresponding path.
  - Modify all path containing '/home/cwh' appearing in rank-reid/rank-reid.py  to your corresponding path.

### Pretrain
Pretrain Resnet52 and Siamese Network using source datasets.

```bash
cd rank-reid/pretrain && python pair_train.py
```

This code will save pretrained model in pair-train directory:

```bash
pretrain
├── cuhk_pair_pretrain.h5
├── cuhk_softmax_pretrain.h5
├── eval.py
├── grid-cv-0_pair_pretrain.h5
├── grid-cv-0_softmax_pretrain.h5
├── grid-cv-1_pair_pretrain.h5
├── grid-cv-1_softmax_pretrain.h5
├── grid-cv-2_pair_pretrain.h5
├── grid-cv-2_softmax_pretrain.h5
├── grid-cv-3_pair_pretrain.h5
├── grid-cv-3_softmax_pretrain.h5
├── grid-cv-4_pair_pretrain.h5
├── grid-cv-4_softmax_pretrain.h5
├── grid-cv-5_pair_pretrain.h5
├── grid-cv-5_softmax_pretrain.h5
├── grid-cv-6_pair_pretrain.h5
├── grid-cv-6_softmax_pretrain.h5
├── grid-cv-7_pair_pretrain.h5
├── grid-cv-7_softmax_pretrain.h5
├── grid-cv-8_pair_pretrain.h5
├── grid-cv-8_softmax_pretrain.h5
├── grid-cv-9_pair_pretrain.h5
├── grid-cv-9_softmax_pretrain.h5
├── grid_pair_pretrain.h5
├── grid_softmax_pretrain.h5
├── __init__.py
├── market_pair_pretrain.h5
├── market_softmax_pretrain.h5
├── pair_train.py
├── pair_transfer.py
├── source_pair_pretrain.h5
└── source_softmax_pretrain.h5

```

## TFusion
include directly vision transfering, fusion, learning to rank

```bash
cd TrackViz && python ctrl/transfer.py
```

Results will be saved in TrackViz/data

```bash
TrackViz/data
├── source_target-r-test # transfer after learning to rank on test set
│   ├── cross_filter_pid.log
│   ├── cross_filter_score.log
│   ├── renew_ac.log
│   ├── renew_pid.log
│   └── sorted_deltas.pickle
├── source_target-r-train # transfer after learning to rank on training set
│   ├── cross_filter_pid.log
│   ├── cross_filter_score.log
│   ├── cross_mid_score.log
│   ├── renew_ac.log
│   ├── renew_pid.log
│   └── sorted_deltas.pickle
├── source_target-r-train_diff # ST model built by random classifier minus visual classfier after learning to rank
│   ├── renew_pid.log
│   └── sorted_deltas.pickle
├── source_target-r-train_rand  # ST model built by random classifier after learning to rank
│   ├── renew_pid.log
│   └── sorted_deltas.pickle
├── source_target-test # directly transfer from source to target test set
│   ├── cross_filter_pid_32.log
│   ├── cross_filter_pid.log
│   ├── cross_filter_score.log
│   ├── renew_ac.log
│   ├── renew_pid.log
│   └── sorted_deltas.pickle
├── source_target-train # directly transfer from source to  target training set
│   ├── cross_filter_pid.log # sorted pids by fusion scores
│   ├── cross_filter_score.log # sorted fusion scores corresponding to pids
│   ├── cross_mid_score.log # can be use to generate pseudo lable, ignore it 
│   ├── renew_ac.log #  sorted vision scores corresponding to pids
│   ├── renew_pid.log # sorted pids by vision scores
│   └── sorted_deltas.pickle # store time deltas, so called ST model built by visual classifier
├── source_target-train_diff # store time deltas, ST model built by random classifier minus visual classifier
│   ├── renew_pid.log
│   └── sorted_deltas.pickle
└── source_target-train_rand # store time deltas, built by random visual classifier
    ├── renew_pid.log
    └── sorted_deltas.pickle
```

### Evaluation
Evaluation result will be automatically saved in the log_path, as you specified in rank-reid/rank-reid.py predict_eval(), default location is TrackViz/market_result_eval.log, TrackViz/grid_eval.log  

- GRID evaluation includes rank1, rank5, rank-10 accuracy
- Market-1501 evaluation includes rank1 accuracy and mAP. Rank5 and rank10 should be computed by code in [MATLAB](http://pan.baidu.com/s/1hqMbd4K) provided by Liang Zheng.

## Citation

Please cite this paper in your publications if it helps your research:

```bib
@article{
  title={Unsupervised Cross-dataset Person Re-identification by Transfer Learning of Spatial-Temporal Patterns},
  author={Jianming, Lv and Weihang, Chen and Qing, Li and Can, Yang},
  journal={CVPR},
  year={2018}
}
```    
