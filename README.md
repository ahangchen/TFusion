# TFusion
CVPR2018: Unsupervised Cross-dataset Person Re-identification by Transfer Learning of Spatio-temporal Patterns

![TFusion架构](https://upload-images.jianshu.io/upload_images/1828517-e12da67722080fdf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- We present a novel method to learn  pedestrians' spatio-temporal patterns in  unlabeled target datsets by transferring the visual classifier from the source dataset. The algorithm does not require any prior knowledge about the spatial distribution of cameras nor any assumption about how people move in the target environment.

- We propose a Bayesian fusion model, which  combines the  spatio-temporal patterns learned and the visual features to achieve  high performance of person Re-ID in the unlabeled target datasets.

- We propose a learning-to-rank based  mutual promotion procedure, which uses the fusion classifier to teach the weaker visual classifier by the ranking results on unlabeled dataset. This mutual learning mechanism can be applied to many domain adaptation problems.


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
