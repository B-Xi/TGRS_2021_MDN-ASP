Multi-direction Networks with Attentional Spectral Prior for Hyperspectral Image Classification, TGRS, 2021.
==
[Bobo Xi](https://scholar.google.com/citations?user=O4O-s4AAAAAJ&hl=zh-CN), [Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), [Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html), [Rui song](https://scholar.google.com/citations?user=_SKooBYAAAAJ&hl=zh-CN), Yuchao Xiao, [Yanzi Shi](https://www.researchgate.net/scientific-contributions/Yanzi-Shi-2149921066) and [Qian Du](https://scholar.google.com/citations?user=0OdKQoQAAAAJ&hl=zh-CN).

***
Code for paper: Multi-direction Networks with Attentional Spectral Prior for Hyperspectral Image Classification. (The complete project will be soon released after the paper is open access)

<div align=center><img src="/Image/framework.jpg" width="80%" height="80%"></div>
Fig. 1: The framework of our proposed MDN-ASP for HSI classification. It is composed of four components: multi-direction samples construction, multi-stream feature extraction, feature aggregation with attentional spectral prior (ASP) and a softmax-based classifier. The same color represents the layers with same operation.

Training and Test Process
--
Please simply run the 'run_demo.m' to reproduce our MSC-EDKELM results on [IndianPines](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Indian_Pines) dataset. You can obtain the classification accuracies (15 training samples per class) and the corresponding classification maps shown below. We have successfully test it on Ubuntu 16.04 and Windows systems with Matlab R2017b.

<div align=center><p float="center">
<img src="/Image/falsecolorimage.jpg" width="200"/>
<img src="/Image/IndianP_gt.jpg" width="200"/>
<img src="/Image/trainingMap.jpg" width="200"/>
<img src="/Image/classification_map.jpg" width="200"/>
</p></div>
<div align=center>Fig. 2: The composite false-color image, groundtruth, training samples, and classification map of Indian Pines dataset.</div>
      
References
--
If you find this code helpful, please kindly cite:

[1] B. Xi, J. Li, Y. Li, R. Song, Y. Xiao, Y. Shi, Q. Du "Multi-direction Networks with Attentional Spectral Prior for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, 2021, doi: 10.1109/TGRS.2020.3047682
[2] B. Xi, J. Li, Y. Li, R. Song, Y. Shi, S. Liu, Q. Du "Deep Prototypical Networks With Hybrid Residual Attention for Hyperspectral Image Classification," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 13, pp. 3683-3700, 2020, doi: [10.1109/JSTARS.2020.3004973](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9126161).  

Citation Details
--
BibTeX entry:
```
@ARTICLE{Xi_TGRS2021_MDNASP,
  author={B. {Xi} and J. {Li} and Y. {Li} and R. {Song} and Y. {Xiao} and Y. {Shi} and Q. {Du}},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Multi-direction Networks with Attentional Spectral Prior for Hyperspectral Image Classification}, 
  year={2021},
  volume={},
  number={},
  pages={},}
```
```
@ARTICLE{Xi2020JSTARS,
  author={B. {Xi} and J. {Li} and Y. {Li} and R. {Song} and Y. {Shi} and S. {Liu} and Q. {Du}},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Deep Prototypical Networks With Hybrid Residual Attention for Hyperspectral Image Classification}, 
  year={2020},
  volume={13},
  number={},
  pages={3683-3700},}
 ```

Licensing
--
Copyright (C) 2020 Bobo Xi and Jiaojiao Li

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
