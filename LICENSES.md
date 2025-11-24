# Licenses and Citations

This project incorporates code, models, and methodologies from various open-source projects and research papers. Below are the complete attributions and license information.

---

## Core Detection Models

### YOLO (Ultralytics YOLOv11)
**License**: AGPL-3.0  
**Source**: https://github.com/ultralytics/ultralytics  
**Citation**:
```
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750},
  license = {AGPL-3.0}
}
```

### Faster R-CNN (torchvision)
**License**: BSD-3-Clause  
**Source**: https://github.com/pytorch/vision  
**Citation**:
```
@inproceedings{ren2015faster,
  title={Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  booktitle={Advances in Neural Information Processing Systems},
  pages={91--99},
  year={2015}
}
```

### DETR (Detection Transformer)
**License**: Apache-2.0  
**Source**: https://github.com/facebookresearch/detr  
**Citation**:
```
@inproceedings{carion2020end,
  title={End-to-End Object Detection with Transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={European Conference on Computer Vision},
  pages={213--229},
  year={2020},
  organization={Springer}
}
```

---

## Tracking and Prediction

### DeepSORT
**License**: GPL-3.0  
**Source**: https://github.com/nwojke/deep_sort  
**Citation**:
```
@inproceedings{wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  pages={3645--3649},
  year={2017},
  organization={IEEE}
}
```

### Kalman Filter (filterpy)
**License**: MIT  
**Source**: https://github.com/rlabbe/filterpy  
**Citation**:
```
@book{labbe2014kalman,
  title={Kalman and Bayesian Filters in Python},
  author={Labbe, Roger R},
  year={2014},
  publisher={GitHub},
  url={https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python}
}
```

### Hungarian Algorithm (scipy)
**License**: BSD-3-Clause  
**Source**: https://github.com/scipy/scipy  
**Citation**:
```
@article{scipy2020,
  author = {Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E. and others},
  title = {SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python},
  journal = {Nature Methods},
  volume = {17},
  pages = {261--272},
  year = {2020},
  doi = {10.1038/s41592-019-0686-2}
}
```

---

## Research Methodologies

### DONUT (Angle Sampling and Overprediction)
**License**: Not specified (research paper)  
**Source**: https://github.com/MKnoche/DONUT  
**Citation**:
```
@inproceedings{knoche2025donut,
  title={DONUT: Rethinking Dual-stage Methodology for End-to-end Autonomous Driving},
  author={Knoche, Martin and others},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025},
  note={Accepted for publication}
}
```

**Implementation Notes**: This project adapts the angle sampling strategy (±17° hypotheses) and multi-horizon overprediction approach from DONUT for trajectory forecasting.

### CNN-Transformer Hybrid Architecture
**Citation**:
```
@article{feng2022cnn,
  title={CNN-Transformer Mixed Model for Object Detection},
  author={Feng, Chengjian and Zhong, Yujie and Huang, Weilin},
  journal={arXiv preprint arXiv:2212.06714},
  year={2022}
}
```

**Implementation Notes**: This project implements a three-stage detection pipeline (YOLO → Faster R-CNN → DETR) with fusion, inspired by hybrid CNN-Transformer architectures.

### Trajectron++ (Transformer-based Prediction)
**License**: MIT  
**Source**: https://github.com/StanfordASL/Trajectron-plus-plus  
**Citation**:
```
@inproceedings{salzmann2020trajectron++,
  title={Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data},
  author={Salzmann, Tim and Ivanovic, Boris and Chakravarty, Punarjay and Pavone, Marco},
  booktitle={European Conference on Computer Vision},
  pages={683--700},
  year={2020},
  organization={Springer}
}
```

**Implementation Notes**: Transformer-based trajectory prediction with attention mechanisms over historical trajectories.

### AgentFormer (Multi-Agent Forecasting)
**License**: MIT  
**Source**: https://github.com/Khrylx/AgentFormer  
**Citation**:
```
@inproceedings{yuan2021agent,
  title={AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting},
  author={Yuan, Ye and Weng, Xinshuo and Ou, Yanglan and Kitani, Kris},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={9813--9823},
  year={2021}
}
```

### MultiPath (Probabilistic Trajectory Prediction)
**Citation**:
```
@inproceedings{chai2019multipath,
  title={MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction},
  author={Chai, Yuning and Sapp, Benjamin and Bansal, Mayank and Anguelov, Dragomir},
  booktitle={Conference on Robot Learning},
  pages={86--99},
  year={2019}
}
```

---

## Dataset

### nuScenes Dataset
**License**: CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0)  
**Source**: https://www.nuscenes.org/  
**Citation**:
```
@inproceedings{caesar2020nuscenes,
  title={nuScenes: A Multimodal Dataset for Autonomous Driving},
  author={Caesar, Holger and Bankiti, Varun and Lang, Alex H. and Vora, Sourabh and Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and Baldan, Giancarlo and Beijbom, Oscar},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={11621--11631},
  year={2020}
}
```

**Usage**: This project uses the nuScenes v1.0-mini dataset (10 scenes) for development and evaluation purposes only.

---

## Deep Learning Frameworks

### PyTorch
**License**: BSD-3-Clause  
**Source**: https://github.com/pytorch/pytorch  
**Citation**:
```
@inproceedings{paszke2019pytorch,
  title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and others},
  booktitle={Advances in Neural Information Processing Systems},
  pages={8024--8035},
  year={2019}
}
```

### Hugging Face Transformers
**License**: Apache-2.0  
**Source**: https://github.com/huggingface/transformers  
**Citation**:
```
@inproceedings{wolf2020transformers,
  title={Transformers: State-of-the-Art Natural Language Processing},
  author={Wolf, Thomas and Debut, Lysandre and Sanh, Victor and Chaumond, Julien and Delangue, Clement and Moi, Anthony and Cistac, Pierric and Rault, Tim and Louf, Remi and Funtowicz, Morgan and others},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  pages={38--45},
  year={2020}
}
```

---

## Supporting Libraries

### OpenCV
**License**: Apache-2.0  
**Source**: https://github.com/opencv/opencv  

### NumPy
**License**: BSD-3-Clause  
**Source**: https://github.com/numpy/numpy  

### Pandas
**License**: BSD-3-Clause  
**Source**: https://github.com/pandas-dev/pandas  

### Matplotlib
**License**: PSF-based (similar to Python Software Foundation License)  
**Source**: https://github.com/matplotlib/matplotlib  

---

## License Compatibility

This project is distributed for **academic and research use only**. The primary constraints come from:

1. **YOLO (AGPL-3.0)**: Requires that modifications and derivative works also be open-sourced under AGPL-3.0 if distributed.
2. **nuScenes Dataset (CC BY-NC-SA 4.0)**: Restricts commercial use and requires attribution.
3. **DeepSORT (GPL-3.0)**: Requires derivative works to be licensed under GPL-3.0.

**Recommended License for This Project**: AGPL-3.0 (to comply with the most restrictive component - YOLO)

---

## Original Contributions

The following components are original contributions from this project:

1. **Hybrid Prediction Ensemble**: Weighted combination of Kalman filter (physics-based) and Transformer (learning-based) predictions
2. **Distance-Weighted Error Metrics**: Prioritization of nearby objects in evaluation
3. **Adaptive Noise Tuning**: Speed-based Kalman filter process noise adjustment
4. **Enhanced Velocity Estimation**: 10-frame EMA with median-based outlier rejection
5. **Multi-stage Detection Fusion**: Integration of YOLO, Faster R-CNN, and DETR outputs
6. **Physics-Constrained Kalman Filtering**: Maximum velocity and acceleration enforcement
7. **Uncertainty Quantification Visualization**: Covariance ellipse rendering for prediction confidence

These original contributions are released under the **AGPL-3.0 license** to maintain compatibility with the YOLO component.

---

## Disclaimer

This software is provided "as is" without warranty of any kind, express or implied. The authors and contributors are not liable for any damages arising from the use of this software.

For commercial use or alternative licensing arrangements, please contact the respective copyright holders of the constituent components.

---

## Acknowledgments

We acknowledge the authors of all cited works and the open-source community for making their code and models publicly available. This project would not be possible without their contributions to the field of computer vision and autonomous driving.

---

**Last Updated**: November 23, 2025
