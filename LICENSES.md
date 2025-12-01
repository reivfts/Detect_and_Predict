# Licenses and Citations

This project incorporates code, models, and methodologies from various open-source projects and research papers. Below are the complete attributions and license information.

---

## Detection Models

### YOLO11 (Ultralytics)
**License**: AGPL-3.0  
**Source**: https://github.com/ultralytics/ultralytics  
**Citation**:
```bibtex
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```

### Faster R-CNN (torchvision)
**License**: BSD-3-Clause  
**Source**: https://github.com/pytorch/vision  
**Citation**:
```bibtex
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
```bibtex
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

## Backbone and Feature Extraction

### ResNet (torchvision)
**License**: BSD-3-Clause  
**Source**: https://github.com/pytorch/vision  
**Citation**:
```bibtex
@inproceedings{he2016deep,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={770--778},
  year={2016}
}
```

**Usage in this project**:
- ResNet-50: Backbone for Faster R-CNN (with FPN) and DETR
- ResNet-18: Appearance embeddings for object re-identification (optional)

---

## Tracking Components

### Kalman Filter (filterpy)
**License**: MIT  
**Source**: https://github.com/rlabbe/filterpy  
**Documentation**: https://filterpy.readthedocs.io/  
**Citation**:
```bibtex
@misc{labbe2014filterpy,
  author = {Roger R. Labbe Jr.},
  title = {FilterPy: Python library for Kalman filtering and optimal estimation},
  year = {2014},
  publisher = {GitHub},
  url = {https://github.com/rlabbe/filterpy}
}
```

### Hungarian Algorithm (scipy)
**License**: BSD-3-Clause  
**Source**: https://github.com/scipy/scipy  
**Citation**:
```bibtex
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

### CNN-Transformer Hybrid Architecture
**Citation**:
```bibtex
@article{feng2022cnn,
  title={CNN-Transformer Mixed Model for Object Detection},
  author={Feng, Chengjian and Zhong, Yujie and Huang, Weilin},
  journal={arXiv preprint arXiv:2212.06714},
  year={2022}
}
```

**Implementation Notes**: This project implements a three-stage detection pipeline (YOLO → Faster R-CNN → DETR) with fusion, inspired by hybrid CNN-Transformer architectures.

### Two-Level Trajectory Prediction (LSTM Policy Network)
**Citation**:
```bibtex
@inproceedings{xue2018ss,
  title={SS-LSTM: A Hierarchical LSTM Model for Pedestrian Trajectory Prediction},
  author={Xue, Hao and Huynh, Du Q and Reynolds, Mark},
  booktitle={IEEE Winter Conference on Applications of Computer Vision (WACV)},
  pages={1186--1194},
  year={2018}
}
```

**Implementation Notes**: This project implements a custom 2-layer LSTM network for policy anticipation (6 maneuver classes) inspired by hierarchical prediction approaches.

---

## Dataset

### nuScenes Dataset
**License**: CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0)  
**Source**: https://www.nuscenes.org/  
**Citation**:
```bibtex
@inproceedings{caesar2020nuscenes,
  title={nuScenes: A Multimodal Dataset for Autonomous Driving},
  author={Caesar, Holger and Bankiti, Varun and Lang, Alex H. and Vora, Sourabh and Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and Baldan, Giancarlo and Beijbom, Oscar},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={11621--11631},
  year={2020}
}
```

**Usage**: This project uses the nuScenes v1.0-mini dataset (10 scenes) for development and evaluation purposes only. The dataset is used under the non-commercial license.

---

## Deep Learning Frameworks

### PyTorch
**License**: BSD-3-Clause  
**Source**: https://github.com/pytorch/pytorch  
**Citation**:
```bibtex
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
```bibtex
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

---

## License Compatibility

This project is distributed for **academic and research use only**. The primary constraints come from:

1. **YOLO (AGPL-3.0)**: Requires that modifications and derivative works also be open-sourced under AGPL-3.0 if distributed.
2. **nuScenes Dataset (CC BY-NC-SA 4.0)**: Restricts commercial use and requires attribution.

**Project License**: AGPL-3.0 (to comply with the most restrictive component - YOLO)

---

## Original Contributions

The following components are original contributions from this project:

1. **Three-Stage Detection Fusion**: Integration of YOLO, Faster R-CNN, and DETR outputs
2. **Physics-Constrained Kalman Filtering**: Maximum velocity and acceleration enforcement
3. **Hybrid Trajectory Prediction**: Weighted combination of Kalman filter (physics-based) and Transformer (learning-based) predictions
4. **Policy Anticipation LSTM**: Custom 2-layer LSTM for 6-class maneuver prediction
5. **Enhanced Velocity Estimation**: 10-frame EMA with median-based outlier rejection
6. **Uncertainty Quantification Visualization**: Covariance ellipse rendering for prediction confidence
7. **Simple Image Loader**: Direct image folder loading without metadata requirements

These original contributions are released under the **AGPL-3.0 license**.

---

## Disclaimer

This software is provided "as is" without warranty of any kind, express or implied. The authors and contributors are not liable for any damages arising from the use of this software.

For commercial use or alternative licensing arrangements, please contact the respective copyright holders of the constituent components.

---

## Acknowledgments

We acknowledge the authors of all cited works and the open-source community for making their code and models publicly available. This project would not be possible without their contributions to the field of computer vision and autonomous driving.

---

**Last Updated**: December 1, 2025
