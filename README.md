# LL-YOLO
LL-YOLO: A Lightweight Pavement Disease Detection Algorithm in Low-Light Environments

# Abstract
Intelligent detection of road defects plays a crucial role in urban road maintenance. Accurate and timely identification of cracks and potholes aids in road upkeep and helps extend road lifespans. However, current mainstream road disease detection methods struggle to accurately detect road defects in low-light conditions. In this study, we present Low-Light You Only Look Once (LL-YOLO), a lightweight detection algorithm that effectively identifies pavement defects in low-light environments while maintaining low parameter counts and computational costs. Firstly, the Spatial Pyramid Pooling-Fast Attention Mechanism (SPPFAM) is developed to enhance feature extraction capabilities in both semantic and spatial dimensions. Then, a plug-and-play RepHELAN-GC module, which optimizes parameter count and reduces computational load, is constructed to improve the model's ability to learn multi-scale features. Meanwhile, MPDIoU is integrated to replace the original loss function, thereby enhancing the model's capacity to learn from pavement defect samples under low-light conditions. Finally, the excellent performance of proposed algorithm is demonstrated by using a custom low-light dataset, focusing on mean average precision (mAP50), parameter count, and computational load. Compared to YOLO-based models, the LL-YOLO model achieves an mAP50 of 82.3% on the RDD-LL dataset, outperforming other comparative models. Furthermore, LL-YOLO strikes the best balance between precision and parameter count, demonstrating superior overall performance compared to YOLO variants, RTDETR, and D-FINE algorithm.

# Documentation
## install
Install the package, including all requirements, in a Python>=3.8 environment with PyTorch>=1.8.ultralytics
``` 
pip install ultralytics
```
