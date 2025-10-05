---
title: Brain Tumor MRI Classifier
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# 🧠 Brain Tumor MRI Classification

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.0-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Live_Demo-yellow.svg)](https://huggingface.co/spaces/ABRUBAB/brain-tumor-classifier)

> Advanced Deep Learning System for Automated Brain Tumor Detection and Classification

## 🚀 Try It Live

🔗 **[🚀 Try Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/ABRUBAB/brain-tumor-classifier)**

## 🎯 Overview

A state-of-the-art deep learning system for classifying brain tumors from MRI scans using ensemble learning and explainable AI. The system achieves **96%+ accuracy** through an ensemble of three powerful models: ConvNeXt, Vision Transformer, and EfficientNetV2.

## ✨ Key Features

- 🎯 **High Accuracy**: 96%+ ensemble accuracy, 99%+ individual model F1 scores
- 🔍 **Explainable AI**: Grad-CAM++ visualization for all models (including ViT!)
- 🌐 **Multi-language**: English, Bengali (বাংলা), Spanish (Español)
- ⚡ **Real-time**: < 1 second inference time
- 🎨 **Professional UI**: Full-width dark theme Gradio interface
- 🔄 **Test-Time Augmentation**: Optional accuracy boost
- 🚀 **Live Demo**: Deployed on HuggingFace Spaces

## 🏥 Tumor Types Detected

| Type | Description |
|------|-------------|
| **Glioma** | Tumor originating from glial cells |
| **Meningioma** | Tumor of the meninges |
| **Pituitary** | Tumor in the pituitary gland |
| **No Tumor** | Healthy brain scan |

## 🤖 Model Architecture

### Ensemble of Three State-of-the-Art Models

| Model | Parameters | Val F1 Score | Description |
|-------|-----------|-------------|-------------|
| **ConvNeXt-Base** | 87.6M | 99.9% | Modern CNN architecture |
| **ViT-B/16** | 85.8M | 99%+ | Vision Transformer |
| **EfficientNetV2-M** | 52.9M | 99%+ | Efficient CNN |
| **Ensemble** | - | **96%+** | Weighted average (0.4, 0.35, 0.25) |

## 📊 Performance Metrics

- ✅ **Ensemble Accuracy**: 96%+
- ✅ **Individual Models**: 99%+ F1 scores
- ✅ **Inference Time**: < 1 second per image
- ✅ **Real-time Capable**: Yes
- ✅ **TTA Boost**: ±2% accuracy improvement

## 🔍 Explainable AI

The system provides visual explanations through:

- **Grad-CAM++**: Heatmaps showing decision regions for CNN models
- **Attention Maps**: Visualization for Vision Transformer
- **Confidence Distribution**: Probability breakdown for all classes

## 🛠️ Technology Stack

**Deep Learning**
- PyTorch 2.5
- timm (PyTorch Image Models)
- Transfer learning from ImageNet-21k

**Computer Vision**
- OpenCV
- Albumentations (data augmentation)
- Pillow

**UI & Deployment**
- Gradio 5.0 (professional dark theme)
- HuggingFace Spaces

**Explainability**
- Grad-CAM++
- Attention visualization

## 📁 Project Structure

```
brain-tumor-mri-classification/
├── notebooks/                        # 6 Kaggle notebooks (complete pipeline)
│   ├── notebook-1-setup.ipynb
│   ├── notebook-2-eda-preprocessing.ipynb
│   ├── notebook-3-training.ipynb
│   ├── notebook-4-ensemble-explainability.ipynb
│   ├── notebook-5-github-deployment.ipynb
│   └── notebook-6-final-testing.ipynb
├── models/best/                      # Trained model checkpoints
│   ├── best_convnext_model.pth
│   ├── best_vit_model.pth
│   └── best_efficientnetv2_model.pth
├── app.py                            # HuggingFace Spaces application
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### Using the Live Demo

1. Visit: **[https://huggingface.co/spaces/ABRUBAB/brain-tumor-classifier](https://huggingface.co/spaces/ABRUBAB/brain-tumor-classifier)**
2. Upload a brain MRI scan image
3. Select language (English, Bengali, or Spanish)
4. Choose AI model (Ensemble recommended)
5. Click "ANALYZE SCAN"
6. View medical report and visualizations

### Local Installation

```bash
# Clone repository
git clone https://github.com/ABRUBAB/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification

# Install dependencies
pip install -r requirements.txt

# Run application (requires model checkpoints)
python app.py
```

## 📝 Training Details

**Data Preprocessing**
- Image size: 224×224
- Normalization: ImageNet statistics
- Augmentation: Horizontal/vertical flip, rotation, brightness, contrast, gaussian noise

**Training Configuration**
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-2)
- Scheduler: CosineAnnealingWarmRestarts
- Loss: CrossEntropyLoss
- Early stopping with patience=5
- Mixed precision training

**Hardware**
- GPU: NVIDIA Tesla P100 (Kaggle)
- Training time: ~2-3 hours per model

## 📚 Complete Project Pipeline

This project includes 6 comprehensive Kaggle notebooks:

1. **Setup & Data Collection** - Environment setup, dataset download (~7,000 images)
2. **EDA & Preprocessing** - Exploratory analysis, augmentation pipeline
3. **Model Training** - Train ConvNeXt, ViT, EfficientNetV2 (99%+ F1 each)
4. **Ensemble & Explainability** - Create ensemble, implement Grad-CAM++, build professional UI
5. **GitHub Integration** - Automated deployment files creation
6. **Final Testing** - Live deployment testing, completion verification

🔗 **GitHub Repository**: [brain-tumor-mri-classification](https://github.com/ABRUBAB/brain-tumor-mri-classification)

## 🌐 Multi-language Support

The system provides comprehensive medical reports in:
- 🇺🇸 **English** - Full medical interpretation
- 🇧🇩 **বাংলা (Bengali)** - সম্পূর্ণ বাংলা অনুবাদ
- 🇪🇸 **Español (Spanish)** - Traducción completa al español

## 🎨 User Interface Features

- **Professional Dark Theme**: Full-width responsive design
- **Grad-CAM++ Visualization**: Works for ALL models (ConvNeXt, ViT, EfficientNetV2, Ensemble)
- **Confidence Charts**: Real-time probability distribution
- **Medical Reports**: Comprehensive analysis with recommendations
- **Copy-to-Clipboard**: Easy sharing of results

## ⚠️ Medical Disclaimer

**IMPORTANT**: This AI system is developed for **educational and research purposes only**. It should **NOT** be used as a substitute for professional medical diagnosis. 

- Always seek advice from qualified healthcare providers
- Never disregard professional medical advice based on AI predictions
- This system is not FDA/EMA approved
- Clinical decisions should only be made by certified medical practitioners

## 👨‍💻 Author

**Abdullah Rubab**

- 🏫 Daffodil International University, Bangladesh
- 🎓 Computer Science & Engineering
- 📧 Email: rubab2712@gmail.com | rubab2305101813@diu.edu.bd
- 🐙 GitHub: [@ABRUBAB](https://github.com/ABRUBAB)
- 🔗 LinkedIn: [Abdullah Rubab](https://www.linkedin.com/in/abdullah-rubab-1a947a200/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Brain Tumor MRI Dataset (Kaggle) - 7,023 images
- **Pretrained Models**: timm library (HuggingFace)
- **Framework**: PyTorch Team
- **UI Framework**: Gradio Team
- **Deployment**: HuggingFace Spaces
- **Experiment Tracking**: Weights & Biases

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{rubab2025braintumor,
  author = {Rubab, Abdullah},
  title = {Brain Tumor MRI Classification using Deep Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ABRUBAB/brain-tumor-mri-classification}
}
```

## 📈 Project Statistics

- **Total Development Time**: Complete 6-notebook pipeline
- **Total Lines of Code**: ~2,500+
- **Models Trained**: 3 state-of-the-art architectures
- **Ensemble Accuracy**: 96%+
- **Individual Model F1**: 99%+
- **Languages Supported**: 3
- **Total Parameters**: ~226M combined

## 🔗 Related Links

- [🚀 Live Demo](https://huggingface.co/spaces/ABRUBAB/brain-tumor-classifier)
- [🐙 GitHub Repository](https://github.com/ABRUBAB/brain-tumor-mri-classification)
- [📓 Kaggle Notebooks](https://www.kaggle.com/code/YOURUSERNAME)

---

**⭐ If you found this project helpful, please consider giving it a star on GitHub!**

**🚀 Try the live demo and explore cutting-edge medical AI!**

© 2025 Abdullah Rubab. All rights reserved.
