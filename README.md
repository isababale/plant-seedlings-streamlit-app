# 🌱 SeedScout - AI-Powered Plant Seedling Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Identify weed and crop seedlings instantly using deep learning**

SeedScout is an AI-powered web application that helps farmers, agronomists, and students identify plant seedlings at early growth stages. Built with ResNet18 and trained on the Aarhus University Plant Seedlings Dataset, it achieves **94% overall accuracy** across 12 common plant species.

![SeedScout Demo](assets/demo.gif)
*Demo: Upload an image and get instant predictions with confidence scores*

---

## 🎯 Features

- **🚀 Batch Processing**: Upload and classify multiple images simultaneously
- **📊 Detailed Analytics**: View confidence scores, top-3 predictions, and accuracy metrics
- **🔍 Grad-CAM Visualization**: See which parts of the image the model focuses on
- **💾 CSV Export**: Download all predictions for record-keeping
- **⚠️ Smart Warnings**: Species-specific alerts for lower-accuracy predictions
- **📱 Responsive Design**: Works on desktop and mobile devices
- **🌐 Web-Based**: No installation required - runs in your browser

---

## 📸 Screenshots

### Main Interface
![Main Interface](assets/screenshot_main.png)

### Batch Processing Results
![Batch Results](assets/screenshot_batch.png)

### Grad-CAM Visualization
![Grad-CAM](assets/screenshot_gradcam.png)

---

## 🌾 Supported Species

SeedScout can identify **12 plant species** at seedling stage:

| Species | Model Accuracy | Notes |
|---------|---------------|-------|
| **Charlock** | 99% | ✅ Excellent |
| **Cleavers** | 99% | ✅ Excellent |
| **Common Chickweed** | 98% | ✅ Excellent |
| **Small-flowered Cranesbill** | 100% | ✅ Perfect |
| **Scentless Mayweed** | 99% | ✅ Excellent |
| **Fat Hen** | 97% | ✅ Excellent |
| **Shepherd's Purse** | 98% | ✅ Excellent |
| **Common Wheat** | 98% | ✅ Excellent |
| **Maize** | 99% | ✅ Excellent |
| **Sugar Beet** | 99% | ✅ Excellent |
| **Loose Silky-bent** | 88% | ✅ Good |
| **Black-grass** | 67% | ⚠️ Lower - verify independently |

### ⚠️ Important Limitations
- Only works for seedlings (1-4 weeks old)
- Requires clear, single-plant images
- Best results with natural lighting
- Plant must be one of the 12 supported species

---

## 🚀 Quick Start

### Option 1: Try Online (Recommended)
🌐 **[Launch SeedScout](https://your-app-url.streamlit.app)** *(Coming soon)*

### Option 2: Run Locally

#### Prerequisites
- Python 3.10 or higher
- pip package manager

#### Installation

1. **Clone the repository**
```bash
git clone https://github.com/isababale/plant-seedlings-streamlit-app.git
cd plant-seedlings-streamlit-app
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the model**
- Download `best_resnet18_plant.pth` from [Releases](https://github.com/yourusername/seedscout/releases)
- Place it in the project root directory

4. **Run the app**
```bash
streamlit run app.py
```

5. **Open your browser**
- Navigate to `http://localhost:8501`
- Start classifying seedlings! 🌱

---

## 📦 Project Structure

```
seedscout/
├── app.py                          # Main Streamlit application
├── best_resnet18_plant.pth         # Trained ResNet18 model (not in repo)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── LICENSE                         # MIT License
├── assets/                         # Images and demos
│   ├── demo.gif
│   ├── screenshot_main.png
│   └── ...
├── examples/                       # Example seedling images
│   ├── charlock_example.jpg
│   ├── cleavers_example.jpg
│   └── ...
└── notebooks/                      # Training notebooks (optional)
    └── train_model.ipynb
```

---

## 💻 Usage

### Basic Workflow

1. **Upload Images**
   - Click "Choose images..." button
   - Select one or more seedling photos (JPG, PNG)
   - Images should show single plants at seedling stage

2. **Adjust Settings**
   - Set confidence threshold (default: 60%)
   - Enable/disable Grad-CAM visualization

3. **View Results**
   - See predictions with confidence scores
   - Check model accuracy for predicted species
   - Review warnings for low-confidence or challenging species

4. **Download Results**
   - Export predictions as CSV
   - Save for record-keeping or further analysis

### Best Practices for Images

✅ **Good:**
- Clear, focused photos
- Good natural lighting
- Single plant centered in frame
- Seedling stage (1-4 weeks old)
- Plain background (soil or white)

❌ **Avoid:**
- Blurry or dark images
- Multiple overlapping plants
- Mature/flowering plants
- Heavy shadows or glare
- Plants not in supported list

---

## 🧠 Model Details

### Architecture
- **Base Model**: ResNet18 (Deep Residual Network)
- **Layers**: 18 layers with residual connections
- **Input Size**: 224×224 pixels
- **Output**: 12-class softmax

### Training
- **Dataset**: [Aarhus University Plant Seedlings Dataset](https://vision.eng.au.dk/plant-seedlings-dataset/)
- **Images**: ~5,500 seedling images
- **Preprocessing**: ImageNet normalization
- **Augmentation**: Random flips, rotations, color jittering
- **Framework**: PyTorch 2.0+

### Performance
- **Overall Accuracy**: 94%
- **Training Time**: ~2 hours on CPU
- **Inference Time**: ~50ms per image (CPU)

---

## 📊 Technical Specifications

### Dependencies
- Python 3.10+
- PyTorch 2.0+
- torchvision 0.15+
- Streamlit 1.31+
- Pillow 10.0+
- NumPy 1.24+
- Pandas 2.0+
- OpenCV 4.8+

### Hardware Requirements
**Minimum:**
- 2GB RAM
- CPU (Intel/AMD x64)
- 500MB disk space

**Recommended:**
- 4GB RAM
- CUDA-capable GPU (optional, for faster inference)
- 1GB disk space

---

## 🔬 How It Works

### 1. Image Preprocessing
```python
Image → Resize(224×224) → Normalize(ImageNet stats) → Tensor
```

### 2. Model Inference
```python
ResNet18 → Feature Extraction → Classification → Softmax → Top-3 Predictions
```

### 3. Grad-CAM (Optional)
```python
Forward Pass → Backward Pass → Gradient Weights → Heatmap Overlay
```

### 4. Result Interpretation
- **>90% confidence**: High reliability
- **75-90% confidence**: Good, verify if critical
- **60-75% confidence**: Moderate, double-check recommended
- **<60% confidence**: Low, verification required

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🧪 Add test cases
- 🌱 Contribute training data for new species
- 🎨 Enhance UI/UX

### Development Setup

1. Fork the repository
2. Create a feature branch
```bash
git checkout -b feature/amazing-feature
```

3. Make your changes
4. Commit with clear messages
```bash
git commit -m "Add amazing feature"
```

5. Push to your fork
```bash
git push origin feature/amazing-feature
```

6. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Write clear commit messages
- Test locally before submitting

---

## 📈 Roadmap

### Version 2.0 (Coming Soon)
- [ ] Mobile app (iOS/Android)
- [ ] REST API for integration
- [ ] Support for 50+ species
- [ ] Multi-language interface
- [ ] Offline mode

### Future Ideas
- [ ] Growth stage detection
- [ ] Severity assessment
- [ ] Treatment recommendations
- [ ] Integration with farm management systems
- [ ] Drone/robot compatibility

---

## 🙏 Acknowledgments

- **Dataset**: [Aarhus University Plant Seedlings Dataset](https://vision.eng.au.dk/plant-seedlings-dataset/)
- **Paper**: Giselsson et al. (2017) - "A Public Image Database for Benchmark of Plant Seedling Classification Algorithms"
- **ResNet**: He et al. (2015) - "Deep Residual Learning for Image Recognition"
- **Framework**: PyTorch Team
- **UI**: Streamlit Team

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses
- Aarhus Plant Seedlings Dataset: Available under open access
- PyTorch: BSD License
- Streamlit: Apache 2.0 License

---

## 📧 Contact

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

**Project Link**: [https://github.com/yourusername/seedscout](https://github.com/yourusername/seedscout)

---

## ⚖️ Disclaimer

SeedScout is an educational and research tool. It should not be the sole basis for agricultural decisions. Always verify identifications with agricultural experts, especially for critical weed management decisions. The developers assume no liability for any damages or losses resulting from the use of this software.

---

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/seedscout&type=Date)](https://star-history.com/#yourusername/seedscout&Date)

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{seedscout2024,
  author = {Your Name},
  title = {SeedScout: AI-Powered Plant Seedling Classifier},
  year = {2024},
  url = {https://github.com/yourusername/seedscout}
}
```

And the original dataset:

```bibtex
@article{giselsson2017public,
  title={A public image database for benchmark of plant seedling classification algorithms},
  author={Giselsson, Thomas Mosgaard and others},
  journal={arXiv preprint arXiv:1711.05458},
  year={2017}
}
```

---

<div align="center">

**Made with ❤️ for farmers, students, and plant enthusiasts**

[⬆ Back to Top](#-seedscout---ai-powered-plant-seedling-classifier)

</div>
