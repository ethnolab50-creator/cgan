# CGAN Web Application 🤖

A complete web-based Conditional Generative Adversarial Network (CGAN) for generating handwritten digits using the MNIST dataset.

![CGAN Demo](https://via.placeholder.com/800x400/4CAF50/white?text=CGAN+Web+App+Demo)

## 🚀 Features

- **Interactive Web Interface**: Train and generate digits through a modern web UI
- **Conditional GAN**: Generate specific digits (0-9) conditioned on class labels
- **Real-time Training**: Monitor training progress with live updates
- **PyTorch Backend**: High-performance deep learning with PyTorch
- **MNIST Dataset**: Uses the famous MNIST handwritten digit dataset
- **REST API**: Full REST API for integration with other applications
- **Responsive Design**: Works on desktop and mobile devices
- **No External Dependencies**: Runs completely offline

## 🎯 Live Demo

**Try it now:** [https://ethnolab50-creator.github.io/cgan](https://ethnolab50-creator.github.io/cgan)

## 📋 Prerequisites

- Python 3.8+
- PyTorch
- Flask
- Modern web browser

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ethnolab50-creator/cgan.git
   cd cgan
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open in browser:**
   ```
   http://localhost:5000
   ```

## 🎮 Usage

### Training the Model
1. Set the number of training epochs (1-50)
2. Click "▶️ Train Model"
3. Watch the progress in the modal dialog
4. Model is automatically saved

### Generating Digits
1. Select a digit (0-9) from the dropdown
2. Choose number of samples (1-10)
3. Click "✨ Generate" for specific digit
4. Click "📊 Generate All Digits" for all digits

## 🏗️ Architecture

### Generator Network
- Input: Noise vector (100-dim) + Class label (10-dim)
- Architecture: 4-layer MLP with BatchNorm and ReLU
- Output: 28×28 grayscale image

### Discriminator Network
- Input: 28×28 image + Class label (10-dim)
- Architecture: 4-layer MLP with LeakyReLU
- Output: Real/fake probability

### Training
- Optimizer: Adam (lr=0.0002, β1=0.5, β2=0.999)
- Loss: Binary Cross-Entropy
- Batch size: 64
- Device: CPU/GPU automatic detection

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/api/status` | GET | Get training status |
| `/api/train` | POST | Train the CGAN model |
| `/api/generate` | POST | Generate specific digit |
| `/api/generate_all` | GET | Generate all digits (0-9) |
| `/api/load_model` | POST | Load saved model |

## 📁 Project Structure

```
cgan_web_app/
├── app.py                 # Flask backend with CGAN model
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main web interface
└── static/
    ├── css/
    │   └── style.css     # Responsive styling
    └── js/
        └── app.js        # Frontend JavaScript
```

## 🔧 Technologies Used

- **Backend:** Python, Flask, PyTorch
- **Frontend:** HTML5, CSS3, JavaScript (ES6+)
- **AI/ML:** Conditional GAN, MNIST dataset
- **Icons:** Unicode emoji symbols
- **Styling:** Modern CSS with animations

## 📈 Performance

- **Training Time:** ~2-5 minutes per epoch (CPU)
- **Generation Speed:** <1 second per sample
- **Model Size:** ~4MB (generator + discriminator)
- **Memory Usage:** ~500MB during training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun and Corinna Cortes
- PyTorch team for the excellent deep learning framework
- Flask community for the web framework
- Font Awesome for inspiration (replaced with Unicode)

## 📞 Contact

**ethnolab50-creator**
- GitHub: [@ethnolab50-creator](https://github.com/ethnolab50-creator)
- Project Link: [https://github.com/ethnolab50-creator/cgan](https://github.com/ethnolab50-creator/cgan)

## 🎯 Future Enhancements

- [ ] GPU acceleration support
- [ ] Custom digit drawing interface
- [ ] Model performance metrics dashboard
- [ ] Export generated images
- [ ] Batch processing capabilities
- [ ] Model comparison tools
- [ ] Advanced GAN architectures (DCGAN, WGAN)

---

**⭐ Star this repo if you found it helpful!**

1. Enter the number of epochs (1-50 recommended)
2. Click "Train Model" to start training
3. Monitor the training progress in the modal dialog
4. Wait for training to complete

### Generating Digits

1. **Single Digit Generation:**
   - Select a digit (0-9) from the dropdown
   - Choose number of samples (1-10)
   - Click "Generate Digit"

2. **Generate All Digits:**
   - Click "Generate All Digits" to create samples for all digits (0-9)

### Model Management

- **Load Model:** Load a previously saved model
- **Save Model:** Models are automatically saved after training

## API Endpoints

- `GET /api/status` - Get model training status
- `POST /api/train` - Train the CGAN model
- `POST /api/generate` - Generate specific digit samples
- `GET /api/generate_all` - Generate samples for all digits
- `POST /api/load_model` - Load saved model

## Technical Details

- **Framework:** PyTorch
- **Dataset:** MNIST (automatically downloaded)
- **Architecture:** Conditional GAN with:
  - Generator: 4-layer CNN with conditional input
  - Discriminator: 4-layer CNN with conditional input
- **Optimizer:** Adam (lr=0.0002, beta1=0.5)
- **Loss:** Binary Cross Entropy
- **Device:** CPU (automatically detected)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Flask 2.3+
- NumPy, Matplotlib
- torchvision

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Troubleshooting

1. **PyTorch Installation Issues:**
   - Ensure you have the CPU version installed: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

2. **Port Already in Use:**
   - Change the port in `app.py` if 5000 is occupied

3. **Training Takes Too Long:**
   - Reduce the number of epochs
   - Training on GPU would be faster (requires CUDA installation)

4. **Memory Issues:**
   - Reduce batch size in the code if you encounter memory errors

## License

This project is for educational purposes. Feel free to modify and distribute.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.