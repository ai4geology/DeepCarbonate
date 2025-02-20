# Thin Section Classification Project

This project is a deep learning framework for classifying thin section images into 22 different rock types. The code is organized to support training, validation, and testing of various models, with robust data handling and evaluation metrics.

## Project Structure
```
thin_section_classification/
├── carbonate_1/                # Dataset folder
│   ├── train/                  # Training images
│   ├── val/                    # Validation images
│   └── test/                   # Testing images
├── experiments/                # Saved models and experiment logs
├── dataset.py                  # Data loading and preprocessing
├── eval.py                     # Evaluation metrics and confusion matrix
├── main.py                     # Main script for training and testing
├── model.py                    # Model definitions (ResNet, DenseNet, etc.)
├── trainer.py                  # Training and evaluation logic
└── README.md                   # Project documentation
```

## Getting Started

### 1. Dataset Preparation

Ensure your dataset is organized in the `carbonate_1` folder with the following structure:
```
carbonate_1/
├── train/
│   ├── rock_type_1/
│   ├── rock_type_2/
│   └── ...
├── val/
│   ├── rock_type_1/
│   ├── rock_type_2/
│   └── ...
└── test/
    ├── rock_type_1/
    ├── rock_type_2/
    └── ...
```

Each class folder should contain images of that class.

### 2. Install Dependencies

Create a virtual environment and install the required Python packages:

```bash
pip install -r requirements.txt

```

### 3. Train a Model

Run the main script to train a model:

```bash
python main.py
```

### 4. Testing

After training, the best model will be saved in the experiments folder. You can test it using the test dataset.

### 5. Models Supported

- ResNet
- DenseNet
- EfficientNet
- MobileNetV2

### 6. Evaluation Metrics

- Loss
- Accuracy
- Precision
- Recall
- F1 Score
- AUC (for multi-class classification)
- Top-1 and Top-5 Accuracy
- Confusion Matrix

### 7. Example Output

Epoch 1/50
Training Set - Loss: 1.2345 | Accuracy: 45.67%
Validation Set - Loss: 1.1234 | Accuracy: 56.78%
Detailed Validation Metrics:
• AUC: 0.8765
• Precision: 0.7654
• Recall: 0.7890
• F1 Score: 0.7765
• Top-1 Accuracy: 56.78%
• Top-5 Accuracy: 80.12%

## 8. Contribution

This project is a collaborative effort. Below are the contributors and their specific roles:

- **[Keran Li](#)**  
  ![Keran Li](./source/img/your_photo.jpg)  
  *Project Lead & Developer*  
  - Designed the overall architecture of the project.  
  - Implemented the Deep Learning frameworks.  
  - Curated and organized the dataset.
  
## 9. Cite

For any questions or issues, please contact the project maintainer.

## License

This project is licensed under the MIT License.