# Emotion Detection using Deep Learning

## Table of Contents

- [Overview](#overview)
- [Objectives & Use Cases](#objectives--use-cases)
- [Dataset](#dataset)
- [Data Preparation & Augmentation](#data-preparation--augmentation)
- [Model Architectures](#model-architectures)
  - [ResNet50](#resnet50)
  - [Custom CNN](#custom-cnn)
- [Results & Comparison](#results--comparison)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [License](#license)

---

## Overview

This project implements facial emotion recognition using deep learning on the FER-2013Plus dataset. The goal is to classify facial images into 8 emotion categories: **anger, contempt, disgust, fear, happiness, neutral, sadness, and surprise**. Two models were developed and evaluated: a transfer learning approach using **ResNet50** and a custom-built **CNN**. The CNN significantly outperformed ResNet50 for this task.

---

## Objectives & Use Cases

- **Real-Time Emotion Analysis:** Analyze users' emotions on social media in real time.
- **Content Personalization:** Dynamically recommend content based on detected emotional states (e.g., uplifting content for sadness).
- **Emotional Support:** Enable emotionally aware interfaces for more human-like interactions.
- **User Engagement:** Adapt social media feeds to user moods, boosting satisfaction and retention.
- **Privacy-Centric AI:** Process emotional data locally to ensure user privacy and security.[2]

---

## Dataset

- **Source:** FER-2013Plus
- **Images:** 35,485 grayscale, 48x48 pixels
- **Emotion Categories:** Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
- **Split:** 28,386 training, 7,099 test images
- **Challenges:** Low resolution and overlapping features between some emotions[2]

---

## Data Preparation & Augmentation

To improve robustness and generalizability, the following augmentations were applied:
- **Random Rotations:** ±15°
- **Width/Height Shifts:** ±10%
- **Zooming:** 0.9–1.1x
- **Horizontal Flipping**
- **Rescaling:** Pixel values normalized to [0, 1][2]

---

## Model Architectures

### ResNet50

- **Approach:** Transfer learning with ResNet50, adapted for grayscale 48x48 images
- **Strengths:** Robust feature extraction, skip connections to avoid vanishing gradients
- **Limitations:** Designed for high-res RGB images; overfitting and poor generalization on low-res grayscale images[2]

### Custom CNN

- **Architecture:**
  - 4 convolutional layers (128, 256, 512, 512 filters)
  - MaxPooling and Dropout after each conv layer
  - Flatten → Dense(512) → Dropout → Dense(256) → Dropout → Dense(8, softmax)
- **Advantages:** Tailored for low-res grayscale images, strong localized feature detection


---

## Results & Comparison

| Metric           | ResNet50 | Custom CNN |
|------------------|---------|------------|
| **Test Accuracy**| 0.4957  | 0.7688     |
| **Test Loss**    | 1.4054  | 0.6645     |
| **Precision**    | 0.40    | 0.79       |
| **Recall**       | 0.25    | 0.52       |
| **F1-Score**     | 0.23    | 0.56       |
| **Weighted Avg F1** | 0.43 | 0.75       |

- **CNN outperformed ResNet50** in all metrics.
- **CNN** is better suited for grayscale, low-resolution emotion datasets.
- **ResNet50** struggled due to dataset mismatch and class imbalance.[2]

---

## Conclusion

- The custom CNN achieved **77% accuracy** and superior precision/recall compared to ResNet50 (49% accuracy).
- Model architecture is critical: custom CNNs may outperform pre-trained models for specialized, low-resolution tasks.
- The project demonstrates the importance of matching model complexity and data characteristics for optimal performance.[2]

---

## How to Run

1. **Clone the Repository**
    ```
    git clone <https://github.com/Yashtated/Facial-Emotion-Recognition-FER-Using-the-FER13-Dataset>
    ```
2. **Install Dependencies**
    ```
    pip install -r requirements.txt
    ```
3. **Prepare Data**
    - Organize the FER-2013Plus dataset into `train/` and `test/` folders, each with subfolders for each emotion.
4. **Train Models**
    ```
    python train.py --model cnn --epochs 30
    python train.py --model resnet --epochs 30
    ```
5. **Evaluate**
    ```
    python evaluate.py --checkpoint models/cnn_best.h5
    ```

---

## License

This project is for academic and educational purposes only.


