# Tuberculosis Detection from Chest X-rays

🩺 A deep learning Streamlit app to classify chest X-rays as **Normal** or **Tuberculosis** using transfer learning (VGG16, ResNet50, EfficientNet).

## 📦 Features

- Upload chest X-ray images
- Get instant TB/Normal prediction
- Built with Streamlit
- Evaluated with precision, recall, F1-score

## 🧠 Models Used

- ✅ ResNet50
- ✅ VGG16 (Best Accuracy: 93%)
- ✅ EfficientNetB0

| Model            | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ---------------- | -------- | --------- | ------ | -------- | ------- |
| **ResNet50**     | 57%      | 0.55      | 0.82   | 0.66     | 0.7678  |
| **VGG16**        | 89%      | 0.91      | 0.87   | 0.89     | 0.9508  |
| **EfficientNet** | 50%      | 0.50      | 1.00   | 0.67     | 0.1936  |


