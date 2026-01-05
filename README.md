# Smile Detection with EfficientNetB3

This project implements a **binary image classification system** to detect whether a person is **smiling or not smiling** using deep learning and computer vision techniques.  
The model is trained on the **GENKI-4K dataset** and tested in real-time using a webcam.

The project covers the full pipeline from **face detection and alignment** to **model training, evaluation, and real-time inference**.

---

## Project Summary

- **Task**: Binary classification (Smile vs. Non-Smile)
- **Input**: Face images extracted and aligned from raw images or webcam frames
- **Output**: Smile probability (Sigmoid output)
- **Model**: EfficientNetB3 (Transfer Learning)
- **Frameworks**: TensorFlow / Keras, OpenCV
- **Face Detection**: MTCNN
- **Application**: Offline evaluation + Real-time webcam inference

---

## Model Architecture

- **Backbone**: EfficientNetB3 (pretrained on ImageNet, include_top=False)
- **Custom Classification Head**:
  - Global Average Pooling
  - Dropout (rate = 0.4)
  - Dense (512 units, ReLU activation)
  - Dropout (rate = 0.3)
  - Dense (1 unit, Sigmoid activation)
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy, AUC

---

## Dataset

- **Dataset**: GENKI-4K
- **Classes**:
  - `smile` → label `1`
  - `non_smile` → label `0`
- **Preprocessing Steps**:
  - Face detection using MTCNN
  - Face alignment based on eye landmarks
  - Cropping with padding
  - Resizing to `300 × 300`
  - Data augmentation (flip, rotation, zoom, contrast)

---

## Training Procedure

1. **Face detection, alignment, and cropping**
2. **Dataset split**:
   - Train / Validation / Test
3. **Stage 1**: Train classification head (backbone frozen)
4. **Stage 2**: Fine-tuning last layers of EfficientNetB3
5. **Class balancing** using class weights
6. **Callbacks**:
   - EarlyStopping
   - ModelCheckpoint
   - ReduceLROnPlateau

---

## Evaluation

Model performance is evaluated using:
- Confusion Matrix
- Precision, Recall, F1-score
- AUC score

```python
print(classification_report(y_true, y_pred))
```
---

## Real-Time Demo (Webcam)

The trained model is deployed for real-time smile detection using a webcam:

- Detects and aligns face in each frame
- Predicts smile probability
- Displays label and confidence score live

---

## Technologies Used

- **Programming Language**: Python
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, MTCNN
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn

---

## Course Information

This project was developed as part of the Deep Learning course at the University of Isfahan.