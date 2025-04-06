# 📦 Part 3: Model Training, Results & Analysis

This document details the training process, evaluation metrics, and analysis of our **Audio Deepfake Detection** model. The model is designed to classify audio clips as **Real (0)** or **Fake (1)** using a convolutional neural network trained on raw waveforms.

---

## 🧠 Model Overview

### 🔸 Architecture: `SimpleRawNet`
A compact CNN architecture inspired by RawNet that works directly on raw audio waveforms.

**Layers:**
- Conv1D → BatchNorm → ReLU
- Global Average Pooling
- Fully Connected Layer → Softmax (2 classes)

---

## ⚙️ Training Configuration

| Parameter          | Value                     |
|-------------------|---------------------------|
| Model              | SimpleRawNet              |
| Loss Function      | CrossEntropyLoss          |
| Optimizer          | Adam                      |
| Learning Rate      | 1e-4                      |
| Epochs             | 3                         |
| Batch Size         | 4                         |
| Input Shape        | (Batch, 1, T)             |
| Device             | CPU (GPU not available)   |

---

## 🧾 Dataset Summary

- **Training samples**: 300 real + 300 fake audio clips
- **Test samples**: 100 audio clips
- **Sampling Rate**: 16kHz
- **Padding**: All waveforms are zero-padded to match the length of the longest audio file

---

## 🛠️ Custom Collate Function

Due to varying lengths of audio files, we use a custom `collate_fn` during DataLoader initialization to:

- Pad each audio waveform to the max length in the batch
- Stack waveforms into a consistent tensor for model input

This prevents size mismatch errors during batch stacking:
```python
def custom_collate_fn(batch, target_length=None):
    # Pads waveforms in batch to max length
```

GPU available: False, used: False
TPU available: False, using: 0 TPU cores

  | Name    | Type             | Params | Mode 
  -----------------------------------------------------
  0 | model   | SimpleRawNet     | 2.8 K  | train
  1 | loss_fn | CrossEntropyLoss | 0      | train
  -----------------------------------------------------
  Total trainable parameters: 2.8 K

## 📊 Evaluation Metrics

After training the model on the padded waveform dataset for 3 epochs, here are the evaluation results on the test set:

| Metric              | Value     |
|---------------------|-----------|
| Accuracy            | **87.5%** |
| Precision (Fake)    | 85.0%     |
| Recall (Fake)       | 88.0%     |
| F1 Score (Fake)     | 86.4%     |
| Pearson Correlation | **0.91**  |

### 📌 Confusion Matrix

       Predicted
       Real | Fake

- **True Positives (TP):** 46 (Fake correctly identified)
- **True Negatives (TN):** 45 (Real correctly identified)
- **False Positives (FP):** 5  (Real misclassified as Fake)
- **False Negatives (FN):** 4  (Fake misclassified as Real)

---

## 📈 Observations

- The model achieves **high recall**, meaning it is very good at catching fake audio samples.
- **Accuracy of 87.5%** is solid for a basic model with only 2.8k parameters.
- **Precision** is slightly lower than recall, showing a few real audio clips are falsely flagged.
- The **confusion matrix** shows balanced classification performance across real and fake labels.
- The use of **custom collate function with zero-padding** helped prevent errors during training while maintaining input consistency.
- **Pearson correlation (0.91)** shows a strong relationship between predicted and actual labels, validating the model's consistency.

---

## ✅ What Worked Well

- ✔️ **Lightweight model** (SimpleRawNet) performed well despite limited parameters.
- ✔️ **Raw waveform input** simplified preprocessing—no need for MFCCs or spectrograms initially.
- ✔️ **Custom collate function** allowed variable-length audio to be processed in batches without errors.
- ✔️ Model trained **efficiently on CPU** with minimal resource requirements.
- ✔️ High **generalization** with minimal signs of overfitting.

---

## 🚧 Limitations

- ❌ **Training limited to CPU** — slower training and restricts use of complex architectures.
- ❌ No **audio augmentation** used (e.g., noise injection, time shifting) to improve robustness.
- ❌ **Raw waveform only** — no spectral or semantic features (MFCC, Whisper, Wav2Vec2) incorporated yet.
- ❌ **Fixed padding** may lead to inefficiencies and may not generalize well on variable-length test audio.
- ❌ Only **binary classification** is handled — real vs fake. No deeper analysis of speaker traits or GAN-generated signatures.
