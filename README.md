# audio-deepfake-detection

# 🎙️ Audio Deepfake Detection - Momenta Assignment

## 🚀 Goal

Develop an ML/DL model to detect AI-generated (deepfake) human speech.

## 🧠 Research Summary

We explored 3 promising approaches for audio deepfake detection:
1. **RawNet2** (Selected for implementation)
2. SE-ResNet with SpecAugment
3. LCNN with Max Feature Map activation

Details of each model are available in [`research/model_comparisons.md`](research/model_comparisons.md).

## ✅ Selected Model

- **RawNet2 (Simplified Version)**: Operates on raw audio waveforms.
- Designed for real-time detection without handcrafted features like MFCC or spectrograms.


## 📊 Results

- Accuracy on small ASVspoof subset: ~85%
- Real-time compatible with minimal preprocessing

## 📚 Dataset

Download subset of **ASVspoof 2019 LA** from:
👉 https://zenodo.org/record/2677612

## 🛠️ Getting Started

```bash
git clone https://github.com/Abhishek0-7/audio-deepfake-detection
cd audio-deepfake-detection-momenta
pip install -r requirements.txt

```
Then run the notebook:

```bash
jupyter notebook notebooks/implementation.ipynb
```


---

### ✅ `requirements.txt`

```txt
torch>=2.0
torchaudio
torchmetrics
pytorch-lightning
librosa
jupyter
numpy


