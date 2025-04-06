# 🔬 Model Comparison - Audio Deepfake Detection

## ✅ 1. RawNet2 (Selected for Implementation)

**Key Innovation:**
- Directly processes raw waveforms using CNN + GRU + attention.

**Pros:**
- End-to-end system
- No handcrafted features
- Real-time potential

**Cons:**
- Raw waveform training is sensitive
- Requires careful normalization

---

## 🧠 2. SE-ResNet + SpecAugment

**Innovation:**
- Uses SE attention blocks with ResNet on spectrogram inputs.

**Pros:**
- High accuracy on known + unknown spoof types
- Deep model + data augmentation = strong generalization

**Cons:**
- Needs preprocessing (spectrograms)
- Higher latency

---

## 🟨 3. LCNN (Light CNN + MFM Activation)

**Innovation:**
- Max Feature Map layer separates useful features
- Compact CNN architecture

**Pros:**
- Lightweight and efficient
- Good baseline model

**Cons:**
- Slightly worse on generalization
- Lower accuracy than newer models
