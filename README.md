# Facial Emotion Recognition (FER2013)

Real-time facial **emotion recognition** using a compact CNN trained on the
**FER2013** dataset, with a live webcam demo.

Classifies 7 emotions: `Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise`.

## Pipeline

```
FER2013 CSV ──► preprocess (48x48 grayscale) ──► CNN (VGG-style) ──► emotion_model.h5 ──► webcam demo
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get the FER2013 dataset (CSV)
#    Download fer2013.csv and place it next to train.py.
#    Source: https://www.kaggle.com/datasets/msambare/fer2013
#    (train.py also tries an automatic mirror download)

# 3. Train
python train.py --epochs 40 --batch 64

# 4. Run the live webcam demo
python app.py
```

## Files

| File | Purpose |
|---|---|
| `train.py` | downloads data, builds & trains the CNN, saves `emotion_model.h5` |
| `app.py` | real-time webcam emotion recognition using the trained model |
| `new.ipynb` | full step-by-step notebook of the same pipeline |
| `requirements.txt` | Python dependencies |

## Training notes

- Input: 48×48 grayscale faces, pixels normalized to `[0, 1]`
- Architecture: 4× {Conv-BN-ReLU} blocks + dense head (1.5M params)
- Optimizer: Adam (1e-3) with ReduceLROnPlateau + EarlyStopping
- Uses the TensorFlow/Keras API

## Model quality

FER2013 is a notoriously noisy dataset (~55-60% human agreement on the
training crowdsourcing). A well-trained CNN on the full training split reaches
**~62-65% validation accuracy**; `Disgust` has few samples so it is often
under-represented. Improvements: class weighting, data augmentation (rotation,
shifts, flips), or a pretrained backend (VGG16/MobileNet features).

## License

MIT © 2026 Lalit Kumar