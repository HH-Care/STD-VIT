
# 🧪 STD Image Classifier using Vision Transformer (ViT-B16)

This project uses a **Vision Transformer (ViT-B16)** model to classify penile-related STD images into six medical categories. The model is built using TensorFlow and `vit-keras`, with a custom classification head fine-tuned for sexually transmitted disease (STD) recognition.

---

## 🧠 Classes

The model predicts one of the following classes:

- `Genital_warts`
- `HSV` (Herpes Simplex Virus)
- `Normal`
- `Penile_cancer`
- `Penile_candidiasis`
- `Syphillis`

---

## 📁 File Structure

```
STD-VIT/
│
├── main.py           # Main script for model loading and prediction
├── README.md                   # Documentation
├── requirements.txt            # Python dependencies
└── weights/
    └── viT_std_model_weights.h5  # Pretrained model weights (not included in repo)
```

---

## ⚙️ Environment Setup

### 1. Create Virtual Environment

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Model Weights

📥 [Click to Download Weights](https://rb.gy/t1s353)

Save the `.h5` file inside the `weights/` folder or set the path using an environment variable.

---

## 🚀 Run Prediction

```bash
python main.py path/to/image.jpg
```

Or specify custom model weight path:

```bash
export STD_MODEL_WEIGHTS_PATH=/custom/path/to/weights.h5
python main.py path/to/image.jpg
```

---

## 📊 Output

- The model will print the predicted class and confidence.
- It will also **display the image with the prediction** using `matplotlib`.

---

## 🧠 Model Architecture

This project uses:
- **ViT-B16 (pretrained on ImageNet21k)** as a base
- Custom classification head: `Flatten -> Dense(6, softmax)`
- Fine-tuned on a proprietary dataset of penile STD images.

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only** and must **not be used for clinical or diagnostic purposes** without regulatory approval. Always consult a licensed healthcare provider.

---

## 👨‍💻 Authors

- **Thanveer Ahamad**
- **Janitha Prathapa**
- **Yudara Kularathne**

---

## 📬 License

This project is licensed under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).  
Commercial use is prohibited without prior permission. For more details, see the [LICENSE](./LICENSE) file.
