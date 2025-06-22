
---


# 🐶🐱 Cat vs Dog Image Classifier with MobileNetV2

This project is a complete image classification pipeline built using TensorFlow and TensorFlow Hub to classify images as either **cats** or **dogs**. It uses the **MobileNetV2** pre-trained model as a feature extractor and is trained on the popular **Dogs vs. Cats dataset** from Kaggle.

---

## 📁 Dataset

- **Source:** [Kaggle Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats)
- Downloaded and extracted using the Kaggle API.
- Resized 2000 images (`1000 cats + 1000 dogs`) to `224x224` pixels and saved in a new directory.

---

## 🔧 Project Structure

```

📦cats-vs-dogs-classifier
┣ 📁 image resized/
┃ ┗ 📄 \[224x224 resized images]
┣ 📄 train.zip
┣ 📄 dogs-vs-cats.zip
┣ 📄 kaggle.json
┣ 📄 Untitled27.ipynb
┗ 📄 README.md

````

---

## 📌 Features

- ✅ Downloads and extracts dataset using Kaggle API
- ✅ Resizes and preprocesses images
- ✅ Generates binary labels (0 = cat, 1 = dog)
- ✅ Splits data into training and test sets
- ✅ Scales pixel values between 0 and 1
- ✅ Uses MobileNetV2 as a frozen feature extractor
- ✅ Classifies new images as Cat or Dog

---

## 📚 Libraries Used

- Python
- NumPy
- Matplotlib
- OpenCV (`cv2`)
- TensorFlow + TensorFlow Hub
- Scikit-learn (for `train_test_split`)
- Pillow (PIL)
- Google Colab helpers (e.g., `cv2_imshow`)

---

## 🧠 Model Architecture

- **Base Model:** [MobileNetV2 (TFHub)](https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/5)
- **Type:** Transfer Learning (Feature Extraction)
- **Final Layer:** `Dense(2, activation='softmax')`
- **Loss Function:** `SparseCategoricalCrossentropy(from_logits=True)`
- **Optimizer:** `Adam`

---

## 🏋️ Training

```python
model.fit(X_train_scaled, Y_train, epochs=5)
````

Trains for 5 epochs on 80% of the dataset (20% is used as test set).

---

## 🖼️ Prediction

After training, the user can input the path to an image:

```bash
Path of the image to be predicted: /content/test_image.jpg
```

The image is:

* Resized to 224x224
* Normalized
* Reshaped
* Passed to the model for prediction

Output:

```
The image represents a Cat
```

---

## 🧪 Example

<img src="https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg" width="200"/> → 🐱
Prediction: `Cat`

---

## 🛠 How to Run

1. Clone the repo:

```bash
git clone https://github.com/your-username/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier
```

2. Upload your `kaggle.json` in the notebook environment.

3. Run all cells in `Untitled27.ipynb` (or rename the notebook to something more meaningful).

---

## ✨ Future Improvements

* Add image augmentation for better generalization
* Support more animal classes (multiclass classification)
* Convert model to TensorFlow Lite for mobile deployment
* Use fine-tuning instead of frozen feature extraction

---

## 📜 License

This project is for educational purposes. Dataset is governed by Kaggle's competition rules.

---

## 🙌 Acknowledgments

* [Kaggle](https://www.kaggle.com/)
* [TensorFlow Hub](https://tfhub.dev/)
* [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)

---




