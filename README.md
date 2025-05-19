# 🏢 GAN-Based Medical Imaging Analysis for Brain Tumor Detection

Detecting brain tumors early can save lives. This project combines cutting-edge deep learning techniques and user-friendly design to create a powerful solution for medical imaging analysis.

---

## 🌟 Key Features

- **Tumor Detection**: Analyze uploaded brain images to detect tumor presence.  
- **Tumor Classification**: Identify the type of tumor.  
- **Tumor Localization**: Highlight the tumor's location on the image.  

The system is trained on a robust dataset with multiple tumor classes to ensure high accuracy and reliability.

---

## ⚙ Process Workflow

### 1. Image Preprocessing
Input images are refined for optimal analysis through:

- Resizing: Standardize dimensions  
- Rotation: Align images  
- Normalization: Ensure consistent pixel values  
- Cropping: Focus on the region of interest  
- Skull Stripping: Remove irrelevant parts  
- Denoising: Apply Gaussian filtering to remove noise  

---

### 2. Synthetic Image Generation with PGGAN
Progressive GAN (PGGAN) generates synthetic images that:

- Mimic high-resolution brain scans  
- Increase dataset variability, enhancing robustness  

**Why PGGAN?** Its progressive layer addition minimizes artifacts and ensures detailed, realistic images.

---

### 3. Data Augmentation
To further diversify the dataset, the following techniques are applied:

- Rotation  
- Shearing  
- Width/Height Shifts  
- Flipping  

---

### 4. Tumor Detection and Classification
Using **VGG19**, a pre-trained CNN:

- **Transfer Learning**: Retrain fully connected layers for tumor classification  
- **Activation Functions**:  
  - ReLU for hidden layers  
  - Softmax for multi-class predictions  
- **Normalization**: Accelerates training and improves performance  

---

### 5. Model Evaluation
Performance is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Cohen's Kappa Coefficient  
- AUC (Area Under the Curve)  
- Confusion Matrix  

---

## 📂 Project File Structure



```markdown
Directory structure:
└── singupurapusaicharan-Brain-tumor-detection-using-VGG19/
    ├── README.md
    ├── annotations.json.txt
    ├── annotations.py
    ├── app.py
    ├── errors.php
    ├── model.ipynb
    ├── model.py
    ├── register.php
    ├── requirements.txt
    ├── server.php
    ├── logs/
    │   ├── train/
    │   └── validation/
    ├── model/
    │   └── vgg19_model2.keras
    ├── static/
    │   ├── login.css
    │   ├── login.js
    │   ├── picture3.avif
    │   ├── result.css
    │   ├── script.js
    │   └── style.css
    └── templates/
        ├── index.html
        ├── login.html
        └── result.html
 

```


---

## 🧠 Technologies Used

### Deep Learning Models
- VGG19 for tumor classification

### Frontend
- HTML, CSS, JavaScript, React.js

### Backend
- Flask for model integration

### Programming & Frameworks
- Python, TensorFlow, Keras

### Tools
- Matplotlib for visualizations  
- Jupyter Notebook for training

---

## 📸 Example Screenshot



<img src="https://github.com/user-attachments/assets/d302f264-8c62-471d-8c0c-2e79e45224ac" width="400" height="350" />

*Sample output showing the detected tumor and its classification.*

---

## 💡 Why This Project Stands Out

- **Innovative**: Combines GANs for synthetic image generation and VGG19 for classification  
- **Impactful**: Focuses on real-world application in healthcare  
- **User-Friendly**: Delivered as an intuitive web application  
