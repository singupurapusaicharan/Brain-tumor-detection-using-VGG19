# GAN-Based Medical Imaging Analysis for Brain Tumor Detection

This project aims to develop a web application where users can upload brain images, and the system will determine whether a tumor is present, identify its type, and detect its location. The model is trained on a dataset containing different tumor classes.

Process Overview:
Image Preprocessing:

Input images undergo preprocessing steps, including resizing, rotation, normalization, cropping, skull stripping, and denoising (removing speckle noise using a Gaussian filter).
Synthetic Image Generation with PGGAN:

We utilize Progressive GAN (PGGAN) to generate synthetic images that resemble real samples. PGGAN progressively adds layers to both the generator and discriminator, allowing it to generate high-resolution images.
Data Augmentation:

To increase the diversity of training samples, we apply various augmentation techniques like rotation, shearing, width/height shifts, and flipping.
Tumor Detection and Classification:

The model used for tumor detection and classification is VGG19, a pre-trained convolutional neural network. We implement transfer learning, where the initial convolutional layers remain untrained, while the fully connected layers are replaced and trained on our dataset.
Normalization is applied for faster training and better model performance, while Softmax is used in the final layer for tumor class prediction and ReLU activation is applied in the hidden layers for non-linearity.
Model Evaluation:

The performance of the model is evaluated using various metrics, including accuracy, precision, recall, F1-score, Cohen's Kappa coefficient, AUC (Area Under the Curve), and a confusion matrix.
Technologies Used:
Deep Learning Models: VGG19 (for classification)
Frontend: HTML, CSS, JavaScript, React.js
Backend: Flask (to integrate the web app with the models)
Programming: Python
Frameworks: TensorFlow, Keras
Tools: Matplotlib for visualizations, Jupyter Notebook for model training
