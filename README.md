# Colorectal Cancer Prediction Project

## Authors
- Kadir Altunel
- David Mosciszki
- Bryam Piedra

## Overview
This project focuses on leveraging machine learning, specifically Vision Transformers (ViT), to classify colorectal cancer histology images. By utilizing the Colorectal Histology dataset, we aim to enhance the speed and accuracy of colorectal cancer diagnosis.

## Motivation
Colorectal cancer is one of the leading causes of cancer-related deaths. Early and accurate diagnosis can significantly impact patient outcomes. Recent advancements in AI and deep learning, especially in medical imaging, provide new opportunities to improve diagnosis and prognosis.

## Dataset
The dataset used for this project is the **Colorectal Histology Dataset**, which includes:
- **5,000 histological images** (150x150 pixels, RGB, magnified 20x).
- **8 texture labels**: Adipose, Complex, Debris, Empty, Lympho, Mucosa, Stroma, Tumor.
- Images were digitized with an Aperio ScanScope.

More details can be found on the [TensorFlow Dataset Catalog](https://tensorflow.org/datasets/catalog/colorectal_histology).

## Methodology
### Vision Transformer (ViT)
We used a pre-trained Vision Transformer (ViT) model from Hugging Face. The key steps include:
1. Dividing images into patches and flattening them into vectors.
2. Feeding these vectors into a transformer encoder with positional embeddings.
3. Using a classification layer to predict the label.

### Implementation Steps
1. **Data Preparation**:
   - Images were preprocessed and split into training, validation, and test sets with stratified sampling.
   - Labels were encoded using `LabelEncoder`.

2. **Model Training**:
   - A custom ViT model was built using PyTorch, pre-trained on ImageNet-21k.
   - Key hyperparameters: 
     - Epochs: 50
     - Batch size: 32
     - Learning rate: 0.001
   - Optimizer: Adam.

3. **Evaluation**:
   - Training, validation, and test accuracies were monitored across epochs.
   - Final test accuracy: **97%**
   - Final validation accuracy: **96%**

### Results
- The ViT model demonstrated robust performance with high accuracy in classifying colorectal histology textures.
- Training and validation losses showed steady decreases, indicating effective learning.

## Visualizations
Loss and accuracy plots over epochs are provided in the project's results section. These showcase the model's strong and consistent performance.

## Future Work
- Extend the dataset to include other types of medical images.
- Experiment with different deep learning architectures, such as CNNs and advanced vision transformers.
- Optimize hyperparameters and incorporate techniques like data augmentation.

## References
1. Dosovitskiy, A., et al. [An Image Is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
2. Arun, I., et al. "Automation in Histopathology: Present and Future." Journal of Pathology Informatics, 2020.
3. "Colorectal Cancer: Epidemiology, Risk Factors, and Prognostic Factors." BMJ, 2020.
4. [Colorectal Histology Dataset](https://tensorflow.org/datasets/catalog/colorectal_histology)
5. [ViT Base Patch16 224 In21k on Hugging Face]([https://huggingface.co/google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k))

---

## Contact
For questions or collaboration inquiries, please reach out to the authors.

