# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** Tech Titans
**Team Members:**Sakshi Nigam, Rumaisha Qadeer, Shivam Patel, Kartik Kushwaha
**Submission Date:** 13 October, 2025


Smart Product Pricing Challenge — Documentation

1. Executive Summary

This project aims to accurately predict product prices using a multimodal deep learning model that combines visual and textual information.
The main idea is to integrate EfficientNet-B0 for image understanding and DistilBERT for text comprehension, allowing the model to learn meaningful connections between product visuals, descriptions, and pricing patterns.

2. Methodology Overview
2.1 Problem Analysis

The goal was to estimate the optimal price of online catalog products using both image and textual data.
Through exploratory data analysis, we found that:

Product descriptions contain important details such as brand, features, and materials.

Product images strongly influence price because they show design quality and visual appeal.

There were many outliers and missing images that needed preprocessing.

Combining text and image data provided better accuracy than using either alone.

Key Observations:

Cleaning the text by removing HTML tags and extra spaces improved tokenization.

SMAPE was a suitable metric since it treats percentage errors fairly.

Visual data augmentation helped prevent overfitting due to limited samples.

2.2 Solution Strategy

A multimodal hybrid approach was used to combine both language and visual features for predicting continuous price values.
Text features were extracted using DistilBERT, while image features were obtained using EfficientNet-B0.
Both features were merged and passed through fully connected layers for regression, using Smooth L1 Loss and optimized with AdamW and CosineAnnealingLR scheduling.

Approach type: Hybrid multimodal model (text + image)
Core innovation: Integration of pretrained language and vision models with fine-tuned projection layers to learn a shared price representation.

3. Model Architecture
3.1 Architecture Diagram
           ┌────────────────────────┐
           │   Product Description   │
           │  (Text Input)           │
           └──────────┬──────────────┘
                      │
                      ▼
              [DistilBERT Encoder]
                      │
                      ▼
                Text Feature Vector
                      │
                      │
           ┌────────────────────────┐
           │    Product Image        │
           │   (Visual Input)        │
           └──────────┬──────────────┘
                      │
                      ▼
           [EfficientNet-B0 Encoder]
                      │
                      ▼
                Image Feature Vector
                      │
                      ▼
          ┌──────────────────────────┐
          │   Concatenate Features    │
          └──────────┬───────────────┘
                      │
                      ▼
          Fully Connected Layers (ReLU)
                      │
                      ▼
          Regression Output → Predicted Price

3.2 Model Components

Text Processing Pipeline:

Preprocessing: Removed HTML tags, cleaned whitespace, truncated text to 128 tokens

Model: DistilBERT (pretrained transformer encoder)

Parameters: max length = 128, dropout = 0.3, ReLU activation, fine-tuned last transformer layer

Image Processing Pipeline:

Preprocessing: Resize, random crop, horizontal flip, color jitter, normalize

Model: EfficientNet-B0 (pretrained on ImageNet)

Parameters: input size = 192×192, dropout = 0.3, ReLU activation, early layers frozen during warm-up

4. Model Performance

Validation Results:

SMAPE Score: around 12.8 (best validation performance)

Mean Absolute Error: 0.36

Root Mean Squared Error: 0.48

R² Score: 0.89

The model achieved stable performance with early stopping after 6–7 epochs, showing good generalization and minimal overfitting.

5. Conclusion

This multimodal model successfully captures both textual and visual cues to predict product prices accurately.
By combining DistilBERT and EfficientNet representations, the system performs well and can be applied to real-world e-commerce pricing scenarios.
Key lessons include the importance of balanced multimodal fusion, proper regularization, and fine-tuning for stability and accuracy.

Appendix

A. Code and Dataset
Complete code, dataset, and outputs are available here:
Google Drive Link:
https://drive.google.com/drive/folders/150vq4pOQm8OtfZWukf3rQBYPtqY-EEgV

B. Additional Results

Training and validation loss curves showed smooth convergence.

Data augmentation reduced SMAPE variation across different folds.

The final model was saved as best_model.pth and predictions were stored in test_out.csv.