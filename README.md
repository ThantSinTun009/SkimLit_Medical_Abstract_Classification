# 🧠 SkimLit: Medical Abstract Classification

This project focuses on **automatic classification of sentences in medical abstracts** using different deep learning and machine learning approaches.

We experiment with multiple NLP models ranging from simple baselines to advanced embedding and positional encoding techniques.

---

## 📌 Problem Statement

Given a medical abstract, classify each sentence into one of the following categories:

- Background  
- Objective  
- Methods  
- Results  
- Conclusions  

This helps in **quick scientific literature understanding (SkimLit task)**.

---

## 🎥 YouTube Tutorial Series

The way I learnt project is explained step-by-step in the following video series (In *Burmese*):

### ▶️ Part 1: Baseline Model
Introduction to dataset and baseline machine learning model.
https://youtu.be/1evvGMdTgY4

---

### ▶️ Part 2: Convolutional 1D Models
Using Conv1D models for text classification.
https://youtu.be/ZmaiD9Y1S0w

---

### ▶️ Part 3: Embedding Models
Word embeddings and hybrid NLP architectures.
https://youtu.be/DupFb4CWGOM

---

### ▶️ Part 4: Positional Embeddings
Adding positional encoding for better sequence understanding.
https://www.youtube.com/watch?v=PART4_LINK

---

## 🏗️ Model Approaches Used

This project includes:

- Baseline TF-IDF + classical ML model
- 1D Convolutional Neural Network (Conv1D)
- Word Embedding models (Keras / TensorFlow Hub)
- Hybrid models (token + character inputs)
- Positional embeddings for sequence awareness

---

## 📁 Project Structure
```
SkimLit_Medical_Abstract_Classification/
│
├── skimlit_model1.keras # Trained deep learning model
├── app.py # Streamlit web app
├── requirements.txt
├── notebooks/
└── README.md
```
