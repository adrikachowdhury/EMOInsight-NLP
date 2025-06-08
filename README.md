# 😊😔😠 Emotion Detection
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project crafts an emotion classification system by blending a Transformer (DistilBERT), a Bidirectional LSTM (BiLSTM), and a Multi-Layer Perceptron (MLP) on the SemEval-2019 Task 3 (EmoContext) dataset, targeting emotions (Happy, Sad, Angry, Others) in three-turn dialogues. The approach yields a performance that outstrips the original study's baseline and top result, underscoring the strength of this hybrid method. Micro-averaged F1 was chosen as the evaluation metric, aggregating true positives, false positives, and false negatives across all classes to weight each sample equally, making it ideal for the dataset's imbalance, where larger classes like "Others" dominate, ensuring a balanced overall performance assessment.

---

## 🚀 Features
The pipeline processes text data through the following stages:
1. **Preprocessing:** Cleans and normalizes text, preserving `[SEP]` separators for multi-turn context.
2. **Model**: Hybrid model consisting of Transformer, Sequential Learning, and classification through MLP.
5. **Evaluation:** Computes micro-averaged metrics and provides interpretability via LIME.

The project uses SMOTE to handle class imbalance and includes logging to track progress.

## 📊 Dataset
- **SemEval-2019 Task 3 (Cleansed version)** [A. Chatterjee et al., 2019 – ACL Anthology](https://aclanthology.org/S19-2005.pdf)
- Source: [HuggingFace Dataset](https://huggingface.co/datasets/oneonlee/cleansed_emocontext)

## 🧠 Model Architecture
1. **Transformer (DistilBERT):** Extracts contextual features from preprocessed text.
2. **BiLSTM:** Processes sequential features from DistilBERT.
3. **MLP:** Classifies emotions based on BiLSTM outputs.

## 🛠️ Tools and Libraries
- _Python 3.7+:_ The core programming language for the project.
- _PyTorch:_ Used to build and train the BiLSTM model and manage tensor operations.
- _Transformers (Hugging Face):_ Provides the pre-trained DistilBERT model and tokenizer for feature extraction, with the distilbert-base-uncased checkpoint.
- _scikit-learn:_ Implements the MLP classifier, SMOTE for class balancing, and evaluation metrics (e.g., precision, recall, F1, accuracy).
- _NLTK:_ Supports text preprocessing with stopwords, stemming, and lemmatization.
- _unidecode:_ Converts accented characters to ASCII for normalization.
- _emoji:_ Converts emojis to text representations (e.g., :smile:).
- _emoticon_fix:_ Corrects and standardizes emoticons in text.
- _lime:_ Enables LIME explainability to interpret model predictions.
- _joblib:_ Efficiently saves and loads scikit-learn models and scalers.
- _imbalanced-learn:_ Provides SMOTE for handling class imbalance in the training data.
- _pandas:_ Manages and processes dataset files (e.g., CSV handling).
- _numpy:_ Supports numerical operations and array manipulations.
- _matplotlib/seaborn:_ Visualizes label distributions for analysis.
- _logging:_ Tracks training progress and errors in training.log.

## ⚙️ Setup
1. Download/clone the repository from GitHub.
2. Install dependencies available on `requirements.txt`
3. Open `EmoDetect.ipynb` in a notebook (Google Colab/Jupyter) and run all cells to train and evaluate the model through hyperparameter tuning.
