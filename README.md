# Heart Disease Detection Project

## Overview

This project implements a **Heart Disease Detection System** using both:

1. **Rule-Based Expert System** (Experta)
2. **Machine Learning Model** (Decision Tree Classifier)

It compares the performance of **human-defined rules** against **data-driven models** in predicting heart disease.

## Features

✔ **Data Preprocessing** - Cleaning and feature selection from `data/`
✔ **Data Visualization** - Jupyter notebooks for analysis (`notebooks/`)
✔ **Rule-Based Expert System** - Uses `experta` for risk assessment (`rule_based/`)
✔ **Machine Learning Model** - Trained Decision Tree Classifier (`ml_model/`)
✔ **Model Evaluation** - Accuracy, Precision, Recall, F1-score comparison
✔ **Comparison Script** - Benchmarks expert system vs. machine learning

## Installation

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/omarabdelhady802/heart.git
```

### 2️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

## Usage

### 🔹 Train the Machine Learning Model

```sh
python ml_model/train.py
```

### 🔹 Run the Expert System

```sh
python rule_based/experta_engine.py
```

### 🔹 Compare the Models

```sh
python report/comparison.py
```

## Project Structure

```
📂 heart-disease-detection
│── 📂 data                 # Heart disease dataset
│── 📂 ml_model             # Machine learning models (Decision Tree)
│── 📂 rule_based           # Rule-based expert system (Experta)
│── 📂 notebooks            # Jupyter notebooks for analysis
│── 📂 report               # Project documentation
│── 📂 utils                # Utility functions
│── heart_disease_model.pkl # Saved trained ML model
│── requirements.txt        # Required dependencies
│── README.md               # Documentation
```

## Dependencies

- Python 3.11+
- experta
- pandas
- numpy
- scikit-learn
- joblib

## Author

Omar Aabdelhady Mohamed 2305480
