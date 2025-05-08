# ML Model Explorer with Streamlit

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.0+-green.svg)
![Scikit-learn](https://img.shields.io/badge/scikitlearn-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

An interactive web application for exploring machine learning models on different datasets with real-time visualization.

---

## Features

- **Three Popular Datasets**: Iris, Breast Cancer, and Wine datasets
- **Multiple Classifiers**: KNN, SVM, and Random Forest
- **Interactive Parameter Tuning**: Adjust model parameters via sliders
- **Performance Metrics**: Real-time accuracy scores
- **Data Visualization**: PCA projection visualization
- **Responsive UI**: Clean, user-friendly interface

---

## Installation

1. Clone the repository:
```
git clone https://github.com/M-Awais-Hussain/ML-Model-Explorer-with-Streamlit.git
```

2. Install dependencies:
```
pip install -r requirements.txt
```

---

## Usage
Run the application:
```
streamlit run app.py
```
The app will automatically open in your default browser at http://localhost:8501

---

## Interface
1. Sidebar Controls:

  - Select dataset (Iris, Breast Cancer, or Wine)

  - Choose classifier (KNN, SVM, or Random Forest)

  - Adjust model-specific parameters

2. Main Display:

  - Dataset information and preview

  - Model performance metrics

  - Actual vs Predicted values

  - 2D PCA visualization

---

## License
This project is licensed under the MIT License 
