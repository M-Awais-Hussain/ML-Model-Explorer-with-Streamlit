import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
st.write(
    """
# Explore different ML models and datasets
## By: Muhammad Awais Hussain
    """
)

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

model_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random_Forest')
)

# Return both the data and feature names
def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    x = data.data
    y = data.target
    return x, y, data

x, y, data = get_dataset(dataset_name)

# Display dataset information
st.write("##### Shape of dataset:", x.shape)
st.write("##### Number of classes:", len(np.unique(y)))

# Display a preview of the dataset with feature names
st.write("#### First five rows of the dataset:")
st.write(pd.DataFrame(x, columns=data.feature_names).head())

# Parameter UI section for model parameters
def add_parameter_ui(model_name):
    param = dict()
    if model_name == 'KNN':
        n_neighbors = st.sidebar.slider('n_neighbors', 1, 15)
        param['n_neighbors'] = n_neighbors
    elif model_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        param['C'] = C
    elif model_name == 'Random_Forest':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        param['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        param['n_estimators'] = n_estimators
        
    return param

params = add_parameter_ui(model_name)

# Get classifier based on the selected model
def get_classifier(model_name, params):
    clf = None
    if model_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
    elif model_name == 'SVM':
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                     max_depth=params['max_depth'])
    
    return clf

clf = get_classifier(model_name, params)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# Fit the model
clf.fit(x_train, y_train)

# Predictions
y_pred = clf.predict(x_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)

# Display accuracy and classifier
st.write(f"Classifier = {model_name}")
st.write(f"Accuracy = {acc:.2f}")

# Display test set predictions vs actual values
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
st.write(results_df.head())


# PCA(Principle Component Analysis)
pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:,0]
x2 = x_projected[:,1]

fig  = plt.figure(figsize = (12,4))

plt.scatter(x1,x2,
        c = y, alpha=0.8,
        cmap='viridis')
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar()

st.pyplot(fig)



