import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
import streamlit as st
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import confusion_matrix

data_set = load_digits()

x_train, x_test, y_train, y_test = train_test_split(data_set.data,data_set.target,test_size=0.2,random_state=10)

st.set_page_config(page_title="Digits classification",page_icon='1️⃣',layout='wide')

st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Digits classification using SVM (Support Vector Machine)</h1>", unsafe_allow_html=True)
st.write("""Train SVM classifier using sklearn digits dataset (i.e. from sklearn.datasets import load_digits) and then,

1) Measure accuracy of your model using different kernels such as rbf and linear.\n
2) Tune your model further using regularization and gamma parameters and try to come up with highest accurancy score\n
3) Use 80% of samples as training data size\n""")

st.subheader("Dataset images (example)")

col1,col2,col3 = st.columns(3)
fig, ax = plt.subplots()

with col1:
    ax.matshow(data_set.images[0], cmap='gray')
    st.pyplot(fig)

with col2:
    ax.matshow(data_set.images[1], cmap='gray')
    st.pyplot(fig)

with col3:
    ax.matshow(data_set.images[2], cmap='gray')
    st.pyplot(fig)

st.write("Training data size 80%")
st.code('''x_train, x_test, y_train, y_test = train_test_split(data_set.data,data_set.target,test_size=0.2,random_state=10)''')

k1 , k2 , k3 = st.columns(3)

rbf = joblib.load('rbf.job')
linear = joblib.load('linear.job')
poly = joblib.load('poly.job')
model = joblib.load('model.job')

with k1:
    cm = confusion_matrix(y_test,rbf.predict(x_test))
    k1.subheader("rbf kernel:")
    k1.code("Score :"+str(rbf.score(x_test,y_test)))
    fig, ax = plt.subplots()
    sns.heatmap(cm,cmap='Blues',annot=True)
    ax.set_title("Confusion matrix heat map representation")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    k1.pyplot(fig)

with k2:
    cm = confusion_matrix(y_test,linear.predict(x_test))
    k2.subheader("linear kernel:")
    k2.code("Score :"+str(linear.score(x_test,y_test)))
    fig, ax = plt.subplots()
    sns.heatmap(cm,cmap='Greens',annot=True)
    ax.set_title("Confusion matrix heat map representation")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    k2.pyplot(fig)

with k3:
    cm = confusion_matrix(y_test,poly.predict(x_test))
    k3.subheader("poly kernel:")
    k3.code("Score :"+str(poly.score(x_test,y_test)))
    fig, ax = plt.subplots()
    sns.heatmap(cm,cmap='Reds',annot=True)
    ax.set_title("Confusion matrix heat map representation")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    k3.pyplot(fig)

st.subheader("Tuning the model")
st.write("SVM model with rbf kernel and gamma = 2 gives highest scores")
st.code("Score :"+str(model.score(x_test,y_test)))
