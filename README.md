# Wine Classification using Naive Bayes

This project classifies wines into 3 categories using Gaussian and Multinomial Naive Bayes classifiers. The dataset used is the `wine` dataset from `sklearn.datasets`.

## Requirements

- scikit-learn
- matplotlib
- seaborn
- pandas

## Dataset

The dataset used is the `wine` dataset, which contains chemical analysis results of wines grown in the same region in Italy but derived from three different cultivars.

## Steps

1. **Load Dataset**: Load the wine dataset using `load_wine()` from `sklearn.datasets`.
2. **Data Split**: Split the data into training and testing sets using `train_test_split()` from `sklearn.model_selection`.
3. **Model Training**: Train the models using Gaussian and Multinomial Naive Bayes classifiers.
4. **Evaluation**: Evaluate the models on the test set and compare their performance.
5. **Visualization**: Plot the confusion matrices for both classifiers.

## Code

```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix

# Load dataset
data_set = load_wine()

# Split data into training and testing sets
input = data_set.data
label = data_set.target
x_train, x_test, y_train, y_test = train_test_split(input, label, random_state=10, test_size=0.2)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
gaussian_score = gaussian.score(x_test, y_test)
print(f'Gaussian Naive Bayes Score: {gaussian_score}')

# Confusion Matrix for Gaussian Naive Bayes
cn_gaussian = confusion_matrix(y_test, gaussian.predict(x_test))
sns.heatmap(cn_gaussian, cmap='Greens', annot=True, xticklabels=data_set.target_names, yticklabels=data_set.target_names)
plt.title("Confusion Matrix - Gaussian Naive Bayes")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()

# Multinomial Naive Bayes
multinomial = MultinomialNB()
multinomial.fit(x_train, y_train)
multinomial_score = multinomial.score(x_test, y_test)
print(f'Multinomial Naive Bayes Score: {multinomial_score}')

# Confusion Matrix for Multinomial Naive Bayes
cn_multinomial = confusion_matrix(y_test, multinomial.predict(x_test))
sns.heatmap(cn_multinomial, cmap='Blues', annot=True, xticklabels=data_set.target_names, yticklabels=data_set.target_names)
plt.title("Confusion Matrix - Multinomial Naive Bayes")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()

# Displaying Dataframe Info
df = pd.concat([pd.DataFrame(data_set.data, columns=data_set.feature_names), pd.DataFrame(data_set.target, columns=['target'])], axis='columns')
print(df.head())
print(df.min())
print(df.max())
