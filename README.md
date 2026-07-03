from google.colab import files
uploaded = files.upload()

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# LOAD DATASET
df = pd.read_csv('bank-full.csv', sep=';') # Added sep=';'
print("First 5 Rows")
print(df.head())

# DATASET INFORMATION

print("\nDataset Information")
print(df.info())
# CHECK MISSING VALUES
print("\nMissing Values")
print(df.isnull().sum())

# REMOVE DUPLICATES
df.drop_duplicates(inplace=True)

# ENCODE CATEGORICAL COLUMNS
label_encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

# FEATURES AND TARGET
X = df.drop('y', axis=1)
y = df['y']
# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# BUILD DECISION TREE MODEL

model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# PREDICTION

y_pred = model.predict(X_test)
# ACCURACY
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score")
print(accuracy)


# CLASSIFICATION REPORT

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# CONFUSION MATRIX


print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))


# VISUALIZE DECISION TREE


plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True
)

plt.show()
