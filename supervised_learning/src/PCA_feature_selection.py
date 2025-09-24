import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# import sys


uri = Path("../../res/iris.csv")
build_dir = Path("../build/")
output_txt = open(Path("../build/pca.txt"), "w")

columns = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]

# load dataset
df = pd.read_csv(uri, names=columns)

# split dataset into features and class:
# features
X = df.iloc[:, :-1].copy()
# class
y = df.iloc[:, -1].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

eigenvectors = pca.components_
eigenvalues = pca.explained_variance_

print("PCA Feature Selection\n", file=output_txt)

print("\nThe eigenvalues are {}\n".format(eigenvalues), file=output_txt)
print("The eigenvectors are:", file=output_txt)
print(eigenvectors, file=output_txt)

pca_range = np.arange(1, len(eigenvalues) + 1)

# plt.figure()
plt.plot(pca_range, eigenvalues, marker="o", linestyle="-")
plt.title("Scree Plot")
plt.xlabel("PCA")
plt.ylabel("Eigenvalues")
plt.grid(True)
plt.savefig(Path("../build/scree.png"))
plt.close()


kf = KFold(n_splits=5, shuffle=True, random_state=42)


y_pred = cross_val_predict(
    estimator=DecisionTreeClassifier(),
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    X=X_pca,
    y=y,
)

cm = confusion_matrix(y, y_pred=y_pred)
acc = accuracy_score(y, y_pred=y_pred)

print("\nThe confusion matrix is:", file=output_txt)
print("{}\n".format(cm), file=output_txt)
print("The accuracy score is: {}\n".format(acc), file=output_txt)


output_txt.close()
