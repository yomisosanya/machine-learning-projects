import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


uri = Path("../../res/iris.csv")
build_dir = Path("../build/")

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
pca.fit(X_scaled)

eigenvectors = pca.components_
eigenvalues = pca.explained_variance_

print("The eigenvalues are {}".format(eigenvalues))
print("The eigenvectors are:")
print(eigenvectors)

pca_range = np.arange(1, len(eigenvalues) + 1)

# plt.figure()
plt.plot(pca_range, eigenvalues, marker="o", linestyle="-")
plt.title("Scree Plot")
plt.xlabel("PCA")
plt.ylabel("Eigenvalues")
plt.grid(True)
plt.savefig(Path("../build/scree.png"))
