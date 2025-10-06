import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import sys


uri = Path("../../res/iris.csv")
out_file = open(Path("../build/compare_model.txt"), "w")

columns = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]

# load dataset
df = pd.read_csv(uri, names=columns)

# split dataset into features and class:
# features
X = df.iloc[:, :-1].copy()
# class
y = df.iloc[:, -1].copy()

# kf = KFold(n_splits=5, shuffle=True, randome_state=42)


def get_params():

    yield "Gaussian Naive Bayes", GaussianNB()

    yield "K Nearest Neighbor", KNeighborsClassifier()

    yield "Linear SVC", LinearSVC()

    yield "Decision Tree", DecisionTreeClassifier()

    yield "Random Forest", RandomForestClassifier()


def print_scores(title, cm, acc, file=sys.stdout):

    print("{}\n".format(title), file=file)
    print("The confusion matrix is: ", file=file)
    print(cm, file=file)
    print("The sum of the values is {}".format(np.sum(cm)), file=file)
    print("The accuracy score: {}\n".format(acc), file=file)


def get_scores(X, y, *, clf):
    """ """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = cross_val_predict(estimator=clf, X=X, y=y, cv=kf)

    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)

    return cm, acc


for title, model in get_params():
    """ """
    cm, acc = get_scores(X=X, y=y, clf=model)
    print_scores(title=title, cm=cm, acc=acc, file=out_file)


class_names = df["class"].unique()

for title, model in get_params():
    """ """
    cm, acc = get_scores(X=X, y=y, clf=model)
    # print_scores(title=title, cm=cm, acc=acc)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.title(title)
    out = Path("../build") / "compare-{}.png".format(title.lower().replace(" ", "-"))
    plt.savefig(out)
    plt.close()


out_file.close()
