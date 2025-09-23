import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


uri = Path("../../res/iris.csv")

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


def print_scores(title, cm, acc):

    print("{}\n".format(title))
    print("The confusion matrix is: ")
    print(cm)
    print("The sum of the values is {}".format(np.sum(cm)))
    print("The accuracy score: {}\n".format(acc))


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
    print_scores(title=title, cm=cm, acc=acc)


# print("length of column is {}".format(len(df.columns)))

# print(y)

# print(X)
