import numpy as np
import pandas as pd
from pathlib import Path


uri = Path("../../res/iris.csv")

columns = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]

# load dataset
df = pd.read_csv(uri, names=columns)
