import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# import iris dataset to play with
iris = datasets.load_iris()

iris = pd.DataFrame(iris.data, columns=iris.feature_names)

target = datasets.load_iris()
iris['target'] = target.target 

del(target)

print(type(iris))
print (iris.head())

print((iris.describe()))

# make a correlation matrix of all columns
plt.matshow(iris.corr())

import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))
corr = iris.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)

print(f)
# 


