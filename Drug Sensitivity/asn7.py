#!/usr/bin/env python
# coding: utf-8

"""
- Importing data from internet then formatting/preprocessing it
    - Only interesting thing was combining no-calls and resistant into
    one category
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

X = pd.read_csv("Data/training_data.csv.gz", sep=',',
                index_col=0)
training_targets = pd.read_csv("Data/training_targets.csv",
                               sep=',', index_col=0)
training_targets = training_targets.loc[X.index]
training_targets = training_targets.replace(to_replace=["no-call", "resistant"],
                                            value="not-sensitive")
y = training_targets[["rapamycin"]]

"""
- Using a PCA to reduce dimensionality then plotting it and coloring
it by sensitivity to rapamycin
"""

sns.set_theme(style="darkgrid")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
sns.relplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y.values.ravel())

"""
- First step taken here is creating three models and implementing
sclaing and feature reduction(l1) in their pipelines
- Then a data fram is created for storing the scoring of each model
- Next each model is run through a cross validating scorer, and the
mean of 5 folds is used as it's score
- Finally, these scores are plotted along with the model to see how they preformed
"""

lda = LinearDiscriminantAnalysis(solver='lsqr',
                                 shrinkage='auto')
lr = LogisticRegression()
gpc = GaussianProcessClassifier()
rf = RandomForestClassifier()

accuracy = pd.DataFrame(columns=['model', 'score'])

for clf, name in [(lr, 'Logistic'),
                  (gpc, 'Gaussian Process'),
                  (lda, 'LinearDA'),
                  (rf, 'Random Forest')]:
    score = cross_val_score(make_pipeline(StandardScaler(),
                                          SelectFromModel(LinearSVC(penalty="l1", dual=False, max_iter=10000)), clf),
                            X, y.values.ravel(), cv=5)
    accuracy = accuracy.append({'model': name, 'score': score.mean()}, ignore_index=True)

sns.catplot(x="model", y="score", kind="bar", data=accuracy)
plt.show()
