#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

X = pd.read_csv("Data/training_data.csv.gz", sep=',',
                index_col=0)
training_targets = pd.read_csv("Data/training_targets.csv",
                               sep=',', index_col=0)
training_targets = training_targets.loc[X.index]
training_targets.fillna(inplace=True, value='no-call')
y = training_targets.replace(to_replace=["no-call", "resistant"],
                             value="not-sensitive")

prob = pd.DataFrame(columns=['drug', 'score'])
drug = "colo320dm"

for s in y.columns:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y[[s]], test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    pipe = make_pipeline(StandardScaler(), SelectFromModel(LinearSVC(penalty="l1", dual=False, max_iter=10000)), clf)
    pipe.fit(X_train, y_train.values.ravel())
    predict = pipe.predict_proba(X.loc[drug].to_numpy().reshape(1, -1))

    prob = prob.append({'drug': s, 'score': predict[0][1]}, ignore_index=True)

print(prob)
print(prob.iloc[prob["score"].argmax()])
