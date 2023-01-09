# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 09:49:33 2021

@author: User
"""

# import libraries
from pandas import read_csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

###### STEP 1: LOAD DATASET ######

# define the location of the dataset
path = ''

# load the dataset
df = read_csv(path)

# info of data
df.info()

# print the first 5 data
print("\n")
print(df.head())

###### STEP 2: CLEAN THE DATASET ######

# checking for null values
df.dropna()
print(df.isna().sum())
print("\n")

# delete duplicate rows
df.drop_duplicates(inplace=True)

# calculate duplicates
dups = df.duplicated()

# report if there are any duplicates
print("\nAre there any duplicates?:", dups.any())
print("\n")

# remove columns
drop_cols = ['Unnamed: 32','id']
df = df.drop(drop_cols, axis = 1)

# assign one and zero to diagnosis
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# removing highly correlated features
corr_matrix = df.corr().abs() 
mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
upper_tri = corr_matrix.mask(mask)
to_drop = [x for x in upper_tri.columns if any(upper_tri[x] > 0.9)]
df = df.drop(to_drop, axis = 1)
print(f"The reduced dataset has {df.shape[1]} columns. \n")

# statistics of diagnosis
print(df['diagnosis'].value_counts())

# plotting the labels with the frequency 
Labels = ['ΚΑΛΟΗΘΗΣ','ΚΑΚΟΗΘΗΣ']
classes = pd.value_counts(df['diagnosis'], sort = True)
classes.plot(kind = 'bar', rot=0)
plt.title("ΣΥΧΝΟΤΗΤΑ ΕΜΦΑΝΙΣΗΣ ΚΑΘΕ ΤΥΠΟΥ ΟΓΚΩΝ")
plt.xticks(range(2), Labels)
plt.xlabel("ΚΑΤΗΓΟΡΙΑ")
plt.ylabel("ΣΥΧΝΟΤΗΤΑ")
plt.show()

# info of data
print("\n")
df.info()

# print the first 5 data
print("\n")
print(df.head())

# creating features and label 
x = df.drop('diagnosis', axis = 1)
y = df['diagnosis']

# removing the outliers
clf = LocalOutlierFactor()
pred = clf.fit_predict(x)
X_score = clf.negative_outlier_factor_
outlier_score = pd.DataFrame()
outlier_score["score"] = X_score
threshold = -2.0
filterr = outlier_score["score"] < threshold
outlier_index = outlier_score[filterr].index.tolist()
x = x.drop(outlier_index)
y = y.drop(outlier_index).values

# splitting data into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

print("\n")
print("X_train",len(X_train))
print("X_test",len(X_test))
print("Y_train",len(Y_train))
print("Y_test",len(Y_test))
print("\n")

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

###### STEP 3: RANDOM FOREST ALGORITHM ######

rand_forest = RandomForestClassifier(criterion = 'entropy', max_depth = 15, max_features = 'auto', 
                                  min_samples_leaf = 2, min_samples_split = 3, n_estimators = 150)
rand_forest.fit(X_train, Y_train)

y_pred = rand_forest.predict(X_test)

count = 0
sum = 0

for i in range(len(y_pred)): 
    if(y_pred[i] == Y_test[i]):
        print("Expected value: {}, Predicted value: {}  => Correct \n".format(Y_test[i],y_pred[i])) 
        count = count+1
    else:
       print("Expected value: {}, Predicted value: {}  => Incorrect \n".format(Y_test[i],y_pred[i])) 
print("\n")        
correct_percen = count/len(y_pred)*100

print("Percentage of correct prediction: {:.2f}% \n".format(correct_percen))   
incorrect_percen = 100-correct_percen

x = [correct_percen, incorrect_percen]
labels = ['Correct prediction', 'Incorrect prediction']
colors = ['tab:red', 'tab:blue']

fig, ax = plt.subplots()
ax.pie(x, labels = labels, colors = colors, autopct='%.2f%%')
ax.set_title('Random Forest prediction pie chart')
plt.show()