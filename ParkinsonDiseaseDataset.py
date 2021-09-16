
#importing libraries
import numpy as np
import pandas as pd

#importing dataset
dataset = pd.read_csv("parkinsons.csv")
X = dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23]].values
y = dataset.iloc[:,17].values

#splitting the dataset into train and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state =0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#applying PCA

from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
variance = pca.explained_variance_ratio_

#fitting into knn model

from sklearn.neighbors import KNeighborsClassifier
classifi = KNeighborsClassifier(n_neighbors = 8,p=2,metric ='minkowski')
classifi.fit(X_train,y_train)

#predicting reults
y1_pred = classifi.predict(X_test)

#fitting the model in SVM
from sklearn.svm import SVC
classifi2 = SVC()
classifi2.fit(X_train,y_train)

#predicting reults
y2_pred = classifi2.predict(X_test)

#fitting the data in random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifi3 = RandomForestClassifier(n_estimators=16,criterion = "entropy",random_state=0)
classifi3.fit(X_train,y_train)

#predicting reults
y3_pred = classifi3.predict(X_test)

#Analyzing
from sklearn.metrics import confusion_matrix,accuracy_score

#KNN model
print("----For KNN Model----")
cm=confusion_matrix(y_test,y1_pred)
print("Confusion Matrix: ")
print(cm)
print("Accuracy : " + str(accuracy_score(y_test,y1_pred)))

print()

#SVM model
print("----For SVM Model----")
cm2=confusion_matrix(y_test,y2_pred)
print("Confusion Matrix: ")
print(cm2)
print("Accuracy : " + str(accuracy_score(y_test,y2_pred)))

print()

#Random Forest Classifier Model
print("----For Forest Classifier Model----")
cm3=confusion_matrix(y_test,y3_pred)
print("Confusion Matrix: ")
print(cm3)
print("Accuracy : " + str(accuracy_score(y_test,y3_pred)))