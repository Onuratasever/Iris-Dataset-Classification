import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data_from_file = pd.read_excel('Iris.xls')

# print(data_from_file)

data_x = data_from_file.iloc[:, :4].values
data_y = data_from_file.iloc[:, -1:].values

# print(data_y)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=0)

# ========================== Data Scaling =====================================

sc = StandardScaler()

x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

# ========================== Logistic Regression ==============================

logr = LogisticRegression(random_state=0)
logr.fit(x_train_scaled, y_train.ravel())

y_logistic_pred = logr.predict(x_test_scaled)

logistic_confusion_matrix = confusion_matrix(y_test,y_logistic_pred)

print("Logistic Regression Confusion Matrix:")
print(logistic_confusion_matrix)

# ========================== KNN Classifying ==================================

knn = KNeighborsClassifier(n_neighbors=4, metric='minkowski')
knn.fit(x_train_scaled, y_train.ravel())

y_knn_pred = knn.predict(x_test_scaled)

knn_confusion_matrix = confusion_matrix(y_test,y_knn_pred)

print("KNN Confusion Matrix:")
print(knn_confusion_matrix)

# ========================== SVC (Support Vector Machine) =====================

svc = SVC(kernel='rbf')
svc.fit(x_train_scaled, y_train.ravel())

y_svc_pred = svc.predict(x_test_scaled)

svc_confusion_matrix = confusion_matrix(y_test, y_svc_pred)

print("SVC Confusion Matrix:")
print(svc_confusion_matrix)

# ========================== Guassian Naive Bayes =============================

gnb = GaussianNB()
gnb.fit(x_train_scaled, y_train.ravel())

y_gnb_pred = gnb.predict(x_test_scaled)

gnb_confusion_matrix = confusion_matrix(y_test, y_gnb_pred)

print('GNB Confusion Matrix:')
print(gnb_confusion_matrix)

# ========================== Decision Tree ====================================

dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(x_train_scaled, y_train.ravel())
y_dtc_pred = dtc.predict(x_test_scaled)

dtc_confusion_matrix = confusion_matrix(y_test, y_dtc_pred)

print('Decision Tree Confusion Matrix:')
print(dtc_confusion_matrix)

# ========================== Random Forest ====================================

rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(x_train_scaled, y_train.ravel())

y_rfc_pred = rfc.predict(x_test_scaled)
rfc_confusion_matrix = confusion_matrix(y_test, y_rfc_pred)

print('Random Forest Confusion Matrix:')
print(rfc_confusion_matrix)