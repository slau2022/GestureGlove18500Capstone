'''
SVM Accuracy:  0.7361538461538462
Perceptron Accuracy:  0.6369230769230769
KNN Accuracy:  0.7438461538461538
Random forest Accuracy:  0.8176923076923077  
Neural net accuracy:  0.7307692307692307
'''

import pandas as pd 
import sklearn
from sklearn import preprocessing
train_df = pd.read_csv('simulated_train_data.csv',delimiter=' ') 
test_df = pd.read_csv('simulated_test_data.csv',delimiter=' ')
headers = train_df.columns.values
label = headers[0]
features = headers[1:]

scaled = preprocessing.scale(train_df[features])
X = pd.DataFrame(scaled, columns=features)
y = train_df[label]

scaled_test = preprocessing.scale(test_df[features])
X_test = pd.DataFrame(scaled_test, columns=features)
y_test = test_df[label]


from sklearn import metrics

#svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
model = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, max_iter=1500)) #SVC(gamma='auto')
model.fit(X, y)
y_pred = model.predict(X_test)  
print("SVM Accuracy: ", metrics.accuracy_score(y_test,y_pred))

#perceptron
from sklearn.linear_model import Perceptron
model = Perceptron(tol=1e-3, random_state=0)
model.fit(X,y)
y_pred = model.predict(X_test)
print("Perceptron Accuracy: ", metrics.accuracy_score(y_test,y_pred))

#knn
from sklearn.neighbors import KNeighborsClassifier 
model = KNeighborsClassifier(n_neighbors=19).fit(X,y)
y_pred = model.predict(X_test)  
print("KNN Accuracy: ", metrics.accuracy_score(y_test,y_pred))

#random forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100).fit(X,y)
y_pred = model.predict(X_test) 
print("Random forest Accuracy: ", metrics.accuracy_score(y_test,y_pred))

#neural net 
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(solver='lbfgs',alpha=1e-5, hidden_layer_sizes=(10,7),random_state=1)
model.fit(X,y)
y_pred = model.predict(X_test) 
print("Neural net accuracy: ", metrics.accuracy_score(y_test,y_pred))
