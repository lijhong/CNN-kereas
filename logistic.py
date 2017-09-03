import numpy as np
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()
X = iris.data[:, :2]  
Y = iris.target
#np.unique(Y)   # out: array([0, 1, 2])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train, Y_train)

prepro = logreg.predict_proba(X_test_std)
acc = logreg.score(X_test_std,Y_test)