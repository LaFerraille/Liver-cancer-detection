from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def xgbc(X_train, y_train, X_test, y_test):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def knn(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def voting_classifier(X_train, y_train, X_test, y_test, soft=True):
    model1 = LogisticRegression(max_iter=1000)
    model2 = XGBClassifier()
    model3 = RandomForestClassifier()
    model4 = KNeighborsClassifier()
    if soft:
        model = VotingClassifier(estimators=[('lr', model1), ('xgb', model2), ('rf', model3), ('knn', model4)], voting='soft')
    else:
        model = VotingClassifier(estimators=[('lr', model1), ('xgb', model2), ('rf', model3), ('knn', model4)], voting='hard')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
