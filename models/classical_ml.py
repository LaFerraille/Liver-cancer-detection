from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, SparsePCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from src.random_baseline import compute_random_baseline

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

def plot_feature_importances(X, y, k=10):

    # Train the classifier
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)

    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns
    plt.figure(figsize=(10, 6))
    plt.title(f"{k} Most Important Features")
    plt.bar(range(k), importances[indices[:k]], align='center')
    plt.xticks(range(k), [features[i] for i in indices[:k]], rotation=90)
    plt.xlim([-1, k])
    plt.show()

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Define your model training and prediction function here
def train_model(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def pipeline(X, y, apply_pca=False, n_components=15, sparse=False):

    # Encode y
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA if selected
    if apply_pca:
        pca = PCA(n_components=n_components)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
    
    if sparse:
        pca = SparsePCA(n_components=n_components, alpha=5)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)

    # Define the models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier(),
        'KNN': KNeighborsClassifier(),
        'Voting Classifier': VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000)),
                ('rf', RandomForestClassifier()),
                ('xgb', XGBClassifier())
            ],
            voting='hard'
        )
    }

    # Compute random baseline accuracy
    accuracies = {'Random Baseline': compute_random_baseline(y_test)}

    # Train models and store results
    for model_name, model in models.items():
        y_pred = train_model(model, X_train_scaled, y_train, X_test_scaled)
        accuracies[model_name] = accuracy_score(y_test, y_pred)

    # Sort models by accuracy
    sorted_model_names = sorted(accuracies, key=accuracies.get)
    sorted_accuracies = [accuracies[model_name] for model_name in sorted_model_names]

    # Plotting the results with a line plot and dotted lines
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_model_names, sorted_accuracies, 'o--')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.grid(True)
    plt.show()

    return sorted_model_names, sorted_accuracies


