from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def rf_train_model(X_train, y_train):
    """Trains a Random Forest Classifier."""
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model


def lr_train_model(X_train,y_train):
    """Trains a Logistic Regression.""" 
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train,y_train)
    return lr_model