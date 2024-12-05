from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def normalize_data(data):
    """Normalizes 'Time' and 'Amount' features and drops original columns."""
    scaler = StandardScaler()
    data['Scaled_Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data['Scaled_Time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)
    return data

def handle_class_imbalance(X, y):
    """Applies SMOTE to balance the dataset."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    #check new class distribution
    print(y_resampled.value_counts())
    return X_resampled, y_resampled
