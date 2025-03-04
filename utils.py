import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import create_ffnn

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1]  # All columns except last
    y = df.iloc[:, -1]  # Last column as target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def train_fog_nodes(X_train, y_train):
    input_dim = X_train.shape[1]
    model = create_ffnn(input_dim)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model
