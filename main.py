from utils import preprocess_data, train_fog_nodes
from models import create_ffnn
import pickle

# Load and preprocess dataset
X_train, X_test, y_train, y_test = preprocess_data("dataset/processed_data_adult.csv")

# Train federated model at fog nodes
global_model = train_fog_nodes(X_train, y_train)

# Save global model for aggregation
with open("global_model.pkl", "wb") as f:
    pickle.dump(global_model.get_weights(), f)

print("✅ Federated learning process completed!")
