from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from feast import FeatureStore
import mlflow
import pandas as pd
import uvicorn

app = FastAPI(title="Fraud Detection API")

# 1. Connect to Feast (The "Mini-Fridge" / Redis)
store = FeatureStore(repo_path="feature_repo")

# 2. Connect to MLflow and load the Champion Model (The "Recipe Book")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model_uri = "models:/my_new_bouncer/latest"

print("Loading Champion Model from MLflow...")
try:
    # Use pyfunc so it loads whichever model won (Logistic, XGBoost, etc.)
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model (did you register it in MLflow?): {e}")
    model = None

# Define what an incoming card swipe looks like
class Transaction(BaseModel):
    cc_num: int
    amt: float

@app.post("/predict")
def predict_fraud(txn: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    # 3. Ask Redis for this specific card's latest history
    feature_vector = store.get_online_features(
        features=[
            "card_stats:txn_count_1h",
            "card_stats:txn_count_24h",
            "card_stats:avg_amt_24h",
            "card_stats:card_avg_amt"
        ],
        entity_rows=[{"cc_num": txn.cc_num}]
    ).to_dict()

    # 4. Format the features into a Pandas DataFrame for the model
    # Note: We must pass them in the exact same order the model was trained on
    df = pd.DataFrame(feature_vector)
    features_for_model = df[["txn_count_1h", "txn_count_24h", "avg_amt_24h", "card_avg_amt"]]

    # 5. Make the prediction
    fraud_prob = float(model.predict(features_for_model)[0])

    # Return the decision
    return {
        "cc_num": txn.cc_num,
        "transaction_amount": txn.amt,
        "fraud_probability": fraud_prob,
        "is_fraud": fraud_prob > 0.5 
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)