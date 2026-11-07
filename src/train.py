import pandas as pd
from feast import FeatureStore
from xgboost import XGBClassifier
import mlflow
from sklearn.metrics import roc_auc_score, average_precision_score

# 1. Point to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("fraud-detection-feast")

def train():
    print("1. Loading raw keys...")
    cols = ["cc_num", "trans_date_trans_time", "is_fraud"]
    df = pd.concat([
        pd.read_csv("data/raw/fraudTrain.csv", usecols=cols),
        pd.read_csv("data/raw/fraudTest.csv", usecols=cols),
    ])
    df["event_timestamp"] = pd.to_datetime(df["trans_date_trans_time"], utc=True)
    entity_df = df[["cc_num", "event_timestamp", "is_fraud"]]

    print("2. Asking Feast for the new features...")
    store = FeatureStore(repo_path="feature_repo")
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "card_stats:txn_count_1h",
            "card_stats:txn_count_24h",
            "card_stats:avg_amt_24h",
            "card_stats:card_avg_amt"
        ]
    ).to_df()

    # Clean Pandas NaN
    training_data.loc[training_data["txn_count_1h"] < 0, "txn_count_1h"] = 0
    training_data.loc[training_data["txn_count_24h"] < 0, "txn_count_24h"] = 0

    print("3. Training the New Model (XGBoost)...")
    features = ["txn_count_1h", "txn_count_24h", "avg_amt_24h", "card_avg_amt"]
    
    # Split data
    train_size = int(len(training_data) * 0.8)
    train_df = training_data.iloc[:train_size]
    test_df = training_data.iloc[train_size:]

    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, n_jobs=-1)
    
    with mlflow.start_run(run_name="feast_xgboost"):
        model.fit(train_df[features], train_df["is_fraud"])
        
        preds = model.predict_proba(test_df[features])[:, 1]
        roc_auc = roc_auc_score(test_df["is_fraud"], preds)
        pr_auc = average_precision_score(test_df["is_fraud"], preds)
        
        mlflow.log_metrics({"roc_auc": roc_auc, "pr_auc": pr_auc})
        
        print("4. Saving to MLflow...")
        # Notice we are explicitly naming it here!
        mlflow.xgboost.log_model(
            model, 
            "model", 
            registered_model_name="my_new_bouncer"
        )
        print(f"Done! Saved 'my_new_bouncer'. PR AUC: {pr_auc:.4f}")

if __name__ == "__main__":
    train()