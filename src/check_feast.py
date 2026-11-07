import pandas as pd
from feast import FeatureStore

store = FeatureStore("feature_repo")
cc = int(pd.read_parquet("data/processed/card_features.parquet").cc_num.iloc[0])

res = store.get_online_features(
    features=["card_stats:txn_count_1h", "card_stats:txn_count_24h",
              "card_stats:avg_amt_24h", "card_stats:card_avg_amt"],
    entity_rows=[{"cc_num": cc}],
).to_dict()
print(res)