import pandas as pd
from pathlib import Path

cols = ["cc_num", "trans_date_trans_time", "amt"]
df = pd.concat([
    pd.read_csv("data/raw/fraudTrain.csv", usecols=cols),
    pd.read_csv("data/raw/fraudTest.csv", usecols=cols),
])
df["event_timestamp"] = pd.to_datetime(df["trans_date_trans_time"], utc=True)
df = df.sort_values(["cc_num", "event_timestamp"]).reset_index(drop=True)

# closed="left" = only PAST transactions (the current one is excluded, no leakage)
g = df.set_index("event_timestamp").groupby("cc_num")["amt"]
df["txn_count_1h"] = g.rolling("1h", closed="left").count().values.astype("int64")
df["txn_count_24h"] = g.rolling("24h", closed="left").count().values.astype("int64")
df["avg_amt_24h"] = g.rolling("24h", closed="left").mean().fillna(0).values.astype("float32")

# average amount over the card's whole past
gg = df.groupby("cc_num")["amt"]
n_past = gg.cumcount()
df["card_avg_amt"] = ((gg.cumsum() - df["amt"]) / n_past.where(n_past > 0)).fillna(0).astype("float32")

out = df[["cc_num", "event_timestamp", "txn_count_1h", "txn_count_24h", "avg_amt_24h", "card_avg_amt"]]
Path("data/processed").mkdir(parents=True, exist_ok=True)
out.to_parquet("data/processed/card_features.parquet", index=False)
print(out.shape)
print(out.head())