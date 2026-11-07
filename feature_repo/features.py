from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

card = Entity(name="card", join_keys=["cc_num"])

source = FileSource(
    path="../data/processed/card_features.parquet",
    timestamp_field="event_timestamp",
)

card_stats = FeatureView(
    name="card_stats",
    entities=[card],
    ttl=timedelta(0),  # features never expire
    schema=[
        Field(name="txn_count_1h", dtype=Int64),
        Field(name="txn_count_24h", dtype=Int64),
        Field(name="avg_amt_24h", dtype=Float32),
        Field(name="card_avg_amt", dtype=Float32),
    ],
    source=source,
)