import json
import time
import datetime
import argparse
import pandas as pd
from confluent_kafka import Producer
import os


# ASSICURARSI CHE KAFKA DOCKER SIA ATTIVO !
# poi lanciare sul terminale PyCharm:
# python producer_sensor.py


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# -------- Path utility --------
def to_absolute_model_path(path_str):
    """
    Se path_str è già assoluto lo restituisce invariato.
    Se è relativo (es. models/rf_model.pkl) lo rende assoluto rispetto a PROJECT_ROOT.
    """
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(PROJECT_ROOT, path_str)


conf = {
    "bootstrap.servers": "kafka:9092",
    "client.id": "sensor-simulator",
    "acks": "all"
}

producer = Producer(conf)

parser = argparse.ArgumentParser()
parser.add_argument("--speedup", type=float, default=1.0,
                    help="Fattore di velocità simulazione (es. 10 = 10x più veloce)")
args = parser.parse_args()

# ======================
# Caricamento dataset
# ======================
data_path = "/data/ims_1test_raw.parquet" # to_absolute_model_path("ims_1test_raw.parquet")
if os.path.exists(data_path):
    try:
        df_raw = pd.read_parquet(data_path)
        print(f"[PRODUCER] Loaded data to be streamed: {data_path}")
    except Exception as e:
        print("[CLPRODUCEROUD] Failed loading data to be streamed:", e)



# ======================
# Callback consegna
# ======================
def delivery_report(err, msg):
    if err is not None:
        print("Delivery failed: {}".format(err))
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

# ======================
# Produzione messaggi
# ======================
for idx, row in df_raw.iterrows():
    message = {
        "bearing_id": row["bearing_id"],
        "filename": row["filepath"],
        "label": row["label"],
        "x": row["x"].tolist(),
        "y": row["y"].tolist()
    }

    producer.produce(
        "sensor-data",
        key=str(row["bearing_id"]),
        value=json.dumps(message),
        callback=delivery_report
    )

    producer.poll(0)

    # Flush ogni 100 messaggi
    if idx % 100 == 0:
        producer.flush()

    # >>> simulazione tempo reale acquisizione (1s per riga / speedup)
    time.sleep(1.0 / args.speedup)

producer.flush()
print(f"Finished producing all sensor data: {datetime.datetime.now()}.")
