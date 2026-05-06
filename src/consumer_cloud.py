"""
Cloud-side Kafka consumer simulator for predictive-maintenance experiment.

- ML: RF or XGBoost on extracted features
- DL: Keras CNN (fallback)
- Simulates realistic latency profiles
- Logs metrics: latency, cpu, mem, accuracy, timestamp
- Optional CodeCarbon tracking
"""

# python consumer_cloud.py --bootstrap kafka:9092 --mode ml --scaler models/scaler.pkl --ml-model models/rf_model.pkl --out-csv cloud_results_ml_rf.csv --track-emissions --emissions-file cloud_emissions_ml_rf.csv
# python consumer_cloud.py --bootstrap kafka:9092 --mode ml --scaler models/scaler.pkl --ml-model models/xgb_model.pkl --out-csv cloud_results_ml_xgb.csv --track-emissions --emissions-file cloud_emissions_ml_xgb.csv
# python consumer_cloud.py --bootstrap kafka:9092 --mode dl --scaler models/raw_scaling_params.pkl --out-csv cloud_results_dl.csv --track-emissions --emissions-file cloud_emissions_dl.csv

import argparse, json, time, os
import pickle
import psutil
import joblib
from featuresExtraction import *
import os

BASE_DIR = os.path.dirname(__file__)          # /app/src
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # /app
#PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# -------- Path utility --------
def to_absolute_model_path(path_str):
    """
    Se path_str è già assoluto lo restituisce invariato.
    Se è relativo (es. models/rf_model.pkl) lo rende assoluto rispetto a PROJECT_ROOT.
    """
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(BASE_DIR, path_str)


# -------- confluent_kafka imports
try:
    from confluent_kafka import Consumer, Producer
    KAFKA_AVAILABLE = True
except Exception:
    KAFKA_AVAILABLE = False

# -------- Keras fallback --------
KERAS_AVAILABLE = False
try:
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.models import load_model
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False




# -------- TFLite runtime / TF fallback --------
TFLITE_AVAILABLE = False
tflite_interpreter = None
try:
    import tflite_runtime.interpreter as tflite_runtime_interpreter
    tflite_interpreter = tflite_runtime_interpreter
    TFLITE_AVAILABLE = True
except Exception:
    try:
        import tensorflow as tf
        tflite_interpreter = tf.lite
        TFLITE_AVAILABLE = True
    except Exception:
        TFLITE_AVAILABLE = False

# Keras fallback (for .h5 or SavedModel, if needed)
KERAS_AVAILABLE = False
try:
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.models import load_model
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False



# -------- CodeCarbon --------
CODECARBON_AVAILABLE = False
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except Exception as e:
    print(f"[WARN] CodeCarbon import failed: {e}")
    CODECARBON_AVAILABLE = False


# -------- Latency simulation --------
LATENCY_PROFILES = {
    "low": (0.02, 0.05),
    "medium": (0.08, 0.12),
    "high": (0.15, 0.25)
}

class TFLiteModel:
    def __init__(self, model_path: str):
        import tensorflow as tf
        self.interp = tf.lite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.input_details = self.interp.get_input_details()
        self.output_details = self.interp.get_output_details()
        # Non facciamo subito allocate_tensors, lasciamo spazio al resize
        print(f"[DL] Model loaded from {model_path}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        input_index = self.input_details[0]['index']
        expected_shape = tuple(self.input_details[0]['shape'])

        # Normalizza input: batch + float32
        if x.ndim == 1:  # sequenza mono-canale
            x = np.expand_dims(x, axis=0)  # aggiunge batch
            x = np.expand_dims(x, axis=-1)  # aggiunge channel
        elif x.ndim == 2:  # (batch, length) -> aggiungo channel
            x = np.expand_dims(x, axis=-1)

        x = x.astype(np.float32)

        # Resize dinamico se serve
        if not np.array_equal(expected_shape, x.shape):
            print(f"[DL] Resizing input tensor from {expected_shape} to {x.shape}")
            self.interp.resize_tensor_input(input_index, x.shape)
            self.interp.allocate_tensors()
            self.input_details = self.interp.get_input_details()
            self.output_details = self.interp.get_output_details()

        self.interp.set_tensor(input_index, x)
        self.interp.invoke()
        return self.interp.get_tensor(self.output_details[0]['index'])



def simulate_latency(profile):
    low, high = LATENCY_PROFILES[profile]
    time.sleep(np.random.uniform(low, high))

# -------- ML loader --------
def load_ml_pipeline(scaler_path, model_path):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ML model not found: {model_path}")
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    print(f"[CLOUD] ML pipeline loaded: scaler={scaler_path}, model={model_path}")
    return scaler, model

# -------- DL loader --------
def load_dl_pipeline(tflite_path=None, keras_path=None):
    if tflite_path and TFLITE_AVAILABLE and os.path.exists(tflite_path):
        try:
            tmodel = TFLiteModel(tflite_path)
            print(f"[CLOUD] Loaded TFLite model: {tflite_path}")
            return ("tflite", tmodel)
        except Exception as e:
            print("[CLOUD] TFLite load failed:", e)
    if keras_path and KERAS_AVAILABLE and os.path.exists(keras_path):
        try:
            kmodel = load_model(keras_path)
            print(f"[CLOUD] Loaded Keras model: {keras_path}")
            return ("keras", kmodel)
        except Exception as e:
            print("[CLOUD] Keras load failed:", e)
    raise RuntimeError("No DL runtime available or model files missing.")

    #old cnn = load_model(keras_path, compile=False)
    # old print(f"[CLOUD] DL CNN loaded from {keras_path}")
    # old return cnn


def main(args):
    print(f"\n=== Avvio consumer cloud in modalità {args.mode} ===")
    cpu_limit = float(os.getenv("CPU_LIMIT", 10))

    if not KAFKA_AVAILABLE:
        print("[CLOUD] Warning: confluent_kafka non disponibile. Serve Kafka per lo streaming.")

    consumer, producer = None, None
    if KAFKA_AVAILABLE:
        consumer = Consumer({
            'bootstrap.servers': args.bootstrap,
            'group.id': 'cloud-group',
            'auto.offset.reset': 'earliest'
        })
        consumer.subscribe([args.topic])
        producer = Producer({'bootstrap.servers': args.bootstrap})

    # ----- Label encoder
    label_path = to_absolute_model_path(args.label_encoder_file)
    label_encoder = None
    if os.path.exists(label_path):
        try:
            label_encoder = joblib.load(label_path)
            print(f"[CLOUD] Loaded label encoder: {label_path}")
        except Exception as e:
            print("[CLOUD] Failed loading label encoder:", e)

    # Pipelines
    #ml_scaler, ml_model, dl_model = None, None, None
    #dl_scaler = None  # dict con mean/std per x e y

    if args.mode == "ml":
        scaler_path = to_absolute_model_path(args.scaler)
        model_path = to_absolute_model_path(args.ml_model)
        ml_scaler, ml_model = load_ml_pipeline(scaler_path, model_path)

    else:
        tflite_path = to_absolute_model_path(args.tflite_path)
        dl_kind, dl_model = load_dl_pipeline(tflite_path=tflite_path)

        scaler_path = to_absolute_model_path(args.scaler)
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    dl_scaler = pickle.load(f)
                print(f"[CLOUD] Loaded DL scaling params: {scaler_path}")
            except Exception as e:
                print("[CLOUD] Failed to load DL scaling params:", e)

        '''
        keras_path = to_absolute_model_path(args.keras_model)
        dl_model = load_dl_pipeline(keras_path=keras_path)

        scaler_path = to_absolute_model_path(args.scaler)
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    dl_scaler = pickle.load(f)
                print(f"[CLOUD] Loaded DL scaling params: {scaler_path}")
            except Exception as e:
                print("[CLOUD] Failed to load DL scaling params:", e)
                '''


    # Emissions tracker
    tracker = None
    if CODECARBON_AVAILABLE and args.track_emissions:
        try:
            tracker = EmissionsTracker(save_to_file=True,
                                       measure_power_secs=15, # output_dir=os.path.join(PROJECT_ROOT, "logs"),
                                       output_file=args.emissions_file,
                                       tracking_mode="process" # LUIS ADDED tracking mode
                                       #cpu_count=8,  # o 16
                                       #cpu_model = "Intel Xeon"
                                       ) # cpu_power = 65 LUIS aggiunto cpu power per forzare codecarbon a vedere cloud meno constrained dell'edge
            tracker.start()
            print("[CLOUD] EmissionsTracker started.")
        except Exception as e:
            print("[CLOUD] EmissionsTracker could not be started:", e)
    print("[CLOUD] Consumer ready. Mode:", args.mode)

    # --- Process object for per-process CPU/mem ---
    proc = psutil.Process(os.getpid())
    logs, total, correct = [], 0, 0

    idle_count = 0
    max_idle = 30

    try:
        while True:
            msg = consumer.poll(timeout=1.0) if KAFKA_AVAILABLE else None
            if msg is None:
                idle_count += 1
                if idle_count >= max_idle:
                    print("[CLOUD] Nessun messaggio ricevuto per troppo tempo. Chiudo consumer.")
                    break
                continue
            idle_count = 0

            if KAFKA_AVAILABLE:
                if msg.error():
                    print("[CLOUD] Kafka error:", msg.error())
                    continue

                try:
                    data = json.loads(msg.value().decode("utf-8"))
                except Exception:
                    data = msg.value()
                    if isinstance(data, bytes):
                        data = json.loads(data.decode("utf-8"))
            else:
                print("[CLOUD] No Kafka available. Exiting.")
                break

            frame_id = data.get("id", None)
            true_label = data.get("label", None)
            x = np.array(data.get("x", []))
            y = np.array(data.get("y", []))
            if x.size == 0 or y.size == 0:
                print("[CLOUD] Warning: empty signal received, skipping frame", frame_id)
                continue

            start_time = time.time()
            simulate_latency(args.latency_profile)

            # Inference
            if args.mode == "ml":
                fx = extract_all_features(x, fs=args.fs)
                fy = extract_all_features(y, fs=args.fs)
                if fx is None or fy is None or len(fx) == 0 or len(fy) == 0:
                    print(f"[CLOUD] Warning: feature extraction failed, skipping frame {frame_id}")
                    continue

                # --- Trasformazione in array ordinato ---
                fx_arr = np.array([v for k, v in (fx.items())], dtype=np.float32)
                fy_arr = np.array([v for k, v in (fy.items())], dtype=np.float32)

                feat_vec = np.concatenate([fx_arr, fy_arr]).reshape(1, -1)
                feat_scaled = ml_scaler.transform(feat_vec)
                pred_num = ml_model.predict(feat_scaled)[0]
                pred_out = (
                    label_encoder.inverse_transform([int(pred_num)])[0]
                    if label_encoder is not None else str(pred_num)
                )
            else:
                sig = np.stack([x, y], axis=-1).reshape((1, len(x), 2))
                if dl_scaler is not None:
                    sig_x = (sig[0, :, 0] - dl_scaler["global_mean_x"]) / (dl_scaler["global_std_x"] + 1e-8)
                    sig_y = (sig[0, :, 1] - dl_scaler["global_mean_y"]) / (dl_scaler["global_std_y"] + 1e-8)
                    sig = np.stack([sig_x, sig_y], axis=-1).reshape((1, len(sig_x), 2))
                else:
                    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

                t_feat = 0.0
                t_inf_start = time.time()
                if dl_kind == "tflite":
                    out = dl_model.predict(sig)
                    pred_num = int(np.argmax(out, axis=1)[0])
                else:
                    out = dl_model.predict(sig, verbose=0)
                    pred_num = int(np.argmax(out, axis=1)[0])

                pred_out = (
                    label_encoder.inverse_transform([pred_num])[0]
                    if label_encoder is not None else str(pred_num)
                )
                t_inf = time.time() - t_inf_start
                '''
                sig = np.stack([x, y], axis=-1).reshape((1, len(x), 2))
                if dl_scaler is not None:
                    sig_x = (sig[0, :, 0] - dl_scaler["global_mean_x"]) / (dl_scaler["global_std_x"] + 1e-8)
                    sig_y = (sig[0, :, 1] - dl_scaler["global_mean_y"]) / (dl_scaler["global_std_y"] + 1e-8)
                    sig = np.stack([sig_x, sig_y], axis=-1).reshape((1, len(sig_x), 2))
                else:
                    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

                out = dl_model.predict(sig, verbose=0)
                pred_num = int(np.argmax(out, axis=1)[0])
                pred_out = (
                    label_encoder.inverse_transform([pred_num])[0]
                    if label_encoder is not None else str(pred_num)
                )'''

            latency_ms = (time.time() - start_time) * 1000.0

            # --- CPU/memoria processo ---
            cpu_cloud = proc.cpu_percent(interval=0.1)
            cpu_normalized = cpu_cloud / cpu_limit
            mem_cloud = proc.memory_percent()

            total += 1
            if true_label is not None and str(pred_out) == str(true_label):
                correct += 1
            acc_cloud = correct / total if total > 0 else 0.0

            row = {
                "id": frame_id,
                "pred": pred_out,
                "true": true_label,
                "latency_ms": latency_ms,
                "cpu_percent": cpu_cloud,
                "cpu_perc_norm": cpu_normalized,
                "mem_percent": mem_cloud,
                "accuracy_cumulative": acc_cloud,
                "timestamp": time.time(),
                "mode": args.mode,
                "profile": "cloud"
            }
            logs.append(row)

            if args.produce_results and KAFKA_AVAILABLE:
                try:
                    producer.produce(args.result_topic, json.dumps(row).encode("utf-8"))
                except Exception as e:
                    print("[CLOUD] Error producing result:", e)
                if total % 50 == 0 and KAFKA_AVAILABLE:
                    producer.flush()

            if len(logs) >= args.flush_every:
                df = pd.DataFrame(logs)
                header = not os.path.exists(args.out_csv)
                df.to_csv(args.out_csv, index=False, mode='a' if not header else 'w', header=header)
                logs = []

            print(f"[CLOUD] Frame {frame_id} | pred {pred_out} | true {true_label} | "
                  f"lat {latency_ms:.1f} ms | cpu {cpu_normalized}% | mem {mem_cloud}% | acc {acc_cloud:.3f}")

    except KeyboardInterrupt:
        print("[CLOUD] Interrupted by user.")
    finally:
        if logs:
            df = pd.DataFrame(logs)
            header = not os.path.exists(args.out_csv)
            df.to_csv(args.out_csv, index=False, mode='a' if not header else 'w', header=header)
        if KAFKA_AVAILABLE and consumer is not None:
            consumer.close()
        if tracker is not None:
            try:
                tracker.stop()
            except Exception:
                pass
        print("[CLOUD] Terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ml", "dl"], required=True)
    parser.add_argument("--latency-profile", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--bootstrap", default="kafka:9092")
    parser.add_argument("--topic", default="sensor-data")
    parser.add_argument("--produce-results", action="store_true")
    parser.add_argument("--result-topic", default="cloud-results")
    parser.add_argument("--scaler", help="Path scaler per ML o DL", default="models/scaler.pkl")  #, per DL "raw_scaling_params.pkl"
    parser.add_argument("--ml-model") #, default="models/rf_model.pkl"
    parser.add_argument("--keras-model", dest="tflite_path", default="models/cnn1d_raw.tflite")  #
    parser.add_argument("--label-encoder", dest="label_encoder_file", default="models/label_encoder.pkl")  #
    parser.add_argument("--fs", type=int, default=20000)
    parser.add_argument("--out-csv", dest="out_csv", default="cloud_results.csv")
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--track-emissions", action="store_true")
    parser.add_argument("--emissions-file", dest = "emissions_file", default="cloud_emissions.csv")
    args = parser.parse_args()
    main(args)
