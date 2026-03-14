import socket
import json
import time
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# ==============================
# SETTINGS
# ==============================

BASE_PORT = 5000
GUI_PORT = 6000
BUFFER_SIZE = 4096

# ==============================
# LOAD TRAINED MODELS (READ ONLY)
# ==============================

rf = joblib.load("hybrid_rf_model.pkl")
scaler = joblib.load("hybrid_scaler.pkl")
lstm_model = tf.keras.models.load_model("hybrid_lstm_model.keras")

print("Models loaded successfully.")

# Create LSTM feature extractor ONCE
feature_extractor = tf.keras.Model(
    inputs=lstm_model.input,
    outputs=lstm_model.layers[-2].output
)

# ==============================
# VEHICLE ID INPUT
# ==============================

VEHICLE_ID = input("Enter Vehicle ID (1-8): ").strip()
PORT = BASE_PORT + int(VEHICLE_ID)

# ==============================
# SOCKET SETUP
# ==============================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("localhost", PORT))

print(f"Vehicle {VEHICLE_ID} listening on port {PORT}")

# ==============================
# FEATURE ORDER (MUST MATCH TRAINING)
# ==============================

FEATURE_COLUMNS = [
    'vehicle_type', 'priority', 'pos_x', 'pos_y', 'weather',
    'speed', 'brake', 'acc', 'delay', 'msg_rate', 'duplicate'
]

# ==============================
# GENERATE RANDOM VEHICLE DATA
# ==============================

def generate_vehicle_data():
    return {
        "vehicle_id": VEHICLE_ID,
        "vehicle_type": random.randint(0, 3),
        "priority": random.randint(0, 1),
        "pos_x": random.uniform(0, 100),
        "pos_y": random.uniform(0, 100),
        "weather": random.randint(0, 2),
        "speed": random.uniform(20, 120),
        "brake": random.randint(0, 1),
        "acc": random.uniform(-5, 5),
        "delay": random.uniform(0, 2),
        "msg_rate": random.uniform(1, 10),
        "duplicate": random.randint(0, 1)
    }

# ==============================
# HYBRID PREDICTION
# ==============================

def hybrid_predict(data_dict):
    df = pd.DataFrame([data_dict])
    df = df[FEATURE_COLUMNS]

    scaled = scaler.transform(df)
    lstm_input = scaled.reshape((scaled.shape[0], 1, scaled.shape[1]))

    lstm_features = feature_extractor.predict(lstm_input, verbose=0)
    combined = np.hstack([scaled, lstm_features])

    prediction = rf.predict(combined)[0]
    return "ATTACK" if prediction == 1 else "SAFE"

# ==============================
# MAIN LOOP
# ==============================

while True:
    try:
        # Generate own vehicle data
        my_data = generate_vehicle_data()
        message = json.dumps(my_data)

        # Broadcast to other vehicles
        for i in range(1, 9):
            if str(i) != VEHICLE_ID:
                target_port = BASE_PORT + i
                sock.sendto(message.encode(), ("localhost", target_port))

        print(f"\nVehicle {VEHICLE_ID} Broadcasted Data")

        # Listen for incoming vehicle messages
        sock.settimeout(1)

        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            received = json.loads(data.decode())

            if received["vehicle_id"] != VEHICLE_ID:
                result = hybrid_predict(received)

                print(f"Received from Vehicle {received['vehicle_id']} → {result}")

                # Send full data + prediction to GUI
                received["prediction"] = result
                sock.sendto(
                    json.dumps(received).encode(),
                    ("localhost", GUI_PORT)
                )

        except (socket.timeout, ConnectionResetError):
            pass

        time.sleep(2)

    except KeyboardInterrupt:
        print(f"\nVehicle {VEHICLE_ID} stopped.")
        break