import streamlit as st
import pandas as pd
import numpy as np
import random
import datetime
import time
import joblib
import plotly.express as px
from tensorflow.keras.models import load_model, Model

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="Vehicle IDS Network", layout="wide")

st.title("🚗 Vehicle Intrusion Detection System")
st.info("Hybrid AI Model: LSTM Feature Extraction + Random Forest")

# ----------------------------------------------------
# VEHICLE TYPE MAP
# ----------------------------------------------------
vehicle_types = {
    0: "Car",
    1: "Bus",
    2: "Ambulance",
    3: "Truck"
}

# ----------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------
rf_model = joblib.load("hybrid_rf_model.pkl")
scaler = joblib.load("hybrid_scaler.pkl")
lstm_model = load_model("hybrid_lstm_model.keras")

feature_extractor = Model(
    inputs=lstm_model.input,
    outputs=lstm_model.layers[-2].output
)

# ----------------------------------------------------
# SESSION STATE
# ----------------------------------------------------
if "running" not in st.session_state:
    st.session_state.running = False

if "vehicles" not in st.session_state:
    st.session_state.vehicles = {}

if "log" not in st.session_state:
    st.session_state.log = []

# ----------------------------------------------------
# CONTROL PANEL
# ----------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶ Start Network"):
        st.session_state.running = True

with col2:
    if st.button("⏹ Stop Network"):
        st.session_state.running = False

with col3:
    if st.button("➕ Add Vehicle"):
        if len(st.session_state.vehicles) < 20:
            available_ids = [i for i in range(1, 21) if i not in st.session_state.vehicles]
            vid = random.choice(available_ids)

            vehicle_type = random.randint(0, 3)

            st.session_state.vehicles[vid] = {
                "type": vehicle_type
            }

# ----------------------------------------------------
# REMOVE VEHICLE
# ----------------------------------------------------
remove_id = st.number_input("Vehicle ID to Remove", min_value=1, max_value=20, step=1)

if st.button("Remove Vehicle"):
    if remove_id in st.session_state.vehicles:
        del st.session_state.vehicles[remove_id]

# ----------------------------------------------------
# DISPLAY ACTIVE VEHICLES
# ----------------------------------------------------
st.subheader("🚦 Active Vehicles")

cols = st.columns(5)

vehicle_items = list(st.session_state.vehicles.items())

for i in range(len(vehicle_items)):
    vid = vehicle_items[i][0]
    data = vehicle_items[i][1]

    with cols[i % 5]:
        vname = vehicle_types[data["type"]]
        st.success(f"Vehicle {vid}")
        st.write(vname)

# ----------------------------------------------------
# NETWORK SIMULATION
# ----------------------------------------------------
if st.session_state.running:

    if len(st.session_state.vehicles) < 2:
        st.warning("Add at least 2 vehicles")

    else:

        vehicles = list(st.session_state.vehicles.keys())

        sender = random.choice(vehicles)
        receiver = random.choice([v for v in vehicles if v != sender])

        sender_type = st.session_state.vehicles[sender]["type"]

        data = {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "sender": sender,
            "receiver": receiver,
            "vehicle_type": vehicle_types[sender_type],
            "priority": random.randint(1, 3),
            "pos_x": round(random.uniform(0, 100), 2),
            "pos_y": round(random.uniform(0, 100), 2),
            "weather": random.randint(0, 3),
            "speed": round(random.uniform(20, 120), 2),
            "brake": random.randint(0, 1),
            "acc": round(random.uniform(-4, 4), 2),
            "delay": round(random.uniform(0, 2), 2),
            "msg_rate": round(random.uniform(1, 10), 2),
            "duplicate": random.randint(0, 1)
        }

        # ----------------------------------------------------
        # FEATURE VECTOR
        # ----------------------------------------------------
        features = np.array([[
            sender_type,
            data["priority"],
            data["pos_x"],
            data["pos_y"],
            data["weather"],
            data["speed"],
            data["brake"],
            data["acc"],
            data["delay"],
            data["msg_rate"],
            data["duplicate"]
        ]])

        # SCALE FEATURES
        scaled_features = scaler.transform(features)

        # LSTM INPUT
        lstm_input = scaled_features.reshape((1, 1, scaled_features.shape[1]))

        # EXTRACT LSTM FEATURES
        lstm_features = feature_extractor.predict(lstm_input, verbose=0)

        # COMBINE FEATURES
        combined_features = np.hstack([scaled_features, lstm_features])

        # RANDOM FOREST PREDICTION
        prediction = rf_model.predict(combined_features)

        if prediction[0] == 1:
            data["status"] = "ATTACK 🚨"
        else:
            data["status"] = "SAFE ✅"

        st.session_state.log.append(data)

        if len(st.session_state.log) > 500:
            st.session_state.log = st.session_state.log[-500:]

# ----------------------------------------------------
# DISPLAY LOGS
# ----------------------------------------------------
if len(st.session_state.log) > 0:

    df = pd.DataFrame(st.session_state.log)

    if "ATTACK 🚨" in df["status"].values:
        st.error("🚨 Intrusion Detected in Vehicle Network!")

    st.subheader("📡 Live Vehicle Communication Feed")
    st.dataframe(df, use_container_width=True, height=350)

    st.subheader("📊 Speed Monitoring")
    fig1 = px.line(df, x="time", y="speed")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🚨 Attack Detection")
    attack_counts = df["status"].value_counts()
    fig2 = px.bar(x=attack_counts.index, y=attack_counts.values)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🌧 Weather Distribution")
    weather_counts = df["weather"].value_counts()
    fig3 = px.pie(values=weather_counts.values, names=weather_counts.index)
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------
# AUTO REFRESH
# ----------------------------------------------------
if st.session_state.running:
    time.sleep(1)
    st.rerun()