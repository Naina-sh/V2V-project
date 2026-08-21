# V2V / VANET Command Center 🚗📡

**Real-Time Intrusion Detection & Road Safety Monitoring Console for Vehicular Ad-hoc Networks (VANETs)**

This project implements a hybrid machine-learning Intrusion Detection System (IDS) for Vehicle-to-Vehicle (V2V) communication networks, wrapped in a professional NOC-style (Network Operations Center) real-time monitoring dashboard built with Streamlit.

Every message exchanged between simulated vehicles is classified as **SAFE ✅** or **ATTACK 🚨** by a hybrid **LSTM + Random Forest** pipeline, while a road-safety engine continuously watches vehicle speeds and inter-vehicle distances for collision risks — all visualized live on a tactical network map.

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      GUIapp.py  (Streamlit Console)                │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ Control Panel│  │ Live Network Map │  │ Alerts / Connections │  │
│  │ + Inspector  │  │  (Plotly, 1 km²) │  │ (real-time feed)     │  │
│  └──────┬───────┘  └────────┬─────────┘  └──────────┬───────────┘  │
│         └──────────┬────────┴───────────────────────┘              │
│              Simulation Engine (vehicle physics,                    │
│              mobility, weather, join/leave events)                 │
└──────────────────────┬─────────────────────────────────────────────┘
                       │ V2V messages (features)
                       ▼
        ┌──────────────────────────────────────┐
        │        HYBRID IDS PIPELINE           │
        │  1. StandardScaler (hybrid_scaler)   │
        │  2. LSTM feature extractor           │
        │     (hybrid_lstm_model.keras)        │
        │  3. RF classifier                    │
        │     (hybrid_rf_model.pkl)            │
        │  → SAFE ✅  /  ATTACK 🚨             │
        └──────────────────────────────────────┘
```

**IDS pipeline in detail** — for every message sent from vehicle A to vehicle B, the following features are extracted: sender type, priority, position (x, y), weather, speed, brake status, acceleration, delay, message rate, duplicate flag, and distance. Features are scaled, passed through an LSTM whose activations become deep features, and the concatenation of scaled features + LSTM features is classified by a Random Forest.

**Road-safety engine** — runs every tick alongside the IDS and raises three levels of alerts:
- ⚠️ **Speed warning** — vehicle exceeds the high-speed threshold
- ⚠️ **Distance warning** — two connected vehicles closer than the warning distance
- 🔴 **Collision risk** — critical distance exceeded at high speed

### Project Structure

| Path | Purpose |
|---|---|
| `GUIapp.py` | ⭐ Main Streamlit NOC console (simulation + IDS + dashboard) |
| `HybridModel_v2.py` | Trains the hybrid LSTM + Random Forest IDS (`hybrid_*` models) |
| `vehicle_model_smote_v2.py` | Trains the standalone RF IDS with SMOTE balancing |
| `fusion_ids_system_v2.py` | Evaluates the fusion IDS on test data |
| `MERGED_DATASET_v2.py` | Merges/cleans source datasets into the training dataset |
| `vehicle_node.py` | Standalone socket-based vehicle node with hybrid IDS |
| `vehicle_aes.py`, `crypto_aes.py` | AES-encrypted V2V socket communication demo |
| `corrected_model.py`, `new.py` | Experimentation / helper scripts |
| `final_merged_v2v_fixed.csv`, `v2v_smart_dataset.csv`, `vehicle_network_datasetkaggle.csv` | Datasets |

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- The pre-trained model files (already included in this repo): `hybrid_lstm_model.keras`, `hybrid_rf_model.pkl`, `hybrid_scaler.pkl`

### 1. Install dependencies

```bash
pip install streamlit pandas numpy joblib plotly scikit-learn tensorflow cryptography
```

### 2. Launch the dashboard

```bash
streamlit run GUIapp.py
```

The console opens in your browser. Press **▶ START** in the Control Panel to bring the network to life.

### 3. (Optional) Re-train the models

```bash
python MERGED_DATASET_v2.py     # build the merged dataset
python HybridModel_v2.py        # train hybrid LSTM + RF IDS
```

---

## 🖥 Dashboard Guide

The console is organized into a numbered flow — a viewer can tell *what is happening* at a glance:

### Command Header & KPI Cards
The top banner shows network status (LIVE/OFFLINE), current weather, join/leave counters and the clock. Below it, six color-coded KPI cards answer "what's happening right now":
**Vehicles Online · Active Links · IDS Attacks (red) · Collision Risks (rose) · Safety Warnings (amber) · Messages Logged (green)**

### 01 · Control Panel (left)
- **Simulation Control** — Start/Stop the network, add/remove vehicles (Car, Bus, Ambulance, Truck)
- **Alert Thresholds** — communication range, max vehicles, high-speed / warning-distance / critical-distance thresholds
- **Vehicle Inspector** — select any vehicle for full details: position, heading, speed, neighbours, nearest vehicle, IDS status and safety status

### 02 · Live Network Map (center)
A 1 km × 1 km tactical map of the network:
- **Node symbols** differ per vehicle type; node colour signals state: red = under attack, amber = too close to another vehicle, yellow = speeding
- **Attacked vehicles** get a red alarm halo ring
- **Link lines** are colour-coded by severity: faint gold = normal, amber = proximity warning, red = collision risk
- Faint circles show each vehicle's communication range — hover any node for speed, IDS status and position

### 03 · Monitoring (right)
- **Live Alerts** — severity chips summarizing the situation (ATTACK / COLLISION / SPEED / DISTANCE / TOTAL), a filter to show only certain alert types, and the scrolling alert feed with tagged messages
- **Active Connections** — every live V2V link with its distance, sortable, colour-coded by severity

### 04 · Detailed Analysis (bottom)
Global filters (vehicle ID, vehicle type, IDS status) scope **all** tabs:
- **LIVE TRAFFIC** — every message exchanged, newest first
- **IDS** — safe/attack breakdown (donut + bar) and the flagged attack messages
- **SAFETY** — current speeds, inter-vehicle distances and collision-risk pairs
- **VEHICLES** — full registry of every vehicle in the network
- **NETWORK EVENTS** — join/leave history
- **ANALYTICS** — speed-over-time, IDS verdict history, weather distribution and the full message log

---

## 🛠 Tech Stack

- **Frontend:** Streamlit, Plotly, custom CSS (warm glossy NOC theme)
- **ML:** TensorFlow/Keras (LSTM), scikit-learn (Random Forest, StandardScaler, SMOTE)
- **Data:** pandas, NumPy
- **Crypto demo:** `cryptography` (AES/Fernet)
- **Companion UI:** React (`vehicle-ui/`)

---

## 📄 License

Academic project — see `PROJECT_DOCUMENTATION_DETAILED.md` for the full report.

| `hybrid_lstm_model.keras`, `hybrid_rf_model.pkl`, `hybrid_scaler.pkl` | Pre-trained models used by the GUI |
| `vehicle-ui/` | React companion front-end |
| `PROJECT_DOCUMENTATION_DETAILED.md` | In-depth project documentation |
