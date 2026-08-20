import streamlit as st
import pandas as pd
import numpy as np
import random
import datetime
import time
import math
import joblib
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model, Model

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="V2V NETWORK CONSOLE", layout="wide",
                   initial_sidebar_state="collapsed")

# ----------------------------------------------------
# CONSOLE THEME (NOC-style, compact, anti-flicker)
# ----------------------------------------------------
st.markdown("""
<style>
    /* ---- base console colours (fixed -> no dim/bright flicker) ---- */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stHeader"] { background-color: #060a12 !important; }
    /* neon gradient backdrop */
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse 900px 500px at 12% -8%, rgba(34,211,238,.10), transparent 60%),
            radial-gradient(ellipse 800px 500px at 88% -5%, rgba(167,139,250,.09), transparent 60%),
            radial-gradient(ellipse 1100px 700px at 50% 115%, rgba(14,116,144,.12), transparent 65%),
            linear-gradient(160deg, #0a1224 0%, #060a12 45%, #0b1020 100%) !important;
        background-attachment: fixed !important;
    }
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #0d1526 0%, #0a101c 100%) !important;
    }
    html, body, [class*="css"], .stApp {
        font-family: 'Consolas', 'Cascadia Mono', 'Segoe UI', monospace;
        color: #9fb3c8;
    }
    * { transition: none !important; animation: none !important; }
    [data-testid="stPlotlyChart"], [data-testid="stDataFrame"],
    .stPlotlyChart, .element-container { opacity: 1 !important; }

    /* ---- top status strip ---- */
    .console-strip {
        display: flex; flex-wrap: wrap; gap: 4px 22px; align-items: center;
        border: 1px solid #1b2942; border-left: 3px solid #22d3ee;
        background: linear-gradient(90deg, rgba(13,21,38,.95), rgba(10,16,28,.9));
        box-shadow: 0 0 20px rgba(34,211,238,.08);
        padding: 6px 14px; margin-bottom: 2px; border-radius: 6px;
        font-size: 0.82rem; letter-spacing: .4px; color: #7c93ad;
    }
    .console-strip b { color: #e2e8f0; font-size: 0.95rem; text-shadow: 0 0 8px rgba(226,232,240,.25); }
    .dot-run  { color: #22c55e; text-shadow: 0 0 10px rgba(34,197,94,.9); }
    .dot-stop { color: #ef4444; text-shadow: 0 0 10px rgba(239,68,68,.9); }
    .v-attack { color: #f87171; font-weight: 700; }
    .v-warn   { color: #fbbf24; }
    .v-ok     { color: #22c55e; }
    .v-crit   { color: #f43f5e; font-weight: 700; }

    /* ---- panels: neon-accented, thin borders ---- */
    .panel {
        border: 1px solid #1b2942; border-top: 2px solid #22d3ee;
        background: linear-gradient(165deg, rgba(16,26,46,.92) 0%, rgba(10,16,28,.88) 100%);
        box-shadow: 0 0 18px rgba(34,211,238,.06), inset 0 1px 0 rgba(148,163,184,.05);
        padding: 8px 10px; margin-bottom: 8px; border-radius: 6px;
    }
    .panel-head {
        font-size: .72rem; letter-spacing: 2px; color: #22d3ee;
        border-bottom: 1px solid #1b2942; padding-bottom: 4px; margin-bottom: 6px;
        font-weight: 700; text-transform: uppercase;
        text-shadow: 0 0 10px rgba(34,211,238,.45);
    }

    /* ---- live alerts feed ---- */
    .feed-item {
        font-size: .78rem; padding: 5px 8px; margin-bottom: 4px;
        border-left: 3px solid #334155; background: #0c1422; color: #9fb3c8;
    }
    .fi-attack   { border-color:#ef4444; background:linear-gradient(90deg, rgba(239,68,68,.16), rgba(28,13,18,.9)); color:#fca5a5; box-shadow:0 0 10px rgba(239,68,68,.18); }
    .fi-critical { border-color:#f43f5e; background:linear-gradient(90deg, rgba(244,63,94,.16), rgba(28,13,18,.9)); color:#fda4af; font-weight:600; box-shadow:0 0 10px rgba(244,63,94,.2); }
    .fi-speed    { border-color:#facc15; background:linear-gradient(90deg, rgba(250,204,21,.13), rgba(26,21,5,.9)); color:#fde68a; }
    .fi-distance { border-color:#a78bfa; background:linear-gradient(90deg, rgba(167,139,250,.13), rgba(21,16,37,.9)); color:#c4b5fd; }
    .fi-join     { border-color:#22c55e; background:linear-gradient(90deg, rgba(34,197,94,.13), rgba(8,24,14,.9)); color:#86efac; }
    .fi-leave    { border-color:#f472b6; background:linear-gradient(90deg, rgba(244,114,182,.13), rgba(26,13,22,.9)); color:#f9a8d4; }
    .fi-safe     { border-color:#334155; background:#0c1422; color:#64748b; }
    .fi-info     { border-color:#334155; background:#0c1422; color:#94a3b8; }
    .feed-item:hover { filter: brightness(1.25); border-left-width: 5px; }
    .feed-item .t { color:#475569; font-size:.68rem; }

    /* ---- connection list ---- */
    .conn-row {
        display:flex; justify-content:space-between; font-size:.78rem;
        padding: 3px 6px; border-bottom: 1px dashed #16202e; color:#9fb3c8;
    }
    .conn-close  { color:#fbbf24; }
    .conn-crit   { color:#f43f5e; }

    /* ---- vehicle inspector ---- */
    .insp { font-size:.8rem; line-height:1.7; }
    .insp .k { color:#64748b; display:inline-block; min-width:110px; }

    /* ---- compact widgets / buttons ---- */
    .stButton > button {
        font-size: .78rem; padding: 4px 8px; border-radius: 4px;
        font-family: 'Consolas', monospace; letter-spacing: .5px;
        border: 1px solid #1b2942;
        background: linear-gradient(160deg, #14213a, #0d1526);
        color: #cbd5e1;
    }
    .stButton > button:hover {
        border-color: #22d3ee; color: #22d3ee;
        box-shadow: 0 0 12px rgba(34,211,238,.35);
    }
    .stButton > button[kind="primary"] {
        border-color: #22c55e; color: #4ade80;
        box-shadow: 0 0 12px rgba(34,197,94,.25);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background-color: #0c1422; border: 1px solid #16202e; border-bottom: none;
        border-radius: 4px 4px 0 0; padding: 5px 16px;
        font-size: .75rem; letter-spacing: 1px; font-weight: 700;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0a101c !important; color: #22d3ee !important;
        border-top: 2px solid #22d3ee;
    }
    [data-testid="stDataFrame"] { border: 1px solid #16202e; }
    h1, h2, h3 { color: #cbd5e1 !important; }
    hr { border-color: #16202e; }
    .stApp .block-container { padding-top: 0.6rem; padding-bottom: 0.5rem; max-width: 100%; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# VEHICLE TYPE MAP
# ----------------------------------------------------
vehicle_types = {0: "Car", 1: "Bus", 2: "Ambulance", 3: "Truck"}
weather_types = {0: "Clear", 1: "Rain", 2: "Fog", 3: "Snow"}

# per-type visual identity on the tactical map
TYPE_STYLE = {
    0: {"symbol": "circle",      "color": "#38bdf8", "code": "CAR"},
    1: {"symbol": "square",      "color": "#f59e0b", "code": "BUS"},
    2: {"symbol": "diamond",     "color": "#f43f5e", "code": "AMB"},
    3: {"symbol": "triangle-up", "color": "#a78bfa", "code": "TRK"},
}

# ----------------------------------------------------
# SIMULATION AREA CONSTANTS
# ----------------------------------------------------
SIM_W = 1000.0   # simulation area width  (meters)
SIM_H = 1000.0   # simulation area height (meters)

# ----------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------
rf_model = joblib.load("hybrid_rf_model.pkl")
scaler = joblib.load("hybrid_scaler.pkl")

lstm_model = load_model("hybrid_lstm_model.keras", compile=False)

feature_extractor = Model(
    inputs=lstm_model.input,
    outputs=lstm_model.layers[-2].output
)

# ----------------------------------------------------
# SESSION STATE
# ----------------------------------------------------
_state_defaults = {
    "running": False,          # network on/off
    "vehicles": {},            # vid -> vehicle dict
    "log": [],                 # V2V message log (IDS pipeline output)
    "events": [],              # join / leave feed
    "alerts": [],              # speed / distance / collision / attack alerts
    "next_auto_id": 1000,      # auto vehicles use IDs >= 1000
    "connections": [],         # [(vid_a, vid_b, dist)]
    "weather_val": random.randint(0, 3),
    "joined_count": 0,
    "left_count": 0,
    "seen_ids": set(),         # every vehicle ID ever seen (for filters)
    "last_status": {},         # vid -> "SAFE ✅" / "ATTACK 🚨"
    "alert_ts": {},            # alert-signature -> last log time (dedup)
}
for _k, _v in _state_defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# simulation settings (persisted; sliders bind to these keys)
for _k, _v in {"comm_range": 300, "max_vehicles": 12, "speed_alert_threshold": 100,
               "distance_alert_threshold": 60, "critical_distance_threshold": 30}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------
def now_str():
    return datetime.datetime.now().strftime("%H:%M:%S")

def log_alert(msg, kind, sig=None, cooldown=8):
    """Append an alert to the live feed (with simple de-dup cooldown)."""
    key = sig if sig is not None else msg
    ts = time.time()
    if key in st.session_state.alert_ts and ts - st.session_state.alert_ts[key] < cooldown:
        return
    st.session_state.alert_ts[key] = ts
    st.session_state.alerts.append({"time": now_str(), "msg": msg, "kind": kind})
    if len(st.session_state.alerts) > 300:
        st.session_state.alerts = st.session_state.alerts[-300:]

def log_event(msg, kind):
    st.session_state.events.append({"time": now_str(), "event": msg, "kind": kind})
    if len(st.session_state.events) > 200:
        st.session_state.events = st.session_state.events[-200:]

def spawn_vehicle(vid, vtype, manual=False):
    st.session_state.vehicles[vid] = {
        "type": vtype,
        "x": random.uniform(0, SIM_W),
        "y": random.uniform(0, SIM_H),
        "speed": random.uniform(30, 100),
        "heading": random.uniform(0, 2 * math.pi),
        "accel": random.uniform(-1, 1),
        "brake": 0,
        "manual": manual,
    }
    st.session_state.seen_ids.add(vid)
    return vid

def spawn_auto_vehicle():
    vid = st.session_state.next_auto_id
    st.session_state.next_auto_id += 1
    vtype = random.choices([0, 1, 2, 3], weights=[60, 15, 10, 15])[0]
    spawn_vehicle(vid, vtype, manual=False)
    st.session_state.joined_count += 1
    log_event(f"Vehicle {vid} ({vehicle_types[vtype]}) joined the network", "join")
    log_alert(f"Vehicle {vid} ({vehicle_types[vtype]}) joined the network", "join")

def remove_auto_vehicle():
    autos = [v for v in st.session_state.vehicles if v >= 1000]
    if not autos:
        return
    vid = random.choice(autos)
    vtype = vehicle_types[st.session_state.vehicles[vid]["type"]]
    del st.session_state.vehicles[vid]
    st.session_state.left_count += 1
    st.session_state.last_status.pop(vid, None)
    log_event(f"Vehicle {vid} ({vtype}) left the network", "leave")
    log_alert(f"Vehicle {vid} ({vtype}) left the network", "leave")

def add_manual_vehicle(vtype):
    used = set(st.session_state.vehicles.keys())
    vid = None
    for i in range(1, 21):
        if i not in used:
            vid = i
            break
    if vid is None:
        return None
    spawn_vehicle(vid, vtype, manual=True)
    st.session_state.joined_count += 1
    log_event(f"Manual vehicle {vid} ({vehicle_types[vtype]}) added", "join")
    return vid

def remove_manual_vehicle():
    manuals = sorted(v for v in st.session_state.vehicles if v < 1000)
    if not manuals:
        return None
    vid = manuals[-1]
    vtype = vehicle_types[st.session_state.vehicles[vid]["type"]]
    del st.session_state.vehicles[vid]
    st.session_state.left_count += 1
    st.session_state.last_status.pop(vid, None)
    log_event(f"Manual vehicle {vid} ({vtype}) removed", "leave")
    return vid

# --- control panel callbacks (run before rerun) ---
def cb_start():
    st.session_state.running = True
    log_event("Network started — V2V communication active", "start")

def cb_stop():
    st.session_state.running = False
    log_event("Network stopped by operator", "stop")

def cb_add(vtype):
    add_manual_vehicle(vtype)

def cb_remove():
    remove_manual_vehicle()

# ----------------------------------------------------
# SIMULATION ENGINE (one tick per dashboard refresh)
# ----------------------------------------------------
def simulate_tick():
    comm_range = st.session_state.comm_range
    max_vehicles = st.session_state.max_vehicles
    speed_alert_threshold = st.session_state.speed_alert_threshold
    distance_alert_threshold = st.session_state.distance_alert_threshold
    critical_distance_threshold = st.session_state.critical_distance_threshold
    weather_val = st.session_state.weather_val

    if st.session_state.running:
        # --- 1. occasionally change weather ---
        if random.random() < 0.02:
            st.session_state.weather_val = random.randint(0, 3)
            weather_val = st.session_state.weather_val

        # --- 2. Automatic vehicle join / leave ---
        if len(st.session_state.vehicles) < max_vehicles and random.random() < 0.18:
            spawn_auto_vehicle()
        if len(st.session_state.vehicles) > 2 and random.random() < 0.10:
            remove_auto_vehicle()

        # --- 3. Kinematics update (persistent motion) ---
        for vid, veh in st.session_state.vehicles.items():
            if random.random() < 0.30:
                veh["accel"] = random.uniform(-1.5, 1.5)
            if random.random() < 0.12:
                veh["brake"] = random.randint(0, 1)
            if veh["brake"]:
                veh["speed"] = max(0, veh["speed"] - random.uniform(3, 8))
            else:
                veh["speed"] = max(0, min(130, veh["speed"] + veh["accel"]))
            veh["heading"] += random.uniform(-0.12, 0.12)
            dist = veh["speed"] / 3.6 * 1.0          # ~1 s per tick
            veh["x"] = (veh["x"] + math.cos(veh["heading"]) * dist) % SIM_W
            veh["y"] = (veh["y"] + math.sin(veh["heading"]) * dist) % SIM_H

        # --- 4. Range-based V2V discovery ---
        vehicles = st.session_state.vehicles
        vids = list(vehicles.keys())
        connections = []
        for i in range(len(vids)):
            for j in range(i + 1, len(vids)):
                a, b = vids[i], vids[j]
                dx = vehicles[a]["x"] - vehicles[b]["x"]
                dy = vehicles[a]["y"] - vehicles[b]["y"]
                dist_ab = math.hypot(dx, dy)
                if dist_ab <= comm_range:
                    connections.append((a, b, dist_ab))
        st.session_state.connections = connections

        # --- 5. Automatic V2V messaging per in-range pair (IDS pipeline) ---
        for sender, receiver, dist_ab in connections:
            sender_type = vehicles[sender]["type"]

            data = {
                "time": now_str(),
                "sender": sender,
                "receiver": receiver,
                "vehicle_type": vehicle_types[sender_type],
                "priority": random.randint(0, 2),
                "weather": weather_types[weather_val],
                "pos_x": round(vehicles[sender]["x"], 1),
                "pos_y": round(vehicles[sender]["y"], 1),
                "speed": round(vehicles[sender]["speed"], 1),
                "brake": vehicles[sender]["brake"],
                "acc": vehicles[sender]["accel"],
                "delay": round(random.uniform(0, 2), 2),
                "msg_rate": round(random.uniform(1, 10), 2),
                "duplicate": random.randint(0, 1),
                "distance": dist_ab
            }

            # 🔥 IDS: LSTM + Random Forest (unchanged feature pipeline)
            features = np.array([[sender_type, data["priority"], data["pos_x"], data["pos_y"],
                                  weather_val, data["speed"], data["brake"], data["acc"],
                                  data["delay"], data["msg_rate"], data["duplicate"]]])

            scaled = scaler.transform(features)
            lstm_input = scaled.reshape((1, 1, scaled.shape[1]))
            lstm_features = feature_extractor.predict(lstm_input, verbose=0)

            combined = np.hstack([scaled, lstm_features])
            prediction = rf_model.predict(combined)

            data["status"] = "ATTACK 🚨" if prediction[0] == 1 else "SAFE ✅"

            st.session_state.log.append(data)
            st.session_state.last_status[sender] = data["status"]

            # IDS attacks also feed the Live Alerts panel
            if prediction[0] == 1:
                log_alert(f"🚨 IDS ATTACK: Vehicle {sender} → Vehicle {receiver} flagged as ATTACK",
                          "attack", sig=f"atk{sender}", cooldown=12)

            if len(st.session_state.log) > 500:
                st.session_state.log = st.session_state.log[-500:]

        # --- 6. Automatic speed / distance / collision-risk safety alerts ---
        for vid, veh in st.session_state.vehicles.items():
            if veh["speed"] > speed_alert_threshold:
                log_alert(f"⚠️ High Speed: Vehicle {vid} travelling at {veh['speed']:.0f} km/h",
                          "speed", sig=f"spd{vid}", cooldown=10)

        for vid_a, vid_b, dist_ab in connections:
            high_speed_pair = (st.session_state.vehicles[vid_a]["speed"] > speed_alert_threshold or
                               st.session_state.vehicles[vid_b]["speed"] > speed_alert_threshold)
            if dist_ab < critical_distance_threshold and high_speed_pair:
                log_alert(f"🔴 CRITICAL COLLISION RISK: Vehicle {vid_a} & Vehicle {vid_b} — "
                          f"{dist_ab:.0f} m apart at high speed",
                          "critical", sig=f"crit{vid_a}_{vid_b}", cooldown=10)
            elif dist_ab < distance_alert_threshold:
                log_alert(f"⚠️ Small Distance: Vehicle {vid_a} & Vehicle {vid_b} only {dist_ab:.0f} m apart",
                          "distance", sig=f"dst{vid_a}_{vid_b}", cooldown=10)

    else:
        st.session_state.connections = []

# ----------------------------------------------------
# TACTICAL MAP RENDERER
# ----------------------------------------------------
def render_network_map():
    vehicles = st.session_state.vehicles
    connections = st.session_state.connections
    comm_range = st.session_state.comm_range
    speed_thr = st.session_state.speed_alert_threshold
    dist_thr = st.session_state.distance_alert_threshold
    crit_thr = st.session_state.critical_distance_threshold

    # status sets for highlighting
    attacked = {v for v, s in st.session_state.last_status.items()
                if s == "ATTACK 🚨" and v in vehicles}
    speeding = {v for v, veh in vehicles.items() if veh["speed"] > speed_thr}
    too_close = set()
    for a, b, d in connections:
        if d < dist_thr:
            too_close.update((a, b))

    fig = go.Figure()

    # communication-range circles (faint)
    for vid, veh in vehicles.items():
        fig.add_shape(type="circle", xref="x", yref="y",
                      x0=veh["x"] - comm_range, y0=veh["y"] - comm_range,
                      x1=veh["x"] + comm_range, y1=veh["y"] + comm_range,
                      line=dict(width=0),
                      fillcolor="rgba(56,189,248,0.045)")

    # connection lines (colour = severity)
    for a, b, d in connections:
        if d < crit_thr:
            colour, width = "#f43f5e", 2.2
        elif d < dist_thr:
            colour, width = "#fbbf24", 1.4
        else:
            colour, width = "rgba(34,211,238,0.35)", 1
        fig.add_trace(go.Scatter(
            x=[vehicles[a]["x"], vehicles[b]["x"]],
            y=[vehicles[a]["y"], vehicles[b]["y"]],
            mode="lines", line=dict(color=colour, width=width),
            hoverinfo="text",
            hovertext=f"{a} ↔ {b} — {d:.0f} m",
            showlegend=False))

    # vehicle nodes — one trace per type (legend = symbol key)
    for t, style in TYPE_STYLE.items():
        vids = [v for v, veh in vehicles.items() if veh["type"] == t]
        if not vids:
            continue
        fig.add_trace(go.Scatter(
            x=[vehicles[v]["x"] for v in vids],
            y=[vehicles[v]["y"] for v in vids],
            mode="markers+text",
            marker=dict(symbol=style["symbol"], size=13,
                        color=["#ef4444" if v in attacked else
                               "#fbbf24" if v in too_close else
                               "#fde047" if v in speeding else style["color"]
                               for v in vids],
                        line=dict(color="#060a12", width=1)),
            text=[str(v) for v in vids],
            textposition="top center",
            textfont=dict(size=9, color="#cbd5e1"),
            name=f"{style['code']} ({vehicle_types[t]})",
            customdata=[[v, vehicle_types[t], vehicles[v]["speed"],
                         st.session_state.last_status.get(v, "—")]
                        for v in vids],
            hovertemplate=("<b>Vehicle %{customdata[0]}</b> — %{customdata[1]}<br>"
                           "Speed: %{customdata[2]:.0f} km/h<br>"
                           "IDS: %{customdata[3]}<br>"
                           "Position: (%{x:.0f}, %{y:.0f}) m"
                           "<extra></extra>")))

    # attacked vehicles: red alarm halo ring
    if attacked:
        fig.add_trace(go.Scatter(
            x=[vehicles[v]["x"] for v in attacked],
            y=[vehicles[v]["y"] for v in attacked],
            mode="markers",
            marker=dict(symbol="circle-open", size=34, color="#ef4444",
                        line=dict(width=3)),
            name="⚠ ATTACKED", hoverinfo="skip"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a101c", plot_bgcolor="#060a12",
        margin=dict(l=10, r=10, t=24, b=6),
        height=640,
        xaxis=dict(range=[-60, SIM_W + 60], showgrid=True, zeroline=False,
                   gridcolor="rgba(30,41,59,.4)", tickfont=dict(size=9),
                   title=dict(text="X (m)", font=dict(size=9))),
        yaxis=dict(range=[-60, SIM_H + 60], showgrid=True, zeroline=False,
                   gridcolor="rgba(30,41,59,.4)", tickfont=dict(size=9),
                   scaleanchor="x", scaleratio=1,
                   title=dict(text="Y (m)", font=dict(size=9))),
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=9),
                    bgcolor="rgba(0,0,0,0)"),
        modebar=dict(orientation="v", bgcolor="rgba(0,0,0,0)"),
        dragmode="pan",
    )
    return fig

# ----------------------------------------------------
# STATUS STRIP / FEED RENDERERS
# ----------------------------------------------------
ALERT_ICON = {"attack": "🚨", "critical": "🔴", "speed": "⚠️", "distance": "⚠️",
              "join": "🟢", "leave": "🔴", "safe": "🟢", "info": "•"}

def render_status_strip():
    running = st.session_state.running
    attack_alerts = sum(1 for a in st.session_state.alerts if a["kind"] == "attack")
    safety_alerts = len(st.session_state.alerts) - attack_alerts
    dot = '<span class="dot-run">● RUNNING</span>' if running else '<span class="dot-stop">● STOPPED</span>'
    st.markdown(
        f'<div class="console-strip">'
        f'<span><b>V2V&nbsp;NETWORK</b>&nbsp;{dot}</span>'
        f'<span>ACTIVE&nbsp;VEHICLES <b>{len(st.session_state.vehicles)}</b></span>'
        f'<span>CONNECTIONS <b>{len(st.session_state.connections)}</b></span>'
        f'<span>IDS&nbsp;ATTACKS <b class="v-attack">{attack_alerts}</b></span>'
        f'<span>SAFETY&nbsp;ALERTS <b class="v-warn">{safety_alerts}</b></span>'
        f'<span>JOINED <b class="v-ok">{st.session_state.joined_count}</b></span>'
        f'<span>LEFT <b>{st.session_state.left_count}</b></span>'
        f'<span>WEATHER <b>{weather_types[st.session_state.weather_val].upper()}</b></span>'
        f'<span>MSG&nbsp;LOG <b>{len(st.session_state.log)}</b></span>'
        f'<span style="margin-left:auto;color:#475569">{now_str()}</span>'
        f'</div>', unsafe_allow_html=True)

def render_alert_feed(n=16):
    kinds = ["All", "🚨 Attacks only", "🔴 Critical", "⚠️ Speed", "⚠️ Distance",
             "🟢 Join/Leave"]
    kind_f = st.radio("FILTER", kinds, index=0, horizontal=True,
                      label_visibility="collapsed", key="alert_filter")
    def _match(a):
        if kind_f == "All":
            return True
        if kind_f.endswith("Attacks only"):
            return a["kind"] == "attack"
        if kind_f.startswith("🔴"):
            return a["kind"] == "critical"
        if kind_f.endswith("Speed"):
            return a["kind"] == "speed"
        if kind_f.endswith("Distance"):
            return a["kind"] == "distance"
        return a["kind"] in ("join", "leave")
    pool = [a for a in st.session_state.alerts if _match(a)]
    alerts = pool[-n:][::-1]     # newest first
    if not alerts:
        st.markdown('<div class="feed-item fi-info">— no alerts — awaiting network activity —</div>',
                    unsafe_allow_html=True)
        return
    for a in alerts:
        cls = a["kind"] if a["kind"] in ("attack", "critical", "speed", "distance",
                                         "join", "leave", "safe") else "info"
        st.markdown(
            f'<div class="feed-item fi-{cls}">{ALERT_ICON.get(a["kind"], "•")} '
            f'<span class="t">{a["time"]}</span>&nbsp; {a["msg"]}</div>',
            unsafe_allow_html=True)

def render_connection_list():
    if st.session_state.connections:
        sort_mode = st.radio("SORT", ["Distance", "ID"], index=0, horizontal=True,
                             label_visibility="collapsed", key="conn_sort")
        conns = sorted(st.session_state.connections,
                       key=lambda c: c[2] if sort_mode == "Distance" else (c[0], c[1]))
    else:
        conns = []
    if not conns:
        st.markdown('<div class="feed-item fi-info">— no active connections —</div>',
                    unsafe_allow_html=True)
        return
    rows = []
    for a, b, d in conns[:18]:
        if d < st.session_state.critical_distance_threshold:
            cls = "conn-crit"
        elif d < st.session_state.distance_alert_threshold:
            cls = "conn-close"
        else:
            cls = ""
        rows.append(f'<div class="conn-row {cls}"><span>{a} ↔ {b}</span>'
                    f'<span>{d:.0f} m</span></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)

def compass(heading):
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(((math.degrees(heading) % 360) + 22.5) // 45) % 8
    return f"{math.degrees(heading) % 360:.0f}° {dirs[idx]}"

def render_inspector(vid):
    vehicles = st.session_state.vehicles
    if vid not in vehicles:
        st.markdown('<div class="feed-item fi-info">— select a vehicle to inspect —</div>',
                    unsafe_allow_html=True)
        return
    veh = vehicles[vid]
    vtype = vehicle_types[veh["type"]]

    # neighbours + nearest
    neigh = sorted(
        [(b if a == vid else a, d) for a, b, d in st.session_state.connections
         if vid in (a, b)], key=lambda t: t[1])
    if neigh:
        nearest_id, nearest_d = neigh[0]
        neigh_txt = ", ".join(f"{n} ({d:.0f}m)" for n, d in neigh[:6])
    else:
        nearest_id, nearest_d = "—", 0
        neigh_txt = "none in range"

    ids_status = st.session_state.last_status.get(vid, "—")
    safety = []
    if veh["speed"] > st.session_state.speed_alert_threshold:
        safety.append('<span class="v-warn">HIGH SPEED</span>')
    if any((vid in (a, b)) and d < st.session_state.distance_alert_threshold
           for a, b, d in st.session_state.connections):
        safety.append('<span class="v-crit">PROXIMITY WARNING</span>')
    if any((vid in (a, b)) and d < st.session_state.critical_distance_threshold
           for a, b, d in st.session_state.connections):
        safety.append('<span class="v-crit">COLLISION RISK</span>')
    safety_txt = " &nbsp;".join(safety) if safety else '<span class="v-ok">NOMINAL</span>'
    ids_html = ('<span class="v-attack">ATTACK 🚨</span>' if ids_status == "ATTACK 🚨"
                else f'<span class="v-ok">{ids_status}</span>' if ids_status != "—" else "—")

    st.markdown(
        f'<div class="insp">'
        f'<span class="k">VEHICLE ID</span><b>#{vid}</b> ({TYPE_STYLE[veh["type"]]["code"]})'
        f'{" · MANUAL" if veh.get("manual") else " · AUTO"}<br>'
        f'<span class="k">TYPE</span>{vtype}<br>'
        f'<span class="k">POSITION</span>x={veh["x"]:.0f} m, y={veh["y"]:.0f} m<br>'
        f'<span class="k">DIRECTION</span>{compass(veh["heading"])}<br>'
        f'<span class="k">SPEED</span>{veh["speed"]:.0f} km/h'
        f'&nbsp;&nbsp;ACCEL {veh["accel"]:+.1f}&nbsp;&nbsp;BRAKE {"ON" if veh["brake"] else "off"}<br>'
        f'<span class="k">NEIGHBOURS</span>{neigh_txt}<br>'
        f'<span class="k">NEAREST VEHICLE</span>{nearest_id} @ {nearest_d:.0f} m<br>'
        f'<span class="k">IDS STATUS</span>{ids_html}<br>'
        f'<span class="k">SAFETY STATUS</span>{safety_txt}'
        f'</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# MAIN CONSOLE (fragment: only this region auto-refreshes)
# ----------------------------------------------------
def _run_dashboard():
    # ---- advance simulation BEFORE any rendering (state is always fresh) ----
    simulate_tick()

    # ---- TOP STATUS STRIP ----
    render_status_strip()

    # ---- MAIN CONTROL-ROOM GRID ----
    left, center, right = st.columns([2.0, 6.2, 2.5], gap="small")

    # ================= LEFT: NETWORK CONTROLS =================
    with left:
        st.markdown('<div class="panel-head">◤ NETWORK CONTROLS</div>', unsafe_allow_html=True)

        b1, b2 = st.columns(2)
        with b1:
            if st.button("▶ START", use_container_width=True, on_click=cb_start):
                pass
        with b2:
            if st.button("⏹ STOP", use_container_width=True, on_click=cb_stop):
                pass

        t_sel = st.selectbox("VEHICLE TYPE", options=list(vehicle_types.keys()),
                             format_func=lambda t: vehicle_types[t], key="add_type",
                             label_visibility="collapsed")
        b3, b4 = st.columns(2)
        with b3:
            if st.button("➕ ADD VEHICLE", use_container_width=True,
                         on_click=cb_add, args=(t_sel,)):
                pass
        with b4:
            if st.button("➖ REMOVE VEHICLE", use_container_width=True,
                         on_click=cb_remove):
                pass

        # ---- network status ----
        if st.session_state.running:
            st.markdown('<div class="feed-item fi-join"><span class="dot-run">●</span> '
                        '<b>NETWORK STATUS: RUNNING</b><br>V2V communication active</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="feed-item fi-info"><span class="dot-stop">●</span> '
                        '<b>NETWORK STATUS: STOPPED</b><br>Press START to activate</div>',
                        unsafe_allow_html=True)

        # ---- simulation settings (compact) ----
        st.markdown('<div class="panel-head" style="margin-top:8px">◤ SIMULATION SETTINGS</div>',
                    unsafe_allow_html=True)
        st.slider("COMM RANGE (m)", 50, 600, key="comm_range", step=10)
        st.slider("MAX VEHICLES", 2, 20, key="max_vehicles")
        st.slider("HIGH-SPEED THR (km/h)", 60, 130, key="speed_alert_threshold")
        st.slider("WARNING DIST (m)", 10, 200, key="distance_alert_threshold")
        st.slider("CRITICAL DIST (m)", 5, 100, key="critical_distance_threshold")

        # ---- vehicle inspector ----
        st.markdown('<div class="panel-head" style="margin-top:8px">◤ VEHICLE INSPECTOR</div>',
                    unsafe_allow_html=True)
        vid_list = sorted(st.session_state.vehicles.keys())
        if vid_list:
            sel = st.selectbox("INSPECT", options=vid_list,
                               format_func=lambda v: f"Vehicle {v} "
                                                     f"({vehicle_types[st.session_state.vehicles[v]['type']]})",
                               key="inspect_vid")
            render_inspector(sel)
        else:
            st.markdown('<div class="feed-item fi-info">— no vehicles in network —</div>',
                        unsafe_allow_html=True)

    # ================= CENTER: LIVE V2V NETWORK MAP =================
    with center:
        st.markdown('<div class="panel-head">🗺 LIVE V2V NETWORK MAP'
                    '&nbsp;&nbsp;<span style="color:#475569">1 km × 1 km operational area</span></div>',
                    unsafe_allow_html=True)
        st.plotly_chart(render_network_map(), use_container_width=True, key="map_main",
                        config={"displaylogo": False, "scrollZoom": True})

    # ================= RIGHT: LIVE MONITORING =================
    with right:
        st.markdown('<div class="panel-head" style="color:#f87171">🚨 LIVE ALERTS</div>',
                    unsafe_allow_html=True)
        render_alert_feed()

        st.markdown('<div class="panel-head" style="margin-top:8px">📡 ACTIVE CONNECTIONS</div>',
                    unsafe_allow_html=True)
        render_connection_list()

    # ================= BOTTOM: OPERATIONS TABS =================
    st.markdown("---")

    # compact filter row
    fcol1, fcol2, fcol3 = st.columns([2.2, 2.2, 2.2])
    with fcol1:
        filter_ids = st.multiselect("VEHICLE ID", options=sorted(st.session_state.seen_ids),
                                    default=None, key="filter_ids")
    with fcol2:
        filter_types = st.multiselect("VEHICLE TYPE", options=list(vehicle_types.values()),
                                      key="filter_types")
    with fcol3:
        filter_status = st.selectbox("IDS STATUS", options=["All", "SAFE ✅", "ATTACK 🚨"],
                                     key="filter_status")

    tab_traffic, tab_ids, tab_safety, tab_veh, tab_events, tab_analytics = st.tabs(
        ["LIVE TRAFFIC", "IDS", "SAFETY", "VEHICLES", "NETWORK EVENTS", "ANALYTICS"])

    df = pd.DataFrame(st.session_state.log) if st.session_state.log else pd.DataFrame()

    def filtered(d):
        if d.empty:
            return d
        f = d
        if filter_ids:
            f = f[f["sender"].isin(filter_ids) | f["receiver"].isin(filter_ids)]
        if filter_types:
            f = f[f["vehicle_type"].isin(filter_types)]
        if filter_status != "All":
            f = f[f["status"] == filter_status]
        return f

    # ---- LIVE TRAFFIC ----
    with tab_traffic:
        st.markdown('<div class="panel-head">◤ LIVE V2V COMMUNICATION TRAFFIC</div>',
                    unsafe_allow_html=True)
        if df.empty:
            st.markdown('<div class="feed-item fi-info">— no traffic — network stopped or no vehicles in range —</div>',
                        unsafe_allow_html=True)
        else:
            show = filtered(df).tail(120).iloc[::-1]
            st.dataframe(show, use_container_width=True, height=300, hide_index=True)

    # ---- IDS ----
    with tab_ids:
        st.markdown('<div class="panel-head">◤ INTRUSION DETECTION SYSTEM</div>',
                    unsafe_allow_html=True)
        if df.empty:
            st.markdown('<div class="feed-item fi-info">— no classified messages yet —</div>',
                        unsafe_allow_html=True)
        else:
            fdf = filtered(df)
            n_safe = int((fdf["status"] == "SAFE ✅").sum())
            n_att = int((fdf["status"] == "ATTACK 🚨").sum())
            st.markdown(
                f'<div class="console-strip">'
                f'<span>SAFE MESSAGES <b class="v-ok">{n_safe}</b></span>'
                f'<span>ATTACK MESSAGES <b class="v-attack">{n_att}</b></span>'
                f'<span>TOTAL <b>{len(fdf)}</b></span>'
                f'<span>ATTACK RATE <b class="v-attack">'
                f'{(100 * n_att / max(len(fdf), 1)):.1f}%</b></span></div>',
                unsafe_allow_html=True)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.plotly_chart(px.pie(names=["SAFE ✅", "ATTACK 🚨"],
                                       values=[n_safe, max(n_att, 0.0001)], hole=0.6,
                                       color_discrete_sequence=["#22c55e", "#ef4444"],
                                       template="plotly_dark"),
                                use_container_width=True, key="ids_pie")
            with c2:
                st.plotly_chart(px.bar(fdf["status"].value_counts(),
                                       color_discrete_sequence=["#22d3ee"],
                                       template="plotly_dark"),
                                use_container_width=True, key="ids_bar")
            st.markdown('<div class="panel-head" style="margin-top:6px">◤ FLAGGED ATTACK MESSAGES</div>',
                        unsafe_allow_html=True)
            atk = fdf[fdf["status"] == "ATTACK 🚨"].tail(40).iloc[::-1]
            if atk.empty:
                st.markdown('<div class="feed-item fi-safe">No attacks detected in current selection.</div>',
                            unsafe_allow_html=True)
            else:
                st.dataframe(atk, use_container_width=True, height=220, hide_index=True)

    # ---- SAFETY ----
    with tab_safety:
        st.markdown('<div class="panel-head">◤ ROAD SAFETY MONITOR</div>', unsafe_allow_html=True)
        vehicles = st.session_state.vehicles
        if not vehicles:
            st.markdown('<div class="feed-item fi-info">— no vehicles to monitor —</div>',
                        unsafe_allow_html=True)
        else:
            speed_thr = st.session_state.speed_alert_threshold
            warn_d = st.session_state.distance_alert_threshold
            crit_d = st.session_state.critical_distance_threshold
            fast = [v for v, veh in vehicles.items() if veh["speed"] > speed_thr]
            close = [(a, b, d) for a, b, d in st.session_state.connections if d < warn_d]
            crit = [(a, b, d) for a, b, d in st.session_state.connections if d < crit_d]
            st.markdown(
                f'<div class="console-strip">'
                f'<span>HIGH-SPEED VEHICLES <b class="v-warn">{len(fast)}</b>'
                f'{" — " + ", ".join(map(str, fast[:8])) if fast else ""}</span>'
                f'<span>PROXIMITY PAIRS <b class="v-warn">{len(close)}</b></span>'
                f'<span>COLLISION-RISK PAIRS <b class="v-crit">{len(crit)}</b></span></div>',
                unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            with s1:
                st.plotly_chart(px.bar(x=[f"V{v}" for v in vehicles],
                                       y=[vehicles[v]["speed"] for v in vehicles],
                                       labels={"x": "vehicle", "y": "km/h"},
                                       title="Current Speeds",
                                       color_discrete_sequence=["#22d3ee"],
                                       template="plotly_dark"),
                                use_container_width=True, key="safety_speeds")
            with s2:
                if not df.empty:
                    fdf = filtered(df)
                    st.plotly_chart(px.line(fdf, x="time", y="distance", markers=True,
                                            title="Distance Between Communicating Vehicles",
                                            color_discrete_sequence=["#a78bfa"],
                                            template="plotly_dark"),
                                    use_container_width=True, key="safety_dist")
                else:
                    st.markdown('<div class="feed-item fi-info">— no distance data yet —</div>',
                                unsafe_allow_html=True)
            if close:
                st.markdown('<div class="panel-head">◤ PROXIMITY / COLLISION RISK</div>',
                            unsafe_allow_html=True)
                for a, b, d in sorted(close, key=lambda c: c[2]):
                    if d < crit_d:
                        cls, tag = "fi-critical", "🔴 COLLISION RISK"
                    else:
                        cls, tag = "fi-distance", "⚠ DISTANCE WARNING"
                    st.markdown(f'<div class="feed-item {cls}">{tag} — '
                                f'Vehicle {a} ↔ Vehicle {b} — {d:.0f} m</div>',
                                unsafe_allow_html=True)

    # ---- VEHICLES ----
    with tab_veh:
        st.markdown('<div class="panel-head">◤ ACTIVE VEHICLE REGISTRY</div>', unsafe_allow_html=True)
        vehicles = st.session_state.vehicles
        if not vehicles:
            st.markdown('<div class="feed-item fi-info">— network empty —</div>',
                        unsafe_allow_html=True)
        else:
            rows = []
            for vid in sorted(vehicles.keys()):
                veh = vehicles[vid]
                neigh = [n for a, b, d in st.session_state.connections
                         for n in ((b,) if a == vid else (a,) if b == vid else ())]
                rows.append({
                    "ID": vid, "Type": vehicle_types[veh["type"]],
                    "Mode": "MANUAL" if veh.get("manual") else "AUTO",
                    "X (m)": round(veh["x"], 1), "Y (m)": round(veh["y"], 1),
                    "Heading": compass(veh["heading"]),
                    "Speed (km/h)": round(veh["speed"], 1),
                    "Accel": round(veh["accel"], 2), "Brake": veh["brake"],
                    "Neighbours": len(neigh),
                    "IDS": st.session_state.last_status.get(vid, "—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         height=280, hide_index=True)

    # ---- NETWORK EVENTS ----
    with tab_events:
        st.markdown('<div class="panel-head">◤ NETWORK EVENTS — JOIN / LEAVE LOG</div>',
                    unsafe_allow_html=True)
        evs = st.session_state.events
        if not evs:
            st.markdown('<div class="feed-item fi-info">— no events yet — press ▶ START —</div>',
                        unsafe_allow_html=True)
        else:
            for ev in reversed(evs[-40:]):
                icon = {"join": "🟢", "leave": "🔴", "start": "▶️", "stop": "⏹"}.get(ev["kind"], "•")
                cls = ev["kind"] if ev["kind"] in ("join", "leave") else "info"
                st.markdown(f'<div class="feed-item fi-{cls}">{icon} '
                            f'<span class="t">{ev["time"]}</span>&nbsp; {ev["event"]}</div>',
                            unsafe_allow_html=True)

    # ---- ANALYTICS ----
    with tab_analytics:
        st.markdown('<div class="panel-head">◤ NETWORK ANALYTICS</div>', unsafe_allow_html=True)
        if df.empty:
            st.markdown('<div class="feed-item fi-info">— no data yet — press ▶ START —</div>',
                        unsafe_allow_html=True)
        else:
            fdf = filtered(df)
            a1, a2 = st.columns(2)
            with a1:
                st.plotly_chart(px.line(fdf, x="time", y="speed", markers=True,
                                        title="Speed Over Time",
                                        color_discrete_sequence=["#22d3ee"],
                                        template="plotly_dark"),
                                use_container_width=True, key="an_speed")
                st.plotly_chart(px.bar(fdf["status"].value_counts(), title="IDS Verdicts",
                                       color_discrete_sequence=["#22d3ee"],
                                       template="plotly_dark"),
                                use_container_width=True, key="an_verdicts")
            with a2:
                st.plotly_chart(px.pie(fdf["weather"].value_counts(), title="Weather Distribution",
                                       hole=0.4,
                                       color_discrete_sequence=["#22d3ee", "#a78bfa",
                                                                "#fbbf24", "#f43f5e"],
                                       template="plotly_dark"),
                                use_container_width=True, key="an_weather")
                if "distance" in fdf.columns:
                    st.plotly_chart(px.line(fdf, x="time", y="distance", markers=True,
                                            title="Distance Between Communicating Vehicles",
                                            color_discrete_sequence=["#a78bfa"],
                                            template="plotly_dark"),
                                    use_container_width=True, key="an_dist")
            st.markdown('<div class="panel-head">◤ FULL MESSAGE LOG</div>', unsafe_allow_html=True)
            st.dataframe(fdf, use_container_width=True, height=280, hide_index=True)

# ---- run the console ----
if hasattr(st, "fragment"):
    # st.fragment: only the console region refreshes every second -> zero flicker
    _run_dashboard_frag = st.fragment(run_every=1)(_run_dashboard)
    _run_dashboard_frag()
else:
    # fallback for older Streamlit versions
    _run_dashboard()
    if st.session_state.running:
        time.sleep(1)
        st.rerun()
