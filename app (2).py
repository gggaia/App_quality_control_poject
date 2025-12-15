import streamlit as st
import pandas as pd
import numpy as np
import json
from catboost import CatBoostClassifier
import os

st.set_page_config(
    page_title="Late Order Prediction",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 1. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_late_model.cbm")
    return model

model = load_model()

import joblib

@st.cache_resource
def load_group_stats():
    return joblib.load("pre_group_stats.pkl")

group_stats = load_group_stats()

X_val = pd.read_csv("X_val.csv")   # or your validation DataFrame


# Keep only 50 rows
X50 = X_val.head(50)

# ----------------------------------------
# RUN PREDICTIONS
# ----------------------------------------
probs = model.predict_proba(X50)[:,1]
preds = (probs >= 0.5).astype(int)

# ----------------------------------------
# BUILD LOG TABLE
# ----------------------------------------


LOG_FILE = "prediction_log.csv"

# Only create the initial 50-row file ONCE
if not os.path.exists(LOG_FILE):
    # Keep only 50 rows
    X50 = X_val.head(50)

    probs = model.predict_proba(X50)[:,1]
    preds = (probs >= 0.5).astype(int)

    log_df = pd.DataFrame({
        "origin_port": X50["origin_port"],
        "logistic_hub": X50["logistic_hub"],
        "customer": X50["customer"],
        "threepl": X50["3pl"],
        "customs_procedures": X50["customs_procedures"],
        "material_handling": X50["material_handling"],
        "units": X50["units"],
        "weight": X50["weight"],
        "model_probability": probs,
        "model_prediction": preds
    })

    log_df.to_csv(LOG_FILE, index=False)


def save_submission(entry_dict):
    """Append a new prediction to the log file."""
    new_row = pd.DataFrame([entry_dict])

    if not os.path.exists(LOG_FILE):
        # if the file does not exist, create it with the new row
        new_row.to_csv(LOG_FILE, index=False)
    else:
        # if it exists, append
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(LOG_FILE, index=False)


# ============================================================
# 2. LOAD DISTANCES AS DICTIONARY
# ============================================================

file_cities = os.path.join("./data", "cities_data.csv")

@st.cache_resource
def load_distances_df():
    df = pd.read_csv(file_cities, sep=";")
    df["city_from_name"] = df["city_from_name"].str.strip().str.title()
    df["city_to_name"] = df["city_to_name"].str.strip().str.title()
    return df

df_distances = load_distances_df()

distance_dict = {}
for _, row in df_distances.iterrows():
    c1 = row["city_from_name"]
    c2 = row["city_to_name"]
    d = float(row["distance"])
    distance_dict[(c1, c2)] = d
    distance_dict[(c2, c1)] = d

def get_distance(city_from, city_to):
    if not city_from or not city_to or city_to == "NoHub":
        return 0.0

    city_from = city_from.strip().title()
    city_to = city_to.strip().title()

    if (city_from, city_to) not in distance_dict:
        st.error(f"❌ No distance data for route {city_from} → {city_to}.")
        st.stop()

    return distance_dict[(city_from, city_to)]


# ============================================================
# 3. COUNTRY → LPI
# ============================================================

LPI_COUNTRY = {
    "Netherlands": 4.02, "Greece": 3.20, "Spain": 3.83, "France": 3.84,
    "Italy": 3.74, "Germany": 4.20, "Czechia": 3.68, "Portugal": 3.56,
    "Austria": 4.03, "Sweden": 4.05, "Hungary": 3.42, "Finland": 3.97,
    "Denmark": 3.99, "Romania": 3.12, "Slovakia": 3.03, "Poland": 3.54,
    "Belgium": 4.04
}

CITY_TO_COUNTRY = {
    "Rotterdam": "Netherlands", "Athens": "Greece", "Barcelona": "Spain",
    "Berlin": "Germany", "Milan": "Italy", "Madrid": "Spain",
    "Vienna": "Austria", "Paris": "France", "Munich": "Germany",
    "Amsterdam": "Netherlands", "Stockholm": "Sweden", "Copenhagen": "Denmark",
    "Lyon": "France", "Cologne": "Germany", "Rome": "Italy",
    "Marseille": "France", "Bucharest": "Romania", "Budapest": "Hungary",
    "Naples": "Italy", "Hanover": "Germany", "Malmö": "Sweden",
    "Turin": "Italy", "Lisbon": "Portugal", "Valencia": "Spain",
    "Prague": "Czechia", "Bordeaux": "France", "Bremen": "Germany",
    "Helsinki": "Finland", "Porto": "Portugal", "Venlo": "Netherlands",
    "Hamburg": "Germany", "Warsaw": "Poland", "Dusseldorf": "Germany",
    "Lille": "France", "Zaragoza": "Spain", "Liege": "Belgium",
    "Bratislava": "Slovakia"
}

VALID_HUBS = [
    "NoHub",
    "Venlo",
    "Hamburg",
    "Warsaw",
    "Dusseldorf",
    "Rome",
    "Lille",
    "Zaragoza",
    "Liege",
    "Bratislava"
]

VALID_CUSTOMERS = [
    "Berlin", "Milan", "Madrid", "Vienna", "Paris", "Munich", "Amsterdam",
    "Stockholm", "Copenhagen", "Lyon", "Barcelona", "Cologne", "Rome",
    "Marseille", "Bucharest", "Athens", "Budapest", "Naples", "Hanover",
    "Malmö", "Turin", "Lisbon", "Valencia", "Prague", "Bordeaux",
    "Bremen", "Helsinki", "Porto"
]

def get_country(city):
    return CITY_TO_COUNTRY.get(city, "Unknown")

def get_lpi(city):
    return LPI_COUNTRY.get(get_country(city), 0.0)


# ============================================================
# 4. WEIGHT RISK CATEGORY (uses stored dictionary)
# ============================================================

with open("weight_risk_dict.json") as f:
    weight_risk_dict = json.load(f)

def compute_weight_class(weight):
    if weight < 904:
        return "W1_Light"
    elif weight < 1250:
        return "W2_Medium"
    elif weight < 1641:
        return "W3_Heavy"
    else:
        return "W4_ExtraHeavy"

def compute_weight_risk(weight, material_handling):
    # convert 5 → "5.0", ensure consistent formatting
    material_handling = f"{float(material_handling):.1f}"
    weight_class = compute_weight_class(weight)
    key = f"{weight_class}|{material_handling}"
    return weight_risk_dict.get(key)

def compute_zscore(value, mean_series, std_series, key):
    mean = mean_series.get(key, None)
    std = std_series.get(key, None)

    if mean is None or std is None or std == 0 or np.isnan(std):
        return 0.0

    return (value - mean) / std




# ============================================================
# 5. FEATURE COLUMNS
# ============================================================

FEATURE_COLS = [
    'origin_port','3pl','customs_procedures','logistic_hub','customer',
    'units','weight','material_handling','dist_port_to_hub','distance',
    'total_weight','has_hub','route_geo','route_x_upl',
    'weight_risk_category','3pl_customs','origin_lpi','customer_lpi','hub_lpi',
    'origin_port_dist_port_to_hub_std','origin_port_TO_logistic_hub',
    'logistic_hub_TO_city_customer','FULL_origin_port_TO_logistic_hub_TO_city_customer',
    'zscore_weightbyorigin_port_TO_logistic_hub','zscore_unitsbyorigin_port_TO_logistic_hub',
    'zscore_weightbylogistic_hub_TO_city_customer','zscore_unitsbylogistic_hub_TO_city_customer',
    'zscore_weightbyFULL_origin_port_TO_logistic_hub_TO_city_customer',
    'zscore_unitsbyFULL_origin_port_TO_logistic_hub_TO_city_customer'
]


# ============================================================
# 6. BUILD FEATURE ROW (NO weight_class)
# ============================================================

def build_feature_row(origin_port, threepl, customs, hub, customer, units, weight, material_handling):

    origin_port = origin_port.strip().title()
    customer = customer.strip().title()
    hub = hub.strip().title() if hub != "NoHub" else "NoHub"

    # ============================
    # DISTANCE LOGIC
    # ============================
    if origin_port == customer:
        dist_port_to_hub = 0
        dist_hub_to_customer = 0
        distance = 0
    else:
        dist_port_to_hub = get_distance(origin_port, hub)
        dist_hub_to_customer = get_distance(hub, customer) if hub != "NoHub" else 0
        direct_port_customer = get_distance(origin_port, customer)
        distance = direct_port_customer if hub == "NoHub" else dist_port_to_hub + dist_hub_to_customer

    total_weight = units * weight
    has_hub = 0 if hub == "NoHub" else 1

    # ============================
    # ROUTE KEYS FOR Z-SCORES
    # ============================
    origin_port_TO_logistic_hub = f"{origin_port}_{hub}"
    logistic_hub_TO_city_customer = f"{hub}_{customer}"
    full_route = f"{origin_port}_{hub}_{customer}"

    # ============================
    # WEIGHT RISK CATEGORY
    # ============================
    weight_risk_category = compute_weight_risk(weight, material_handling)

    # ============================
    # Z-SCORE HELPER
    # ============================
    def compute_z(value, mean_series, std_series, key):
        mean = mean_series.get(key, None)
        std = std_series.get(key, None)
        if mean is None or std is None or std == 0 or np.isnan(std):
            return 0.0
        return (value - mean) / std

    # ============================
    # COMPUTE REAL Z-SCORES
    # ============================
    z_w_op_hub = compute_z(
        weight,
        group_stats["mean_weightbyorigin_port_TO_logistic_hub"],
        group_stats["std_weightbyorigin_port_TO_logistic_hub"],
        origin_port_TO_logistic_hub
    )

    z_u_op_hub = compute_z(
        units,
        group_stats["mean_unitsbyorigin_port_TO_logistic_hub"],
        group_stats["std_unitsbyorigin_port_TO_logistic_hub"],
        origin_port_TO_logistic_hub
    )

    z_w_hub_cust = compute_z(
        weight,
        group_stats["mean_weightbylogistic_hub_TO_city_customer"],
        group_stats["std_weightbylogistic_hub_TO_city_customer"],
        logistic_hub_TO_city_customer
    )

    z_u_hub_cust = compute_z(
        units,
        group_stats["mean_unitsbylogistic_hub_TO_city_customer"],
        group_stats["std_unitsbylogistic_hub_TO_city_customer"],
        logistic_hub_TO_city_customer
    )

    z_w_full = compute_z(
        weight,
        group_stats["mean_weightbyFULL_origin_port_TO_logistic_hub_TO_city_customer"],
        group_stats["std_weightbyFULL_origin_port_TO_logistic_hub_TO_city_customer"],
        full_route
    )

    z_u_full = compute_z(
        units,
        group_stats["mean_unitsbyFULL_origin_port_TO_logistic_hub_TO_city_customer"],
        group_stats["std_unitsbyFULL_origin_port_TO_logistic_hub_TO_city_customer"],
        full_route
    )

    # ============================
    # FINAL FEATURE VECTOR
    # ============================
    row = {
        "origin_port": origin_port,
        "3pl": threepl,
        "customs_procedures": customs,
        "logistic_hub": hub,
        "customer": customer,
        "units": units,
        "weight": weight,
        "material_handling": material_handling,
        "dist_port_to_hub": dist_port_to_hub,
        "distance": distance,
        "total_weight": total_weight,
        "has_hub": has_hub,
        "route_geo": full_route,
        "route_x_upl": f"{full_route}_{threepl}",
        "weight_risk_category": weight_risk_category,
        "3pl_customs": f"{threepl}_{customs}",
        "origin_lpi": get_lpi(origin_port),
        "customer_lpi": get_lpi(customer),
        "hub_lpi": get_lpi(hub),
        "origin_port_dist_port_to_hub_std": 0.0,
        "origin_port_TO_logistic_hub": origin_port_TO_logistic_hub,
        "logistic_hub_TO_city_customer": logistic_hub_TO_city_customer,
        "FULL_origin_port_TO_logistic_hub_TO_city_customer": full_route,
        "zscore_weightbyorigin_port_TO_logistic_hub": z_w_op_hub,
        "zscore_unitsbyorigin_port_TO_logistic_hub": z_u_op_hub,
        "zscore_weightbylogistic_hub_TO_city_customer": z_w_hub_cust,
        "zscore_unitsbylogistic_hub_TO_city_customer": z_u_hub_cust,
        "zscore_weightbyFULL_origin_port_TO_logistic_hub_TO_city_customer": z_w_full,
        "zscore_unitsbyFULL_origin_port_TO_logistic_hub_TO_city_customer": z_u_full,
    }

    return pd.DataFrame([[row[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)

# ... (Tutto il codice di caricamento dati Step 1-6 rimane identico) ...

# ============================================================
# 7. STREAMLIT APP UI (LAYOUT A COPPIE DI 2)
# ============================================================

st.title("🚚 Late Order Prediction App")
st.markdown("---")

st.subheader("📝 Inserisci i Dati Spedizione")

# Creiamo un contenitore per raggruppare gli input in modo ordinato
with st.container(border=True):
    
    # --- RIGA 1: Origin & Hub ---
    col1, col2 = st.columns(2)
    with col1:
        origin = st.selectbox("Origin Port:", ["Rotterdam", "Athens", "Barcelona"])
    with col2:
        hub = st.selectbox("Logistic Hub:", VALID_HUBS)

    # --- RIGA 2: Customer & 3PL ---
    col3, col4 = st.columns(2)
    with col3:
        customer = st.selectbox("Customer City:", sorted(VALID_CUSTOMERS))
    with col4:
        threepl = st.selectbox("Courier Provider:", ["v_001","v_002","v_003","v_004"])

    # --- RIGA 3: Customs & Material ---
    col5, col6 = st.columns(2)
    with col5:
        customs = st.selectbox("Customs Procedure:", ["CRF","DTD","DTP"])
    with col6:
        material_handling = st.selectbox("Material Handling Type:", [0.0,1.0,2.0,3.0,4.0,5.0])

    # --- RIGA 4: Units & Weight ---
    col7, col8 = st.columns(2)
    with col7:
        units = st.number_input("Units:", min_value=1, value=10)
    with col8:
        weight = st.number_input("Weight per Unit(g):", min_value=1.0, value=50.0)
    
    st.write("") # Spazio vuoto
    # Bottone a larghezza intera
    btn_predict = st.button("🚀 PREDICT LATE ORDER", use_container_width=True, type="primary")

# ============================================================
# 8. VISUALIZZAZIONE RISULTATO (SOTTO GLI INPUT)
# ============================================================

if btn_predict:
    # 1. Costruzione Features
    X = build_feature_row(origin, threepl, customs, hub, customer, units, weight, material_handling)

    # 2. Predizione
    prob = float(model.predict_proba(X)[0][1])
    pred = int(prob >= 0.5)

    # 3. Salvataggio
    new_entry = {
        "origin_port": origin,
        "logistic_hub": hub,
        "customer": customer,
        "threepl": threepl,
        "customs_procedures": customs,
        "material_handling": material_handling,
        "units": units,
        "weight": weight,
        "model_probability": prob,
        "model_prediction": pred
    }
    save_submission(new_entry)

    st.write("") # Spazio estetico
    st.subheader("📊 Risultato Previsione")

    # 4. Output Visivo
    if pred == 1:
        # RITARDO
        st.error(f"### ⚠️ PREDICTION: LIKELY LATE", icon="🚨")
        st.write(f"**Probability of Delay:** {prob*100:.1f}%")
        st.progress(prob, text="Risk Level: HIGH")
    else:
        # PUNTUALE
        st.success(f"### ✅ PREDICTION: LIKELY ON-TIME", icon="🎉")
        st.write(f"**Probability of Delay:** {prob*100:.1f}%")
        st.progress(prob, text="Risk Level: LOW")


# ============================================================
# 9. DASHBOARD STORICO (IN FONDO)
# ============================================================

st.markdown("---")
st.title("📊 Prediction Log History")

if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)

    # Ordine inverso (più recenti in alto)
    if not df.empty:
        df = df.iloc[::-1]

    # Nomi colonne
    pretty_cols = {c: c.replace("_", " ").title() for c in df.columns}
    df = df.rename(columns=pretty_cols)

    # Stile Tabella
    def highlight_late(val):
        color = '#ffcccc' if val == 1 else '#ccffcc' 
        return f'background-color: {color}'

    st.dataframe(
        df.style.applymap(highlight_late, subset=['Model Prediction']),
        use_container_width=True, 
        hide_index=True
    )
else:
    st.warning("No log file found yet.")