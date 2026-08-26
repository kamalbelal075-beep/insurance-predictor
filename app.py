import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

st.set_page_config(page_title="Insurance Charges Predictor", layout="wide")
st.title("💰 Insurance Charges Prediction")
st.caption("Trains several regression models on the insurance dataset and compares them.")

# ---------------------------------------------------------------------
# 1) Load data
# ---------------------------------------------------------------------
st.sidebar.header("1) Data")

DEFAULT_DATA_PATH = "insurance.csv"  # must sit next to app.py in the repo

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

uploaded_file = st.sidebar.file_uploader(
    "Optionally upload a different insurance.csv", type=["csv"]
)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.sidebar.success("Using your uploaded file.")
else:
    try:
        df = load_data(DEFAULT_DATA_PATH)
        st.sidebar.info(f"Using bundled `{DEFAULT_DATA_PATH}` from the repo.")
    except FileNotFoundError:
        st.error(
            f"Couldn't find `{DEFAULT_DATA_PATH}` next to app.py. "
            "Upload a CSV from the sidebar instead."
        )
        st.stop()

st.subheader("Raw data preview")
st.dataframe(df.head())

# ---------------------------------------------------------------------
# 2) Cleaning
# ---------------------------------------------------------------------
df = df.drop_duplicates()

col1, col2 = st.columns(2)
with col1:
    st.write("**Missing values**")
    st.dataframe(df.isnull().sum())
with col2:
    st.write("**Shape after dropping duplicates**")
    st.write(df.shape)

# ---------------------------------------------------------------------
# 3) Encode categoricals
# ---------------------------------------------------------------------
df_model = df.copy()
df_model['sex'] = df_model['sex'].map({"male": 1, "female": 2})
df_model['smoker'] = df_model['smoker'].map({"yes": 1, "no": 2})
df_model['region'] = df_model['region'].map(
    {"southwest": 1, "northwest": 2, "southeast": 3, "northeast": 4}
)

# ---------------------------------------------------------------------
# 4) Correlation heatmap
# ---------------------------------------------------------------------
st.subheader("Correlation heatmap")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df_model.corr(), annot=True, vmin=-1, vmax=1, cmap="cool", linewidths=2, ax=ax)
st.pyplot(fig)

# ---------------------------------------------------------------------
# 5) Train/test split + scaling
# ---------------------------------------------------------------------
x = df_model.drop(['charges'], axis=1)
y = df_model['charges']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ---------------------------------------------------------------------
# 6) Train models
# ---------------------------------------------------------------------
st.sidebar.header("2) Models to train")
run_svr = st.sidebar.checkbox("SVR (slow-ish, low accuracy on this data)", value=True)
run_xgb = st.sidebar.checkbox("XGBoost", value=XGB_AVAILABLE, disabled=not XGB_AVAILABLE)

@st.cache_resource
def train_models(x_train_scaled, y_train, run_svr, run_xgb):
    models = {}

    lin = LinearRegression().fit(x_train_scaled, y_train)
    models['Linear'] = lin

    knn = KNeighborsRegressor(n_neighbors=3).fit(x_train_scaled, y_train)
    models['KNN'] = knn

    rfr = RandomForestRegressor(n_estimators=200, random_state=42).fit(x_train_scaled, y_train)
    models['RandomForestRegressor'] = rfr

    gbr = GradientBoostingRegressor().fit(x_train_scaled, y_train)
    models['GradientBoostingRegressor'] = gbr

    tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(x_train_scaled, y_train)
    models['DecisionTreeRegressor'] = tree

    if run_svr:
        svr = SVR(C=100, epsilon=0.01, gamma=0.1, kernel='rbf').fit(x_train_scaled, y_train)
        models['SVR'] = svr

    if run_xgb and XGB_AVAILABLE:
        xg = XGBRegressor(n_estimators=100, learning_rate=0.15, max_depth=3, subsample=0.8)
        xg.fit(x_train_scaled, y_train)
        models['XGBRegressor'] = xg

    return models

with st.spinner("Training models..."):
    models = train_models(x_train_scaled, y_train, run_svr, run_xgb)

# ---------------------------------------------------------------------
# 7) Evaluate
# ---------------------------------------------------------------------
rows = []
preds = {}
for name, model in models.items():
    y_pred = model.predict(x_test_scaled)
    preds[name] = y_pred
    rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    })

results_df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)

st.subheader("Model comparison")
st.dataframe(results_df.style.format({"MAE": "{:.2f}", "MSE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"}))

fig2, ax2 = plt.subplots(figsize=(8, 4))
results_sorted = results_df.set_index("Model")["R2"].sort_values()
results_sorted.plot(kind="bar", ax=ax2, title="R2 Score by Model")
ax2.set_ylabel("R2")
st.pyplot(fig2)

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
st.success(f"🏆 Best model: **{best_model_name}** (R² = {results_df.iloc[0]['R2']:.4f})")

# ---------------------------------------------------------------------
# 8) Live prediction
# ---------------------------------------------------------------------
st.header("Try a prediction")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex_in = st.selectbox("Sex", ["male", "female"])
with c2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
with c3:
    smoker_in = st.selectbox("Smoker", ["no", "yes"])
    region_in = st.selectbox("Region", ["southwest", "northwest", "southeast", "northeast"])

if st.button("Predict charges"):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": {"male": 1, "female": 2}[sex_in],
        "bmi": bmi,
        "children": children,
        "smoker": {"yes": 1, "no": 2}[smoker_in],
        "region": {"southwest": 1, "northwest": 2, "southeast": 3, "northeast": 4}[region_in],
    }])[x.columns]  # keep same column order as training data

    input_scaled = scaler.transform(input_df)
    prediction = best_model.predict(input_scaled)[0]
    st.metric("Predicted insurance charge", f"${prediction:,.2f}")
