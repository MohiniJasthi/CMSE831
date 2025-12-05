# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pulp
import plotly.express as px

st.set_page_config(page_title="Multi-Objective Diet Optimizer", layout="wide")

st.title("Multi‑Objective Diet Optimization System")

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

DATA_PATH = "food_nutrition_ghg_dataset.csv"
df = load_data(DATA_PATH)

required_cols = [
    "Food", "Category",
    "Cost_per_100g", "Serving_Size_g",
    "Calories", "Protein_g", "Fat_g", "Carbs_g", "Fiber_g",
    "GHG_kg_CO2e"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

# ================== HELPER: CLASSIFY DIET TYPE ==================
def classify_diet_type(row):
    cat = str(row["Category"]).lower()
    # Adjust these rules to match your dataset
    if any(x in cat for x in ["meat", "pork", "beef", "chicken", "sausage"]):
        return "Meat"
    if "fish" in cat or "seafood" in cat:
        return "Fish"
    if "dairy" in cat or "egg" in cat:
        return "Vegetarian (incl. dairy/eggs)"
    if "vegetable" in cat or "fruit" in cat or "grain" in cat or "legume" in cat:
        return "Plant-based"
    return "Other"

df["Diet_Type"] = df.apply(classify_diet_type, axis=1)

# ================== SIDEBAR SETTINGS ==================
with st.sidebar.expander("Food filters", expanded=False):
    # Base category selection
    all_categories = sorted(df["Category"].dropna().unique().tolist())
    selected_categories = st.multiselect(
        "Include categories",
        options=all_categories,
        default=all_categories
    )

    # Start with category filter
    df_filtered = df[df["Category"].isin(selected_categories)]

    # High‑level diet patterns (mutually exclusive)
    vegan_only = st.checkbox("Vegan (only Plant-based)", value=False)
    vegetarian_only = st.checkbox("Vegetarian (no Meat/Fish)", value=False)
    pescatarian_only = st.checkbox("Pescatarian (Fish ok, no Meat)", value=False)
    meat_only = st.checkbox("Meat‑focused (only Meat/Fish)", value=False)

    # Apply pattern‑diet filters using Diet_Type
    if vegan_only:
        df_filtered = df_filtered[df_filtered["Diet_Type"] == "Plant-based"]
    elif vegetarian_only:
        df_filtered = df_filtered[~df_filtered["Diet_Type"].isin(["Meat", "Fish"])]
    elif pescatarian_only:
        df_filtered = df_filtered[df_filtered["Diet_Type"] != "Meat"]
    elif meat_only:
        df_filtered = df_filtered[df_filtered["Diet_Type"].isin(["Meat", "Fish"])]

    if df_filtered.empty:
        st.error("No foods left after applying dietary filters. Please relax constraints.")
        st.stop()


st.sidebar.subheader("Daily nutrient constraints")
calories_min = st.sidebar.number_input("Min Calories", 1200, 5000, 2000, step=50)
calories_max = st.sidebar.number_input("Max Calories", 1200, 5000, 2500, step=50)
protein_min = st.sidebar.number_input("Min Protein (g)", 0, 300, 75, step=5)
fat_max = st.sidebar.number_input("Max Fat (g)", 10, 300, 90, step=5)
carbs_max = st.sidebar.number_input("Max Carbs (g)", 10, 600, 300, step=10)
fiber_min = st.sidebar.number_input("Min Fiber (g)", 0, 100, 25, step=1)

st.sidebar.subheader("Optimization mode")
opt_mode = st.sidebar.selectbox(
    "Choose optimization objective",
    ["Minimize cost", "Minimize GHG", "Weighted cost + GHG"]
)

daily_budget = st.sidebar.number_input(
    "Daily budget (max total cost)",
    min_value=0.0,
    value=10.0,
    step=0.5
)

if opt_mode == "Weighted cost + GHG":
    weight_cost = st.sidebar.slider("Weight on COST", 0.0, 1.0, 0.5, 0.05)
    weight_ghg = 1.0 - weight_cost
else:
    weight_cost, weight_ghg = (1.0, 0.0) if opt_mode == "Minimize cost" else (0.0, 1.0)

st.sidebar.markdown(
    f"Effective weights → Cost: `{weight_cost:.2f}`, GHG: `{weight_ghg:.2f}`"
)

solve_button = st.sidebar.button("Run Optimization", type="primary")

# ================== DATA EXPLORATORY GRAPHS ==================
st.subheader("Dataset overview")

col1, col2 = st.columns(2)

with col1:
    # Count by Diet_Type
    diet_counts = df["Diet_Type"].value_counts().reset_index()
    diet_counts.columns = ["Diet_Type", "Count"]
    fig_diet = px.bar(
        diet_counts, x="Diet_Type", y="Count",
        title="Foods by diet type"
    )
    st.plotly_chart(fig_diet, use_container_width=True)

with col2:
    # Category-level avg cost vs avg GHG
    cat_summary = (
        df.groupby("Category")[["Cost_per_100g", "GHG_kg_CO2e"]]
        .mean()
        .reset_index()
    )
    fig_cost_ghg = px.scatter(
        cat_summary,
        x="Cost_per_100g",
        y="GHG_kg_CO2e",
        color="Category",
        size_max=25,
        title="Average cost vs GHG per category"
    )
    st.plotly_chart(fig_cost_ghg, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    fig_cost_hist = px.histogram(
        df, x="Cost_per_100g", nbins=30,
        title="Distribution of cost per 100 g"
    )
    st.plotly_chart(fig_cost_hist, use_container_width=True)

with col4:
    fig_ghg_hist = px.histogram(
        df, x="GHG_kg_CO2e", nbins=30,
        title="Distribution of GHG per 100 g"
    )
    st.plotly_chart(fig_ghg_hist, use_container_width=True)

st.markdown("These graphs describe how many foods fall into each diet type and how cost and GHG vary across items and categories.[file:3][web:24]")

st.subheader("Filtered food preview")
st.dataframe(
    df_filtered[[
        "Food", "Category", "Diet_Type",
        "Cost_per_100g", "Serving_Size_g",
        "Calories", "Protein_g", "Fat_g", "Carbs_g", "Fiber_g", "GHG_kg_CO2e"
    ]].head(25),
    use_container_width=True
)

# ================== OPTIMIZATION FUNCTION ==================
def solve_diet_lp(df_foods, weight_cost, weight_ghg):
    foods = df_foods["Food"].tolist()

    cost = df_foods.set_index("Food")["Cost_per_100g"].to_dict()
    calories = df_foods.set_index("Food")["Calories"].to_dict()
    protein = df_foods.set_index("Food")["Protein_g"].to_dict()
    fat = df_foods.set_index("Food")["Fat_g"].to_dict()
    carbs = df_foods.set_index("Food")["Carbs_g"].to_dict()
    fiber = df_foods.set_index("Food")["Fiber_g"].to_dict()
    ghg = df_foods.set_index("Food")["GHG_kg_CO2e"].to_dict()

    model = pulp.LpProblem("Diet_Optimization", pulp.LpMinimize)
    qty = pulp.LpVariable.dicts("qty", foods, lowBound=0)

    total_cost = pulp.lpSum(cost[f] * qty[f] for f in foods)
    total_cal = pulp.lpSum(calories[f] * qty[f] for f in foods)
    total_protein = pulp.lpSum(protein[f] * qty[f] for f in foods)
    total_fat = pulp.lpSum(fat[f] * qty[f] for f in foods)
    total_carbs = pulp.lpSum(carbs[f] * qty[f] for f in foods)
    total_fiber = pulp.lpSum(fiber[f] * qty[f] for f in foods)
    total_ghg = pulp.lpSum(ghg[f] * qty[f] for f in foods)

    # multi‑objective: weighted cost + GHG
    model += weight_cost * total_cost + weight_ghg * total_ghg

    # nutrient constraints (use your sidebar values)
    model += total_cal >= calories_min
    model += total_cal <= calories_max
    model += total_protein >= protein_min
    model += total_fat <= fat_max
    model += total_carbs <= carbs_max
    model += total_fiber >= fiber_min

    # budget
    model += total_cost <= daily_budget

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        return None, {"Status": status}

    rows = []
    for f in foods:
        q = qty[f].varValue
        if q and q > 1e-6:
            rows.append({
                "Food": f,
                "Qty_100g": q,
                "Qty_g": q * 100,
                "Cost": cost[f] * q,
                "Calories": calories[f] * q,
                "Protein_g": protein[f] * q,
                "Fat_g": fat[f] * q,
                "Carbs_g": carbs[f] * q,
                "Fiber_g": fiber[f] * q,
                "GHG_kg_CO2e": ghg[f] * q,
            })
    sol_df = pd.DataFrame(rows)
    totals = {
        "Status": status,
        "Total_cost": float(total_cost.value()),
        "Total_GHG": float(total_ghg.value()),
        "Total_calories": float(total_cal.value()),
        "Total_protein": float(total_protein.value()),
        "Total_fat": float(total_fat.value()),
        "Total_carbs": float(total_carbs.value()),
        "Total_fiber": float(total_fiber.value()),
    }
    return sol_df, totals

# ================== RUN MODEL & SHOW RESULTS ==================
if solve_button:
    # 1) Minimize cost only
    plan_cost, totals_cost = solve_diet_lp(df_filtered, weight_cost=1.0, weight_ghg=0.0)

    # 2) Minimize GHG only
    plan_ghg, totals_ghg = solve_diet_lp(df_filtered, weight_cost=0.0, weight_ghg=1.0)

    # 3) Weighted cost + GHG (use slider value from sidebar)
    plan_both, totals_both = solve_diet_lp(df_filtered, weight_cost=weight_cost, weight_ghg=weight_ghg)

    st.subheader("Optimized diet plans under same constraints")

    tabs = st.tabs(["Minimize Cost", "Minimize GHG", "Weighted Cost+GHG"])

    # ---- Tab 1: Minimize Cost ----
    with tabs[0]:
        st.markdown("### Diet plan: Minimize total cost")
        if plan_cost is None:
            st.warning(f"Model status: {totals_cost['Status']}")
        else:
            st.dataframe(plan_cost, use_container_width=True)
            st.write(totals_cost)

    # ---- Tab 2: Minimize GHG ----
    with tabs[1]:
        st.markdown("### Diet plan: Minimize total GHG")
        if plan_ghg is None:
            st.warning(f"Model status: {totals_ghg['Status']}")
        else:
            st.dataframe(plan_ghg, use_container_width=True)
            st.write(totals_ghg)

    # ---- Tab 3: Weighted Cost + GHG ----
    with tabs[2]:
        st.markdown("### Diet plan: Weighted cost + GHG")
        if plan_both is None:
            st.warning(f"Model status: {totals_both['Status']}")
        else:
            st.dataframe(plan_both, use_container_width=True)
            st.write(totals_both)

    # Optional: summary table comparing the three objectives
    rows = []

    for mode_name, totals in [
        ("Minimize cost", totals_cost),
        ("Minimize GHG", totals_ghg),
        ("Weighted", totals_both),
    ]:
        rows.append({
            "Mode": mode_name,
            "Status": totals.get("Status", "N/A"),
            "Total_cost": totals.get("Total_cost", np.nan),
            "Total_GHG": totals.get("Total_GHG", np.nan),
            "Total_calories": totals.get("Total_calories", np.nan),
            "Total_protein": totals.get("Total_protein", np.nan),
        })
    
    summary = pd.DataFrame(rows)
    st.subheader("Objective comparison across optimization modes")
    st.dataframe(summary, use_container_width=True)

