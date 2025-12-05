import streamlit as st
import pandas as pd
import pulp
import plotly.express as px

st.set_page_config(
    page_title="Multi‑Objective Diet Optimizer",
    layout="wide"
)

st.title("Multi‑Objective Diet Optimization System")
st.markdown(
    """
Design a **daily diet** that is affordable, nutritionally balanced, and low in greenhouse gas (GHG) emissions.  
This app uses **linear programming (PuLP)** to choose quantities of foods from your dataset.
"""
)

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

DATA_PATH = "food_nutrition_ghg_dataset.csv"
df = load_data(DATA_PATH)

required_cols = [
    "Food", "Cost_per_100g", "Calories", "Protein_g",
    "Fat_g", "Carbs_g", "Fiber_g", "GHG_kg_CO2e", "Serving_Size_g"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"These required columns are missing from the CSV: {missing}")
    st.stop()

st.sidebar.header("Model Settings")

# Filter foods by category or tags
with st.sidebar.expander("Food filters", expanded=False):
    all_categories = sorted(df["Category"].dropna().unique().tolist())
    selected_categories = st.multiselect(
        "Include categories",
        options=all_categories,
        default=all_categories
    )
    df_filtered = df[df["Category"].isin(selected_categories)]

    # Optional: vegetarian / vegan style filters based on Category
    vegetarian_only = st.checkbox("Exclude Meat", value=False)
    pescatarian_only = st.checkbox("Exclude Meat but allow Fish", value=False)

    if vegetarian_only:
        df_filtered = df_filtered[~df_filtered["Category"].isin(["Meat", "Fish"])]
    elif pescatarian_only:
        df_filtered = df_filtered[df_filtered["Category"] != "Meat"]

    if df_filtered.empty:
        st.error("No foods left after applying filters. Adjust your selections.")
        st.stop()

st.sidebar.markdown("---")

st.sidebar.subheader("Daily nutrient constraints")

calories_min = st.sidebar.number_input("Min Calories", 1200, 5000, 2000, step=50)
calories_max = st.sidebar.number_input("Max Calories", 1200, 5000, 2500, step=50)

protein_min = st.sidebar.number_input("Min Protein (g)", 0, 300, 75, step=5)
fat_max = st.sidebar.number_input("Max Fat (g)", 10, 300, 90, step=5)
carbs_max = st.sidebar.number_input("Max Carbs (g)", 10, 600, 300, step=10)
fiber_min = st.sidebar.number_input("Min Fiber (g)", 0, 100, 25, step=1)

st.sidebar.markdown("---")

# Cost / GHG preferences
st.sidebar.subheader("Objectives")

daily_budget = st.sidebar.number_input(
    "Daily budget (max cost, in same units as Cost_per_100g)",
    min_value=0.0,
    value=10.0,
    step=0.5
)

weight_cost = st.sidebar.slider("Weight on COST (0 = ignore, 1 = prioritize strongly)", 0.0, 1.0, 0.5, 0.05)
weight_ghg = st.sidebar.slider("Weight on GHG (0 = ignore, 1 = prioritize strongly)", 0.0, 1.0, 0.5, 0.05)

# Normalize weights so they sum to 1 (unless both zero)
if weight_cost == 0 and weight_ghg == 0:
    weight_cost, weight_ghg = 0.5, 0.5
else:
    total_w = weight_cost + weight_ghg
    weight_cost /= total_w
    weight_ghg /= total_w

st.sidebar.markdown(
    f"**Effective objective weights**  \nCost: `{weight_cost:.2f}`  |  GHG: `{weight_ghg:.2f}`"
)

st.sidebar.markdown("---")
solve_button = st.sidebar.button("Run Optimization", type="primary")

st.subheader("Food dataset preview")
st.dataframe(
    df_filtered[[
        "Food", "Category", "Cost_per_100g", "Calories",
        "Protein_g", "Fat_g", "Carbs_g", "Fiber_g", "GHG_kg_CO2e"
    ]].head(20),
    use_container_width=True,
    height=300
)

st.caption(
    "Using real‑world style food data with cost, basic macronutrients, fiber, and GHG emissions per food item.[file:3]"
)

def solve_diet_lp(df_foods: pd.DataFrame):
    foods = df_foods["Food"].tolist()

    # Parameters per 100g
    cost = df_foods.set_index("Food")["Cost_per_100g"].to_dict()
    calories = df_foods.set_index("Food")["Calories"].to_dict()
    protein = df_foods.set_index("Food")["Protein_g"].to_dict()
    fat = df_foods.set_index("Food")["Fat_g"].to_dict()
    carbs = df_foods.set_index("Food")