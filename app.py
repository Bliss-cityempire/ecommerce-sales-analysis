import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Advanced E-Commerce Analytics Suite",
    page_icon="🛍️",
    layout="wide"
)

# --- APP HEADER ---
st.title("🛍️ Advanced E-Commerce Analytics & Predictive Modeling Suite")
st.markdown("""
This platform processes raw e-commerce transaction ledgers to generate real-time operational insights, 
track product performance, and forecast future revenue using predictive modeling.
""")
st.write("---")

# --- SIDEBAR: SAMPLE DATA GENERATOR ---
st.sidebar.header("⚙️ Data Control Center")

@st.cache_data
def generate_mock_ecommerce_data():
    """Generates a realistic e-commerce dataset for demonstration."""
    np.random.seed(42)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=180)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    products = {
        'Electronics': [299.99, 499.99, 89.99],
        'Apparel': [24.99, 45.00, 79.99],
        'Home Goods': [15.50, 120.00, 45.00],
        'Beauty': [19.99, 34.50, 55.00]
    }
    
    data = []
    for current_date in date_range:
        num_orders = np.random.randint(5, 25)
        for _ in range(num_orders):
            category = np.random.choice(list(products.keys()))
            price = np.random.choice(products[category])
            quantity = np.random.randint(1, 4)
            revenue = price * quantity
            
            data.append({
                "Order Date": current_date,
                "Category": category,
                "Quantity": quantity,
                "Unit Price": price,
                "Total Revenue": revenue,
                "Customer Segment": np.random.choice(["Returning", "New", "VIP"], p=[0.5, 0.3, 0.2])
            })
            
    df = pd.DataFrame(data)
    return df

# Initialize Data
raw_df = generate_mock_ecommerce_data()

# File Upload Option for Real Users
uploaded_file = st.sidebar.file_uploader("Upload your transaction ledger (CSV)", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        st.sidebar.success("Successfully loaded custom data!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}. Using demo data instead.")
        df = raw_df
else:
    df = raw_df
    st.sidebar.info("💡 Using built-in simulation data. Upload a CSV to test your own data.")

# --- NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard", "🔮 Predictive Sales Forecast", "🛒 Operational Insights"])

# ==========================================
# TAB 1: EXECUTIVE DASHBOARD
# ==========================================
with tab1:
    st.header("Financial Performance Overview")
    
    # High-Level Metrics
    total_rev = df["Total Revenue"].sum()
    total_orders = len(df)
    avg_order_val = total_rev / total_orders if total_orders > 0 else 0
    total_units = df["Quantity"].sum()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gross Revenue", f"${total_rev:,.2f}")
    m2.metric("Total Orders Processed", f"{total_orders:,}")
    m3.metric("Average Order Value (AOV)", f"${avg_order_val:.2f}")
    m4.metric("Total Units Sold", f"{total_units:,}")
    
    st.write("---")
    
    # Charts Section
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Revenue Contribution by Product Category")
        category_data = df.groupby("Category")["Total Revenue"].sum().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']
        category_data.plot(kind='bar', color=colors, ax=ax)
        ax.set_ylabel("Revenue ($)")
        ax.set_xlabel("Category")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
    with c2:
        st.subheader("Customer Segment Distribution")
        segment_data = df.groupby("Customer Segment")["Total Revenue"].sum()
        
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.pie(segment_data, labels=segment_data.index, autopct='%1.1f%%', startangle=90, colors=['#2ca02c', '#98df8a', '#d62728'])
        ax.axis('equal') 
        plt.tight_layout()
        st.pyplot(fig)

# ==========================================
# TAB 2: PREDICTIVE SALES FORECAST
# ==========================================
with tab2:
    st.header("Linear Regression Predictive Model")
    st.markdown("""
    This module aggregates historical daily revenue data and applies a linear regression trend analysis 
    to forecast revenue growth and velocity over an upcoming operational horizon.
    """)
    
    # Prepare Daily Revenue for Simple Modeling
    daily_revenue = df.groupby("Order Date")["Total Revenue"].sum().reset_index()
    daily_revenue = daily_revenue.sort_values("Order Date")
    
    # Create time steps index for modeling (X)
    daily_revenue['DayIndex'] = np.arange(len(daily_revenue))
    
    X = daily_revenue['DayIndex'].values
    y = daily_revenue['Total Revenue'].values
    
    # Manual Linear Regression implementation for lightweight deployment
    slope, intercept = np.polyfit(X, y, 1)
    
    # User Input Horizon
    forecast_days = st.slider("Select Forecast Horizon (Days into Future)", min_value=7, max_value=90, value=30)
    
    # Generate future indices
    future_X = np.arange(len(daily_revenue), len(daily_revenue) + forecast_days)
    future_predictions = slope * future_X + intercept
    
    # Generate future dates
    last_date = daily_revenue['Order Date'].max()
    future_dates = [last_date + datetime.timedelta(days=int(i)) for i in range(1, forecast_days + 1)]
    
    # Combine historical and predicted for plotting
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_revenue['Order Date'], y, label="Historical Daily Revenue", color="#1f77b4", alpha=0.6)
    ax.plot(daily_revenue['Order Date'], slope * X + intercept, label="Historical Trendline", color="#2ca02c", linestyle="--")
    ax.plot(future_dates, future_predictions, label=f"Predictive Forecast ({forecast_days}d)", color="#d62728", linewidth=2.5)
    
    ax.set_title("Revenue Velocity & Predictive Demand Curve", fontsize=14)
    ax.set_ylabel("Revenue ($)")
    ax.set_xlabel("Timeline")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig)
    
    # Forecast Metrics
    expected_total = future_predictions.sum()
    st.info(f"🔮 **Predictive Model Output:** Based on historical pipeline velocity, the suite forecasts an additional **${expected_total:,.2f}** in gross transactional revenue over the next **{forecast_days} days**.")

# ==========================================
# TAB 3: OPERATIONAL INSIGHTS
# ==========================================
with tab3:
    st.header("Audit Trails & Data Logs")
    st.write("Filter and verify transaction subsets instantly below.")
    
    # Interactive Search Table
    selected_cat = st.selectbox("Filter Data Log by Category:", ["All"] + list(df["Category"].unique()))
    
    filtered_df = df.copy()
    if selected_cat != "All":
        filtered_df = filtered_df[filtered_df["Category"] == selected_cat]
        
    st.dataframe(filtered_df.sort_values(by="Order Date", ascending=False), use_container_width=True)
    
    st.success(f"Showing {len(filtered_df)} corresponding transactional records in historical storage.")