

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Customer Sales Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}

[data-testid="stMetricValue"] {
    font-size: 25px;
    font-weight: 700;
}

h1 {
    font-weight: 800;
}

h2 {
    font-weight: 700;
}

h3 {
    font-weight: 650;
}

.stButton > button {
    width: 100%;
    border-radius: 8px;
    height: 45px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():

    df = pd.read_csv(
        "customer_sales_500_records_31_kpis.csv"
    )

    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"],
        errors="coerce"
    )

    df["Delivery_Date"] = pd.to_datetime(
        df["Delivery_Date"],
        errors="coerce"
    )

    categorical_columns = [
        "Region",
        "City",
        "Category",
        "Product",
        "Customer_Gender",
        "Customer_Segment",
        "Sales_Channel",
        "Salesperson",
        "Payment_Method"
    ]

    for col in categorical_columns:

        if col in df.columns:

            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .fillna("Unknown")
                .astype(str)
            )

    if "Returned" in df.columns:

        df["Returned"] = (
            df["Returned"]
            .astype("string")
            .str.strip()
            .fillna("No")
            .astype(str)
        )

    numeric_columns = [
        "Customer_Age",
        "Days_Since_Last_Order",
        "Units_Sold",
        "Unit_Price",
        "Gross_Sales",
        "Discount_Pct",
        "Discount_Amount",
        "Net_Sales",
        "Cost",
        "Profit",
        "Profit_Margin_Pct",
        "Shipping_Cost",
        "Tax_Pct",
        "Tax_Amount",
        "Total_Order_Value",
        "Delivery_Days",
        "Return_Amount",
        "Customer_Satisfaction"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df["Order_Year"] = (
        df["Order_Date"].dt.year
    )

    df["Order_Month"] = (
        df["Order_Date"].dt.month
    )

    df["Order_Month_Name"] = (
        df["Order_Date"].dt.strftime("%b")
    )

    df["Order_DayOfWeek"] = (
        df["Order_Date"].dt.dayofweek
    )

    df["Order_Quarter"] = (
        df["Order_Date"].dt.quarter
    )

    df["Is_Weekend"] = (
        df["Order_DayOfWeek"] >= 5
    ).astype(int)

    df["Return_Flag"] = (
        df["Returned"]
        .str.lower()
        .eq("yes")
        .astype(int)
    )

    return df


@st.cache_resource
def train_sales_model(df):

    features = [
        "Region",
        "City",
        "Category",
        "Product",
        "Customer_Age",
        "Customer_Gender",
        "Days_Since_Last_Order",
        "Customer_Segment",
        "Sales_Channel",
        "Salesperson",
        "Payment_Method",
        "Units_Sold",
        "Unit_Price",
        "Discount_Pct",
        "Shipping_Cost",
        "Tax_Pct",
        "Delivery_Days",
        "Customer_Satisfaction",
        "Order_Year",
        "Order_Month",
        "Order_DayOfWeek",
        "Order_Quarter",
        "Is_Weekend"
    ]

    target = "Total_Order_Value"

    model_df = df[
        features + [target]
    ].copy()

    model_df = model_df.dropna(
        subset=[target]
    )

    X = model_df[features]
    y = model_df[target]

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_features = [
        col
        for col in features
        if col not in categorical_features
    ]

    try:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            encoder
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=1
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return (
        pipeline,
        mae,
        rmse,
        r2,
        features
    )


@st.cache_resource
def train_return_model(df):

    features = [
        "Region",
        "City",
        "Category",
        "Product",
        "Customer_Age",
        "Customer_Gender",
        "Days_Since_Last_Order",
        "Customer_Segment",
        "Sales_Channel",
        "Salesperson",
        "Payment_Method",
        "Units_Sold",
        "Unit_Price",
        "Discount_Pct",
        "Shipping_Cost",
        "Tax_Pct",
        "Delivery_Days",
        "Customer_Satisfaction"
    ]

    X = df[features].copy()

    y = df["Return_Flag"]

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_features = [
        col
        for col in features
        if col not in categorical_features
    ]

    try:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            encoder
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=1
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline.fit(
        X_train,
        y_train
    )

    accuracy = pipeline.score(
        X_test,
        y_test
    )

    return (
        pipeline,
        accuracy,
        features
    )


@st.cache_data
def create_customer_segments(df):

    features = [
        "Total_Order_Value",
        "Units_Sold",
        "Profit",
        "Customer_Satisfaction",
        "Days_Since_Last_Order"
    ]

    segment_data = df[
        features
    ].copy()

    segment_data = segment_data.fillna(
        segment_data.median(numeric_only=True)
    )

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        segment_data
    )

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(
        scaled_data
    )

    result = df.copy()

    result["Cluster"] = clusters

    labels = {
        0: "Segment A",
        1: "Segment B",
        2: "Segment C",
        3: "Segment D"
    }

    result["Customer_Group"] = (
        result["Cluster"].map(labels)
    )

    return result


df = load_data()

sales_model, mae, rmse, r2, sales_features = (
    train_sales_model(df)
)

return_model, return_accuracy, return_features = (
    train_return_model(df)
)

segmented_df = create_customer_segments(df)


st.sidebar.title(
    "📊 Customer Sales"
)

st.sidebar.markdown(
    "## Navigation"
)

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Dashboard",
        "📈 Sales Analytics",
        "👥 Customer Segmentation",
        "🤖 Sales Prediction",
        "🔄 Return Prediction",
        "📋 Data Explorer"
    ]
)

st.sidebar.markdown("---")

st.sidebar.metric(
    "Total Records",
    f"{len(df):,}"
)

st.sidebar.metric(
    "Total Revenue",
    f"₹{df['Total_Order_Value'].sum():,.0f}"
)

st.sidebar.metric(
    "Total Profit",
    f"₹{df['Profit'].sum():,.0f}"
)


if page == "🏠 Dashboard":

    st.title(
        "📊 Customer Sales Prediction System"
    )

    st.subheader(
        "Interactive Sales Intelligence Dashboard"
    )

    st.markdown("---")

    total_revenue = (
        df["Total_Order_Value"].sum()
    )

    total_profit = (
        df["Profit"].sum()
    )

    total_orders = len(df)

    average_order = (
        df["Total_Order_Value"].mean()
    )

    return_rate = (
        df["Return_Flag"].mean() * 100
    )

    satisfaction = (
        df["Customer_Satisfaction"].mean()
    )

    profit_margin = (
        df["Profit_Margin_Pct"].mean()
    )

    delivery_days = (
        df["Delivery_Days"].mean()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "💰 Total Revenue",
        f"₹{total_revenue:,.0f}"
    )

    c2.metric(
        "💵 Total Profit",
        f"₹{total_profit:,.0f}"
    )

    c3.metric(
        "🛒 Total Orders",
        f"{total_orders:,}"
    )

    c4.metric(
        "📦 Average Order",
        f"₹{average_order:,.0f}"
    )

    st.markdown("")

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "🔄 Return Rate",
        f"{return_rate:.2f}%"
    )

    c6.metric(
        "⭐ Satisfaction",
        f"{satisfaction:.2f}/5"
    )

    c7.metric(
        "📈 Profit Margin",
        f"{profit_margin:.2f}%"
    )

    c8.metric(
        "🚚 Avg Delivery",
        f"{delivery_days:.1f} days"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        monthly = (
            df.groupby(
                ["Order_Year", "Order_Month"]
            )["Total_Order_Value"]
            .sum()
            .reset_index()
        )

        monthly["Period"] = (
            monthly["Order_Year"].astype(str)
            + "-"
            + monthly["Order_Month"]
            .astype(str)
            .str.zfill(2)
        )

        fig = px.line(
            monthly,
            x="Period",
            y="Total_Order_Value",
            markers=True,
            title="📈 Monthly Revenue Trend"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        region_sales = (
            df.groupby("Region")[
                "Total_Order_Value"
            ]
            .sum()
            .reset_index()
            .sort_values(
                "Total_Order_Value",
                ascending=False
            )
        )

        fig = px.bar(
            region_sales,
            x="Region",
            y="Total_Order_Value",
            text_auto=".2s",
            title="🌍 Revenue by Region"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    with col1:

        category_sales = (
            df.groupby("Category")[
                "Total_Order_Value"
            ]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            category_sales,
            names="Category",
            values="Total_Order_Value",
            hole=0.45,
            title="📦 Sales by Category"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        channel_sales = (
            df.groupby("Sales_Channel")[
                "Total_Order_Value"
            ]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            channel_sales,
            names="Sales_Channel",
            values="Total_Order_Value",
            hole=0.45,
            title="🛍️ Sales by Channel"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader(
        "🏆 Top 10 Products"
    )

    top_products = (
        df.groupby("Product")[
            "Total_Order_Value"
        ]
        .sum()
        .reset_index()
        .sort_values(
            "Total_Order_Value",
            ascending=False
        )
        .head(10)
    )

    fig = px.bar(
        top_products,
        x="Total_Order_Value",
        y="Product",
        orientation="h",
        text_auto=".2s",
        title="Top Products by Revenue"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


elif page == "📈 Sales Analytics":

    st.title(
        "📈 Sales Analytics"
    )

    st.write(
        "Analyze revenue, profit, products, regions, "
        "customers and sales channels."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        regions = st.multiselect(
            "Region",
            sorted(df["Region"].unique()),
            default=sorted(
                df["Region"].unique()
            )
        )

    with col2:

        categories = st.multiselect(
            "Category",
            sorted(df["Category"].unique()),
            default=sorted(
                df["Category"].unique()
            )
        )

    with col3:

        channels = st.multiselect(
            "Sales Channel",
            sorted(df["Sales_Channel"].unique()),
            default=sorted(
                df["Sales_Channel"].unique()
            )
        )

    filtered = df[
        df["Region"].isin(regions)
        &
        df["Category"].isin(categories)
        &
        df["Sales_Channel"].isin(channels)
    ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Revenue",
        f"₹{filtered['Total_Order_Value'].sum():,.0f}"
    )

    c2.metric(
        "Profit",
        f"₹{filtered['Profit'].sum():,.0f}"
    )

    c3.metric(
        "Orders",
        f"{len(filtered):,}"
    )

    c4.metric(
        "Units Sold",
        f"{filtered['Units_Sold'].sum():,.0f}"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        city_sales = (
            filtered.groupby("City")[
                "Total_Order_Value"
            ]
            .sum()
            .reset_index()
            .sort_values(
                "Total_Order_Value",
                ascending=False
            )
            .head(15)
        )

        fig = px.bar(
            city_sales,
            x="Total_Order_Value",
            y="City",
            orientation="h",
            title="Top Cities by Revenue"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        salesperson_sales = (
            filtered.groupby("Salesperson")[
                "Total_Order_Value"
            ]
            .sum()
            .reset_index()
            .sort_values(
                "Total_Order_Value",
                ascending=False
            )
        )

        fig = px.bar(
            salesperson_sales,
            x="Salesperson",
            y="Total_Order_Value",
            title="Salesperson Performance"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    with col1:

        age_sales = (
            filtered.groupby("Customer_Age")[
                "Total_Order_Value"
            ]
            .mean()
            .reset_index()
        )

        fig = px.scatter(
            age_sales,
            x="Customer_Age",
            y="Total_Order_Value",
            title="Customer Age vs Average Order Value"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        satisfaction_data = (
            filtered.groupby(
                "Customer_Satisfaction"
            )["Total_Order_Value"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            satisfaction_data,
            x="Customer_Satisfaction",
            y="Total_Order_Value",
            title="Satisfaction vs Average Sales"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


elif page == "👥 Customer Segmentation":

    st.title(
        "👥 Customer Segmentation"
    )

    st.write(
        "K-Means clustering groups customers according "
        "to their purchasing behavior."
    )

    st.markdown("---")

    summary = (
        segmented_df.groupby(
            "Customer_Group"
        )
        .agg(
            Customers=(
                "Order_ID",
                "count"
            ),
            Revenue=(
                "Total_Order_Value",
                "sum"
            ),
            Average_Order=(
                "Total_Order_Value",
                "mean"
            ),
            Average_Profit=(
                "Profit",
                "mean"
            ),
            Satisfaction=(
                "Customer_Satisfaction",
                "mean"
            ),
            Avg_Days_Since_Order=(
                "Days_Since_Last_Order",
                "mean"
            )
        )
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            segmented_df,
            x="Days_Since_Last_Order",
            y="Total_Order_Value",
            size="Profit",
            color="Customer_Group",
            hover_data=[
                "Customer_Age",
                "Product",
                "Region"
            ],
            title="Customer Segmentation"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        segment_revenue = (
            segmented_df.groupby(
                "Customer_Group"
            )["Total_Order_Value"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            segment_revenue,
            x="Customer_Group",
            y="Total_Order_Value",
            color="Customer_Group",
            title="Revenue by Customer Segment"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader(
        "🎯 Business Use of Segmentation"
    )

    st.info(
        """
        Customer segmentation can help identify:

        • High-value customers

        • Low-value customers

        • Frequently purchasing customers

        • Customers becoming inactive

        • Customers with high sales potential

        • Customers suitable for targeted campaigns
        """
    )


elif page == "🤖 Sales Prediction":

    st.title(
        "🤖 Sales Prediction"
    )

    st.write(
        "Enter customer and order details to estimate "
        "the expected Total Order Value."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        region = st.selectbox(
            "Region",
            sorted(df["Region"].unique()),
            key="sales_region"
        )

        city = st.selectbox(
            "City",
            sorted(df["City"].unique()),
            key="sales_city"
        )

        category = st.selectbox(
            "Category",
            sorted(df["Category"].unique()),
            key="sales_category"
        )

        product = st.selectbox(
            "Product",
            sorted(df["Product"].unique()),
            key="sales_product"
        )

        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=30,
            key="sales_age"
        )

        gender = st.selectbox(
            "Gender",
            sorted(df["Customer_Gender"].unique()),
            key="sales_gender"
        )

    with col2:

        days = st.number_input(
            "Days Since Last Order",
            min_value=0,
            max_value=1000,
            value=30,
            key="sales_days"
        )

        segment = st.selectbox(
            "Customer Segment",
            sorted(df["Customer_Segment"].unique()),
            key="sales_segment"
        )

        channel = st.selectbox(
            "Sales Channel",
            sorted(df["Sales_Channel"].unique()),
            key="sales_channel"
        )

        salesperson = st.selectbox(
            "Salesperson",
            sorted(df["Salesperson"].unique()),
            key="sales_salesperson"
        )

        payment = st.selectbox(
            "Payment Method",
            sorted(df["Payment_Method"].unique()),
            key="sales_payment"
        )

        units = st.number_input(
            "Units Sold",
            min_value=1,
            max_value=1000,
            value=2,
            key="sales_units"
        )

    with col3:

        price = st.number_input(
            "Unit Price",
            min_value=1.0,
            max_value=1000000.0,
            value=1000.0,
            key="sales_price"
        )

        discount = st.number_input(
            "Discount %",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            key="sales_discount"
        )

        shipping = st.number_input(
            "Shipping Cost",
            min_value=0.0,
            max_value=100000.0,
            value=100.0,
            key="sales_shipping"
        )

        tax = st.number_input(
            "Tax %",
            min_value=0.0,
            max_value=100.0,
            value=18.0,
            key="sales_tax"
        )

        delivery = st.number_input(
            "Delivery Days",
            min_value=0,
            max_value=100,
            value=3,
            key="sales_delivery"
        )

        satisfaction = st.slider(
            "Customer Satisfaction",
            min_value=1,
            max_value=5,
            value=4,
            key="sales_satisfaction"
        )

    st.markdown("---")

    if st.button(
        "🚀 Predict Sales",
        type="primary"
    ):

        current_date = pd.Timestamp.today()

        input_data = pd.DataFrame([{

            "Region": region,

            "City": city,

            "Category": category,

            "Product": product,

            "Customer_Age": age,

            "Customer_Gender": gender,

            "Days_Since_Last_Order": days,

            "Customer_Segment": segment,

            "Sales_Channel": channel,

            "Salesperson": salesperson,

            "Payment_Method": payment,

            "Units_Sold": units,

            "Unit_Price": price,

            "Discount_Pct": discount,

            "Shipping_Cost": shipping,

            "Tax_Pct": tax,

            "Delivery_Days": delivery,

            "Customer_Satisfaction": satisfaction,

            "Order_Year": current_date.year,

            "Order_Month": current_date.month,

            "Order_DayOfWeek": current_date.dayofweek,

            "Order_Quarter": current_date.quarter,

            "Is_Weekend": int(
                current_date.dayofweek >= 5
            )

        }])

        prediction = sales_model.predict(
            input_data
        )[0]

        st.success(
            f"Predicted Order Value: ₹{prediction:,.2f}"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicted Sales",
            f"₹{prediction:,.2f}"
        )

        c2.metric(
            "Units Sold",
            units
        )

        c3.metric(
            "Revenue per Unit",
            f"₹{prediction / units:,.2f}"
        )

    st.markdown("---")

    st.subheader(
        "📊 Model Performance"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "R² Score",
        f"{r2:.3f}"
    )

    c2.metric(
        "MAE",
        f"₹{mae:,.2f}"
    )

    c3.metric(
        "RMSE",
        f"₹{rmse:,.2f}"
    )


elif page == "🔄 Return Prediction":

    st.title(
        "🔄 Return Prediction"
    )

    st.write(
        "Predict whether an order has a high probability "
        "of being returned."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        region = st.selectbox(
            "Region",
            sorted(df["Region"].unique()),
            key="ret_region"
        )

        city = st.selectbox(
            "City",
            sorted(df["City"].unique()),
            key="ret_city"
        )

        category = st.selectbox(
            "Category",
            sorted(df["Category"].unique()),
            key="ret_category"
        )

        product = st.selectbox(
            "Product",
            sorted(df["Product"].unique()),
            key="ret_product"
        )

        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=30,
            key="ret_age"
        )

        gender = st.selectbox(
            "Gender",
            sorted(df["Customer_Gender"].unique()),
            key="ret_gender"
        )

    with col2:

        days = st.number_input(
            "Days Since Last Order",
            min_value=0,
            max_value=1000,
            value=30,
            key="ret_days"
        )

        segment = st.selectbox(
            "Customer Segment",
            sorted(df["Customer_Segment"].unique()),
            key="ret_segment"
        )

        channel = st.selectbox(
            "Sales Channel",
            sorted(df["Sales_Channel"].unique()),
            key="ret_channel"
        )

        salesperson = st.selectbox(
            "Salesperson",
            sorted(df["Salesperson"].unique()),
            key="ret_salesperson"
        )

        payment = st.selectbox(
            "Payment Method",
            sorted(df["Payment_Method"].unique()),
            key="ret_payment"
        )

        units = st.number_input(
            "Units Sold",
            min_value=1,
            max_value=1000,
            value=2,
            key="ret_units"
        )

    with col3:

        price = st.number_input(
            "Unit Price",
            min_value=1.0,
            max_value=1000000.0,
            value=1000.0,
            key="ret_price"
        )

        discount = st.number_input(
            "Discount %",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            key="ret_discount"
        )

        shipping = st.number_input(
            "Shipping Cost",
            min_value=0.0,
            max_value=100000.0,
            value=100.0,
            key="ret_shipping"
        )

        tax = st.number_input(
            "Tax %",
            min_value=0.0,
            max_value=100.0,
            value=18.0,
            key="ret_tax"
        )

        delivery = st.number_input(
            "Delivery Days",
            min_value=0,
            max_value=100,
            value=3,
            key="ret_delivery"
        )

        satisfaction = st.slider(
            "Customer Satisfaction",
            min_value=1,
            max_value=5,
            value=4,
            key="ret_satisfaction"
        )

    st.markdown("---")

    if st.button(
        "🔍 Predict Return Risk",
        type="primary"
    ):

        input_data = pd.DataFrame([{

            "Region": region,

            "City": city,

            "Category": category,

            "Product": product,

            "Customer_Age": age,

            "Customer_Gender": gender,

            "Days_Since_Last_Order": days,

            "Customer_Segment": segment,

            "Sales_Channel": channel,

            "Salesperson": salesperson,

            "Payment_Method": payment,

            "Units_Sold": units,

            "Unit_Price": price,

            "Discount_Pct": discount,

            "Shipping_Cost": shipping,

            "Tax_Pct": tax,

            "Delivery_Days": delivery,

            "Customer_Satisfaction": satisfaction

        }])

        probability = return_model.predict_proba(
            input_data
        )[0][1]

        prediction = return_model.predict(
            input_data
        )[0]

        st.progress(
            float(probability)
        )

        if prediction == 1:

            st.error(
                f"⚠️ High Return Risk: "
                f"{probability * 100:.2f}%"
            )

        else:

            st.success(
                f"✅ Low Return Risk: "
                f"{probability * 100:.2f}%"
            )

    st.markdown("---")

    st.metric(
        "Return Model Accuracy",
        f"{return_accuracy * 100:.2f}%"
    )


elif page == "📋 Data Explorer":

    st.title(
        "📋 Data Explorer"
    )

    st.write(
        "Explore and filter the complete sales dataset."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        regions = st.multiselect(
            "Region",
            sorted(df["Region"].unique())
        )

    with col2:

        categories = st.multiselect(
            "Category",
            sorted(df["Category"].unique())
        )

    with col3:

        genders = st.multiselect(
            "Gender",
            sorted(df["Customer_Gender"].unique())
        )

    filtered_df = df.copy()

    if regions:

        filtered_df = filtered_df[
            filtered_df["Region"].isin(regions)
        ]

    if categories:

        filtered_df = filtered_df[
            filtered_df["Category"].isin(categories)
        ]

    if genders:

        filtered_df = filtered_df[
            filtered_df["Customer_Gender"].isin(genders)
        ]

    st.write(
        f"Showing {len(filtered_df):,} records"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=550
    )

    csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered CSV",
        data=csv,
        file_name="sales_filtered.csv",
        mime="text/csv"
    )


st.markdown("---")

st.caption(
    "Customer Sales Prediction System | "
    "Python | Scikit-Learn | Streamlit | Plotly"
)

