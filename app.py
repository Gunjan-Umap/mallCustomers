import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Intelligence ML Agent",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #07000f, #120022, #050008);
    color: white;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 4px;
}

.subtitle {
    color: #aaa1bb;
    font-size: 16px;
    margin-bottom: 25px;
}

.agent-box {
    background: linear-gradient(135deg, #18102a, #24143d);
    border: 1px solid #543777;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 22px;
}

.agent-title {
    font-size: 22px;
    font-weight: 700;
}

.agent-text {
    color: #c8c0d5;
    line-height: 1.6;
}

.metric-card {
    background: #130b20;
    border: 1px solid #38254d;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}

.metric-label {
    color: #9990aa;
    font-size: 13px;
    letter-spacing: 1px;
}

.metric-value {
    font-size: 30px;
    font-weight: 750;
    margin-top: 5px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 12px;
}

.cluster-box {
    background: #120a1e;
    border: 1px solid #39254d;
    border-radius: 16px;
    padding: 20px;
    min-height: 190px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("Mall_Customers.csv")


# =========================================================
# SELECT FEATURES
# =========================================================

X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]


# =========================================================
# K-MEANS MODEL
# =========================================================

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_


# =========================================================
# CLUSTER SUMMARY
# =========================================================

summary = df.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    Avg_Income=("Annual Income (k$)", "mean"),
    Avg_Spending=("Spending Score (1-100)", "mean")
).reset_index()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🤖 Customer Intelligence ML Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Unsupervised Machine Learning • K-Means Customer Segmentation'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# ML AGENT STATUS
# =========================================================

st.markdown("""
<div class="agent-box">

<div class="agent-title">
🧠 ML Agent Status
</div>

<div class="agent-text">

Model initialized successfully.<br>
Analyzing customer behavior using <b>Annual Income</b> and
<b>Spending Score</b>.<br>
K-Means has automatically divided the customers into
<b>2 behavioral segments</b>.

</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# KPI CARDS
# =========================================================

total_customers = len(df)

avg_income = df["Annual Income (k$)"].mean()

avg_spending = df["Spending Score (1-100)"].mean()

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-label">
    TOTAL CUSTOMERS
    </div>

    <div class="metric-value">
    {total_customers}
    </div>

    </div>
    """, unsafe_allow_html=True)


with c2:

    st.markdown("""
    <div class="metric-card">

    <div class="metric-label">
    ML CLUSTERS
    </div>

    <div class="metric-value">
    2
    </div>

    </div>
    """, unsafe_allow_html=True)


with c3:

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-label">
    AVG INCOME
    </div>

    <div class="metric-value">
    {avg_income:.1f}k
    </div>

    </div>
    """, unsafe_allow_html=True)


with c4:

    st.markdown(f"""
    <div class="metric-card">

    <div class="metric-label">
    AVG SPENDING
    </div>

    <div class="metric-value">
    {avg_spending:.1f}
    </div>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# CUSTOMER SEGMENTATION GRAPH
# =========================================================

st.markdown(
    '<div class="section-title">🎯 Customer Segmentation Map</div>',
    unsafe_allow_html=True
)


fig = go.Figure()


# ---------------------------------------------------------
# CLUSTER 0
# ---------------------------------------------------------

cluster0 = df[df["Cluster"] == 0]

fig.add_trace(
    go.Scatter(
        x=cluster0["Annual Income (k$)"],
        y=cluster0["Spending Score (1-100)"],
        mode="markers",
        name="Cluster 0",
        text=cluster0["CustomerID"],
        hovertemplate=
        "<b>Customer ID:</b> %{text}<br>" +
        "<b>Income:</b> %{x}k<br>" +
        "<b>Spending Score:</b> %{y}<extra></extra>",
        marker=dict(
            size=10
        )
    )
)


# ---------------------------------------------------------
# CLUSTER 1
# ---------------------------------------------------------

cluster1 = df[df["Cluster"] == 1]

fig.add_trace(
    go.Scatter(
        x=cluster1["Annual Income (k$)"],
        y=cluster1["Spending Score (1-100)"],
        mode="markers",
        name="Cluster 1",
        text=cluster1["CustomerID"],
        hovertemplate=
        "<b>Customer ID:</b> %{text}<br>" +
        "<b>Income:</b> %{x}k<br>" +
        "<b>Spending Score:</b> %{y}<extra></extra>",
        marker=dict(
            size=10
        )
    )
)


# ---------------------------------------------------------
# CENTROIDS
# ---------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode="markers",
        name="Centroids",
        marker=dict(
            symbol="x",
            size=22,
            line=dict(width=3)
        ),
        hovertemplate=
        "<b>Cluster Centroid</b><br>" +
        "Income: %{x:.2f}k<br>" +
        "Spending: %{y:.2f}<extra></extra>"
    )
)


fig.update_layout(
    template="plotly_dark",
    height=560,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,5,20,0)",
    title="K-Means Customer Segmentation",
    xaxis_title="Annual Income (k$)",
    yaxis_title="Spending Score (1-100)",
    legend_title="ML Segments",
    hovermode="closest"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# ML AGENT INTERPRETATION
# =========================================================

st.markdown(
    '<div class="section-title">🧠 ML Agent Analysis</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


for position, column in enumerate([col1, col2]):

    cluster_id = int(summary.iloc[position]["Cluster"])

    customers = int(
        summary.iloc[position]["Customers"]
    )

    income = summary.iloc[position]["Avg_Income"]

    spending = summary.iloc[position]["Avg_Spending"]


    # Determine profile
    if income >= avg_income and spending >= avg_spending:

        profile = "High Income • High Spending"

        insight = (
            "Customers in this segment show both "
            "higher income and higher spending behavior."
        )

    elif income >= avg_income and spending < avg_spending:

        profile = "High Income • Lower Spending"

        insight = (
            "Customers have relatively higher income "
            "but comparatively lower spending behavior."
        )

    elif income < avg_income and spending >= avg_spending:

        profile = "Lower Income • High Spending"

        insight = (
            "Customers have comparatively lower income "
            "but show stronger spending behavior."
        )

    else:

        profile = "Lower Income • Lower Spending"

        insight = (
            "Customers show comparatively lower income "
            "and lower spending behavior."
        )


    with column:

        st.markdown(f"""
        <div class="cluster-box">

        <h3>Cluster {cluster_id}</h3>

        <p>
        <b>Behavior Profile:</b><br>
        {profile}
        </p>

        <p>
        <b>Customers:</b> {customers}
        </p>

        <p>
        <b>Average Income:</b> {income:.2f} k$
        </p>

        <p>
        <b>Average Spending:</b> {spending:.2f}
        </p>

        <p style="color:#aaa1bb;">
        🧠 <b>Agent Insight:</b> {insight}
        </p>

        </div>
        """, unsafe_allow_html=True)


# =========================================================
# CLUSTER DISTRIBUTION
# =========================================================

st.markdown(
    '<div class="section-title">📊 Cluster Distribution</div>',
    unsafe_allow_html=True
)


fig2 = go.Figure()


fig2.add_trace(
    go.Bar(
        x=summary["Cluster"].astype(str),
        y=summary["Customers"],
        text=summary["Customers"],
        textposition="auto",
        name="Customers"
    )
)


fig2.update_layout(
    template="plotly_dark",
    height=380,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Cluster",
    yaxis_title="Number of Customers"
)


st.plotly_chart(
    fig2,
    use_container_width=True
)


# =========================================================
# CENTROIDS
# =========================================================

st.markdown(
    '<div class="section-title">📍 Cluster Centroids</div>',
    unsafe_allow_html=True
)


centroid_df = pd.DataFrame(
    centroids,
    columns=[
        "Annual Income (k$)",
        "Spending Score"
    ]
)

centroid_df.index.name = "Cluster"


st.dataframe(
    centroid_df.round(2),
    use_container_width=True
)


# =========================================================
# CUSTOMER EXPLORER
# =========================================================

st.markdown(
    '<div class="section-title">🔍 Customer Explorer</div>',
    unsafe_allow_html=True
)


selected_cluster = st.selectbox(
    "Select customer segment",
    ["All", 0, 1]
)


if selected_cluster == "All":

    display_df = df

else:

    display_df = df[
        df["Cluster"] == selected_cluster
    ]


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# DOWNLOAD DATA
# =========================================================

csv_data = df.to_csv(index=False)


st.download_button(
    label="⬇️ Download Clustered Customer Data",
    data=csv_data,
    file_name="Customer_Segmentation_Result.csv",
    mime="text/csv"
)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">⚙️ ML Model Information</div>',
    unsafe_allow_html=True
)


st.markdown("""
<div class="agent-box">

<b>Algorithm:</b> K-Means Clustering<br>
<b>Learning Type:</b> Unsupervised Learning<br>
<b>Number of Clusters:</b> 2<br>
<b>Features Used:</b> Annual Income + Spending Score<br>
<b>Centroids:</b> Calculated automatically<br>
<b>Model Status:</b> Active

</div>
""", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div style="
text-align:center;
color:#777;
padding:30px;
">

🤖 Customer Intelligence ML Agent<br>
K-Means Machine Learning Mini Project

</div>
""", unsafe_allow_html=True)

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Mall_Customers.csv")

df = pd.read_csv(csv_path)