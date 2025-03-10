import streamlit as st
import plotly.express as px

def display_anomaly_dashboard(df, plot_theme):
    st.header("🧠 Anomaly Intelligence Dashboard")

    if "Anomaly Type" not in df.columns or "Severity" not in df.columns:
        st.warning("No enriched anomaly data found. Run anomaly detection first.")
        return

    view_mode = st.radio("📋 Select View Mode", ["Summary View", "Table View"], horizontal=True)

    # Filter Controls
    with st.expander("🎛️ Filter & Sort Options", expanded=False):
        types = df["Anomaly Type"].unique().tolist()
        selected_types = st.multiselect("Filter by Anomaly Type", types, default=types)
        sort_by = st.selectbox("Sort Anomalies By", ["Date", "Severity"], index=1)

    filtered = df[df["Anomaly Type"].isin(selected_types)]

    if sort_by == "Date":
        filtered = filtered.sort_values(by="Date", ascending=False)
    else:
        filtered = filtered.sort_values(by="Severity", ascending=False)

    # Smart Cluster Detection
    st.markdown("### 🔍 High-Risk Anomaly Clusters")
    filtered["Cluster"] = (filtered["Date"].diff().dt.days > 3).cumsum()
    cluster_counts = filtered.groupby("Cluster").size().reset_index(name="Anomaly Count")
    high_clusters = cluster_counts[cluster_counts["Anomaly Count"] > 2]

    if not high_clusters.empty:
        for idx, row in high_clusters.iterrows():
            cluster_df = filtered[filtered["Cluster"] == row["Cluster"]]
            start = cluster_df["Date"].min().strftime('%Y-%m-%d')
            end = cluster_df["Date"].max().strftime('%Y-%m-%d')
            st.markdown(f"⚠️ **{row['Anomaly Count']} anomalies detected between {start} and {end}**")

    # Summary View
    if view_mode == "Summary View":
        st.subheader("📌 Anomaly Overview")

        total_anomalies = len(filtered)
        most_common = filtered["Anomaly Type"].value_counts().idxmax()
        top_day = filtered.iloc[0]["Date"]

        col1, col2, col3 = st.columns(3)
        col1.metric("🔍 Total Anomalies", total_anomalies)
        col2.metric("📊 Most Frequent", most_common)
        col3.metric("📅 Most Recent", top_day.strftime("%Y-%m-%d"))

        st.subheader("📈 Anomaly Frequency Timeline")
        daily = filtered.groupby("Date")["Anomaly Type"].count().reset_index(name="Anomaly Count")
        fig = px.line(daily, x="Date", y="Anomaly Count", title="Anomaly Frequency", template=plot_theme)
        st.plotly_chart(fig, use_container_width=True)

    # Table View
    else:
        st.subheader("📋 Anomaly Description Table")

        def describe(row):
            base = f"{row['Anomaly Type']} detected on {row['Date'].strftime('%Y-%m-%d')}:"
            if row['Anomaly Type'] in ["Drop", "Critical Drop"]:
                return f"{base} Sales dropped sharply. 🔍 Check for holiday, promo end, or operational delays."
            elif row['Anomaly Type'] in ["Spike", "Critical Spike"]:
                return f"{base} Sudden spike detected. 🔍 Investigate traffic, influencer push, or flash sale."
            else:
                return f"{base} Minor fluctuation."

        filtered["Suggested Action"] = filtered.apply(describe, axis=1)

        st.dataframe(
            filtered[["Date", "Sales", "Costs", "Profit", "Anomaly Type", "Severity", "Suggested Action"]],
            use_container_width=True
        )

        # Export Option
        st.download_button(
            label="📥 Export to CSV",
            data=filtered.to_csv(index=False).encode('utf-8'),
            file_name="anomaly_report.csv",
            mime="text/csv"
        )

    st.caption("🔎 Tip: Review clustered anomalies or high-severity entries for immediate resolution.")
