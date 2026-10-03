import pandas as pd
import streamlit as st

df = pd.read_csv("customers_segmented.csv")

st.title("Customer Segmentation Dashboard")
st.write("Total customers:", len(df))

st.bar_chart(df["segment"].value_counts())

features = ["data_gb", "voice_min", "sms_count",
            "monthly_spend_da", "topup_count", "tenure_months"]

st.subheader("Average profile of each segment")
profile = df.groupby("segment")[features].mean().round(1)
st.dataframe(profile)

actions = {
    "Data lovers": "Offer bigger data bundles (30 GB or more) and night-time data packs.",
    "Talkers": "Offer unlimited-call plans and bonus minutes on top-ups.",
    "Low spenders": "Send small, cheap top-up promotions to encourage them to recharge more.",
    "Premium": "Offer loyalty rewards, roaming packs and priority support to keep them.",
    "Students": "Offer social media packs and cheap night data for young users.",
}

st.subheader("Choose a segment")
choice = st.selectbox("Segment", list(actions.keys()))

st.write("Customers in this segment:", len(df[df["segment"] == choice]))
st.info("Suggested action: " + actions[choice])

st.subheader("Customer lookup")
customer_id = st.text_input("Enter a customer ID", "C00004")

result = df[df["customer_id"] == customer_id.strip().upper()]

if len(result) == 0:
    st.warning("Customer not found")
else:
    row = result.iloc[0]
    st.success("This customer is in segment: " + row["segment"])
    st.dataframe(result[["customer_id", "wilaya", "plan_type"] + features])
    st.dataframe(result[["customer_id", "wilaya", "plan_type"] + features])
    customer_values = result[features].iloc[0]
    segment_avg = df[df["segment"] == row["segment"]][features].mean()
    percent = (customer_values / segment_avg * 100).round(0)
    st.write("Compared with the segment average (100 = typical):")
    st.bar_chart(percent)
    st.dataframe(result[["customer_id", "wilaya", "plan_type"] + features])
    customer_values = result[features].iloc[0]
    segment_avg = df[df["segment"] == row["segment"]][features].mean()
    percent = (customer_values / segment_avg * 100).round(0)

    st.write("Compared with the segment average (100 = typical):")
    st.bar_chart(percent)

st.subheader("Segments by wilaya")
wilaya = st.selectbox("Wilaya", sorted(df["wilaya"].unique()))

wilaya_data = df[df["wilaya"] == wilaya]
st.write("Customers in", wilaya, ":", len(wilaya_data))
st.bar_chart(wilaya_data["segment"].value_counts())