import io
import pandas as pd
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection Engine",
    page_icon="",
    layout="wide",
)

# Render API Endpoint
API_URL = "https://credit-card-fraud-detection-vb2x.onrender.com/predict"

st.title(" Credit Card Fraud Detection Center")
st.markdown(
    "Real-time and batch machine learning inference engine powered by FastAPI & ClearML."
)
st.divider()

# Tab Navigation: Real-Time vs Batch
tab_realtime, tab_batch = st.tabs(
    [" Real-Time Single Analysis", " Batch CSV Processing"]
)

# REAL-TIME SINGLE ANALYSIS
with tab_realtime:
    st.sidebar.header("Transaction Parameters")
    st.sidebar.markdown("Adjust transaction feature values below:")

    amount = st.sidebar.number_input(
        "Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0
    )
    time_step = st.sidebar.number_input(
        "Time Elapsed (seconds)", min_value=0.0, value=0.0, step=1.0
    )

    st.sidebar.subheader("PCA Features (V1 - V28)")
    pcas = []
    for i in range(1, 29):
        val = st.sidebar.slider(
            f"Feature V{i}", -15.0, 15.0, 0.0, step=0.1, key=f"v{i}"
        )
        pcas.append(val)

    # Construct 30-feature vector: [Time, V1..V28, Amount]
    features_vector = [time_step] + pcas + [amount]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Payload Preview")
        payload_df = pd.DataFrame(
            [features_vector],
            columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"],
        )
        st.dataframe(payload_df.T, height=400, use_container_width=True)

    with col2:
        st.subheader(" Inference Engine Status")
        st.info(f"Target Service Endpoint:\n`{API_URL}`")

        if st.button(
            "Analyze Single Transaction",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Streaming payload to Render service..."):
                try:
                    response = requests.post(
                        API_URL, json={"features": features_vector}, timeout=10
                    )

                    if response.status_code == 200:
                        result = response.json()
                        is_fraud = result.get("is_fraud", 0)
                        fraud_prob = result.get("fraud_probability", 0.0)

                        st.markdown("### Decision Result")
                        if is_fraud == 1:
                            st.error(" **FRAUD DETECTED**\n\nHigh risk flagged.")
                        else:
                            st.success(
                                " **TRANSACTION APPROVED**\n\nNormal profile."
                            )

                        m1, m2 = st.columns(2)
                        m1.metric(
                            "Fraud Probability", f"{fraud_prob * 100:.2f}%"
                        )
                        m2.metric(
                            "Risk Classification",
                            "High" if is_fraud == 1 else "Low",
                        )

                        st.progress(fraud_prob)

                    else:
                        st.error(
                            f"API Error: Status code {response.status_code}"
                        )
                        st.json(response.json())

                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to API service: {e}")

# BATCH CSV PROCESSING
with tab_batch:
    st.subheader("Upload Batch Dataset for Fraud Assessment")
    st.markdown(
        "Upload a `.csv` file containing the 30 numerical features (`Time`, `V1`..`V28`, `Amount`)."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"**Loaded Batch:** {batch_df.shape[0]} rows × {batch_df.shape[1]} columns")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("Process Batch Predictions", type="primary"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                total_rows = len(batch_df)

                for idx, row in batch_df.iterrows():
                    # Extract raw feature list
                    row_features = row.tolist()

                    try:
                        resp = requests.post(
                            API_URL, json={"features": row_features}, timeout=5
                        )
                        if resp.status_code == 200:
                            res = resp.json()
                            results.append(
                                {
                                    "is_fraud": res.get("is_fraud", 0),
                                    "fraud_probability": res.get(
                                        "fraud_probability", 0.0
                                    ),
                                }
                            )
                        else:
                            results.append(
                                {"is_fraud": None, "fraud_probability": None}
                            )
                    except Exception:
                        results.append(
                            {"is_fraud": None, "fraud_probability": None}
                        )

                    # Update progress
                    progress_val = (idx + 1) / total_rows
                    progress_bar.progress(progress_val)
                    status_text.text(
                        f"Evaluated {idx + 1} of {total_rows} transactions..."
                    )

                status_text.success("Batch Processing Complete!")

                # Combine results with original data
                res_df = pd.DataFrame(results)
                output_df = pd.concat([batch_df, res_df], axis=1)

                st.subheader(" Batch Inference Results")
                st.dataframe(output_df, use_container_width=True)

                # Summary Statistics
                fraud_count = (output_df["is_fraud"] == 1).sum()
                approved_count = (output_df["is_fraud"] == 0).sum()

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Processed", total_rows)
                c2.metric("Flagged Fraudulent", fraud_count)
                c3.metric("Approved Transactions", approved_count)

                # CSV Download Button
                csv_buffer = io.BytesIO()
                output_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label=" Download Predictions CSV",
                    data=csv_buffer.getvalue(),
                    file_name="fraud_batch_predictions.csv",
                    mime="text/csv",
                )

        except Exception as err:
            st.error(f"Error reading or processing CSV file: {err}")