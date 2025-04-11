import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ✅ Load your trained model
model_path = model_path = "models/best_xgb_model.pkl"
model = joblib.load(model_path)

# ✅ Title & intro
st.title("🚀 Automated A/B Testing Optimizer")
st.markdown("Upload your website user data to predict purchase behavior using a trained AI model.")

# ✅ File uploader
uploaded_file = st.file_uploader("📁 Upload a CSV file (use '|' as delimiter)", type=["csv"])

if uploaded_file is not None:
    try:
        # ✅ Read the uploaded file with correct delimiter
        df = pd.read_csv(uploaded_file, sep='|')
        st.success("✅ File uploaded successfully!")

        # ✅ Strip column names
        df.columns = df.columns.str.strip()

        # ✅ Drop unnecessary columns
        drop_cols = ['pt_d', 'id']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        # ✅ Handle object columns
        non_numeric_cols = df.select_dtypes(include=['object']).columns

        for col in non_numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass

        le = LabelEncoder()
        for col in non_numeric_cols:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col].astype(str))

        # ✅ Drop rows with NaN values
        df = df.dropna()

        # ✅ Remove target if present
        if 'consume_purchase' in df.columns:
            df = df.drop(columns=['consume_purchase'])

        # ✅ Assign A/B groups AFTER cleaning to match DataFrame size
        if 'uid' in df.columns:
            group_series = df['uid'].apply(lambda x: 'A' if hash(x) % 2 == 0 else 'B')
        else:
            group_series = pd.Series(['Unknown'] * len(df))
            st.warning("⚠️ 'uid' column not found — assigning 'Unknown' to all groups.")

        # ✅ Prepare input for prediction (exclude group column)
        model_input = df.copy()

        # ✅ Make predictions
        predictions = model.predict(model_input)
        prediction_probs = model.predict_proba(model_input)

        # ✅ Show predictions with confidence and group
        st.subheader("✅ Predictions")
        pred_df = pd.DataFrame({
            "Prediction": predictions,
            "Confidence (%)": (np.max(prediction_probs, axis=1) * 100).round(2),
            "Group": group_series.values
        })
        st.write(pred_df)

        # ✅ Show group-wise conversion rate (optional metric)
        if 'Group' in pred_df.columns:
            st.subheader("📊 Group-wise Purchase Prediction Rate")
            conversion = pred_df.groupby("Group")["Prediction"].mean().reset_index()
            conversion.columns = ['Group', 'Predicted Purchase Rate']
            conversion['Predicted Purchase Rate (%)'] = (conversion['Predicted Purchase Rate'] * 100).round(2)
            st.write(conversion)

        # ✅ Show feature importances
        show_importance = st.checkbox("📌 Show Top 10 Feature Importances")
        if show_importance:
            importance_df = pd.DataFrame({
                'Feature': model_input.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            st.write(importance_df.head(10))

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")

else:
    st.info("Please upload a file to begin.")
