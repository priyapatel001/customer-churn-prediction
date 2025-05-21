


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go  # Changed from plotly.express
from PIL import Image
import time
from sklearn.inspection import permutation_importance
from lime import lime_tabular
import eli5
from eli5.sklearn import PermutationImportance
import altair as alt
import plotly.express as px  # Add this line
import plotly.graph_objects as go  # Keep this line


# Set page config
st.set_page_config(
    page_title="🔮 ChurnGuard Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .st-b7 {
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .risk-low {
        color: #2ecc71;
        font-weight: bold;
    }
    .feature-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .lime-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load('customer_churn_model.joblib')
        encoders = joblib.load('encoders.joblib')
        return model_data['model'], model_data['features_names'], encoders
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, feature_names, encoders = load_model()

# Prediction function
def predict_churn(input_data):
    try:
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical features
        for column, encoder in encoders.items():
            if column in input_df.columns:
                input_df[column] = encoder.transform(input_df[column])
        
        # Ensure all features are present
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[feature_names]
        
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)
        
        return prediction[0], proba[0], input_df
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

# Risk assessment
def get_risk_level(probability):
    if probability >= 0.7:
        return "high", "#ff4b4b"
    elif probability >= 0.4:
        return "medium", "#ffa500"
    else:
        return "low", "#2ecc71"

# LIME explanation
def lime_explanation(model, input_df, feature_names):
    try:
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.zeros((1, len(feature_names))),  # Dummy data
            feature_names=feature_names,
            class_names=['No Churn', 'Churn'],
            mode='classification'
        )
        
        # Explain the instance
        exp = explainer.explain_instance(
            input_df.values[0], 
            model.predict_proba,
            num_features=len(feature_names))
        
        return exp
    except Exception as e:
        st.error(f"LIME explanation failed: {e}")
        return None

# Permutation importance
def permutation_importance_analysis(model, input_df):
    try:
        # Calculate permutation importance
        perm = PermutationImportance(model, random_state=42).fit(input_df, np.array([1]))
        return perm
    except Exception as e:
        st.error(f"Permutation importance failed: {e}")
        return None

# Main app
def main():
    st.title("🔮 ChurnGuard Pro")
    st.markdown("""
    **Predict customer churn risk in real-time** and get actionable insights to improve retention.
    """)
    
    # Sidebar with company logo and info
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Your+Logo", width=150)
        st.markdown("### Customer Intelligence Dashboard")
        st.markdown("""
        - **Real-time** churn prediction
        - **Explainable AI** insights
        - **Personalized** retention strategies
        """)
        
        st.markdown("---")
        st.markdown("**How to use:**")
        st.markdown("1. Fill in customer details")
        st.markdown("2. Click 'Predict Churn Risk'")
        st.markdown("3. View insights and recommendations")
        
        st.markdown("---")
        st.markdown("**Model Info:**")
        st.markdown(f"- Algorithm: {model.__class__.__name__ if model else 'N/A'}")
        st.markdown("- Version: 1.3.0")
        st.markdown("- Last updated: Today")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Customer Profile")
        
        with st.expander("Demographic Information", expanded=True):
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.radio("Senior Citizen", ["No", "Yes"])
            partner = st.radio("Partner", ["No", "Yes"])
            dependents = st.radio("Dependents", ["No", "Yes"])
        
        with st.expander("Service Details", expanded=True):
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            contract = st.select_slider("Contract Type", 
                                      options=["Month-to-month", "One year", "Two year"],
                                      value="Month-to-month")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 150.0, 70.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, tenure * monthly_charges)
    
    with col2:
        st.header("📊 Service Features")
        
        with st.expander("Phone Services", expanded=True):
            phone_service = st.radio("Phone Service", ["No", "Yes"])
            if phone_service == "Yes":
                multiple_lines = st.radio("Multiple Lines", ["No", "Yes"])
            else:
                multiple_lines = "No phone service"
        
        with st.expander("Additional Services", expanded=True):
            online_security = st.radio("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.radio("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.radio("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.radio("Tech Support", ["No", "Yes", "No internet service"])
        
        with st.expander("Streaming Services", expanded=True):
            streaming_tv = st.radio("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.radio("Streaming Movies", ["No", "Yes", "No internet service"])
        
        with st.expander("Billing Information", expanded=True):
            paperless_billing = st.radio("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", 
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
    
    # Prediction button
    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        if model is None:
            st.error("Model not loaded. Please check the model files.")
            return
            
        with st.spinner("Analyzing customer profile..."):
            time.sleep(1)  # Simulate processing time
            
            input_data = {
                'gender': gender,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            prediction, proba, input_df = predict_churn(input_data)
            
            if prediction is not None:
                churn_prob = proba[1] * 100
                risk_level, risk_color = get_risk_level(proba[1])
                
                # Results container
                with st.container():
                    st.markdown("---")
                    col_res1, col_res2 = st.columns([1, 3])
                    
                    with col_res1:
                        st.markdown(f"### 🎯 Churn Risk: <span class='risk-{risk_level}'>{risk_level.upper()}</span>", 
                                   unsafe_allow_html=True)
                        st.metric("Probability", f"{churn_prob:.1f}%")
                        
                        # Gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=churn_prob,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title="Churn Risk Level",
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': risk_color},
                                'steps': [
                                    {'range': [0, 40], 'color': "#2ecc71"},
                                    {'range': [40, 70], 'color': "#ffa500"},
                                    {'range': [70, 100], 'color': "#ff4b4b"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_res2:
                        st.markdown("### 🔍 Explanation")
                        
                        # Explanation tabs
                        tab1, tab2, tab3 = st.tabs(["Feature Importance", "LIME Explanation", "What-If Analysis"])
                        
                        with tab1:
                            st.markdown("#### Permutation Feature Importance")
                            perm = permutation_importance_analysis(model, input_df)
                            if perm:
                                # Create feature importance dataframe
                                feat_imp_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': perm.feature_importances_
                                }).sort_values('Importance', ascending=False).head(10)
                                
                                # Create Altair chart
                                chart = alt.Chart(feat_imp_df).mark_bar().encode(
                                    x='Importance:Q',
                                    y=alt.Y('Feature:N', sort='-x'),
                                    color=alt.Color('Importance:Q', scale=alt.Scale(scheme='redyellowgreen')))
                                
                                st.altair_chart(chart, use_container_width=True)
                                
                                # Show top features
                                st.markdown("**Key Influencing Factors:**")
                                for _, row in feat_imp_df.head(3).iterrows():
                                    with st.expander(f"{row['Feature']} (Impact: {row['Importance']:.3f})"):
                                        st.markdown(f"**Current value:** {input_data.get(row['Feature'], 'N/A')}")
                                        st.markdown("**Recommendation:**")
                                        if row['Feature'] == 'Contract' and input_data['Contract'] == 'Month-to-month':
                                            st.markdown("👉 Offer discount for annual contract")
                                        elif row['Feature'] == 'tenure' and input_data['tenure'] < 12:
                                            st.markdown("👉 Provide loyalty bonus for long-term customers")
                                        elif row['Feature'] == 'MonthlyCharges' and input_data['MonthlyCharges'] > 80:
                                            st.markdown("👉 Consider personalized discount or value-added services")
                        
                        with tab2:
                            st.markdown("#### Local Interpretable Model-agnostic Explanations (LIME)")
                            lime_exp = lime_explanation(model, input_df, feature_names)
                            if lime_exp:
                                # Display LIME explanation
                                st.markdown("**How each feature contributes to the prediction:**")
                                
                                # Get explanation data
                                lime_data = [(x[0], x[1]) for x in lime_exp.as_list()]
                                lime_df = pd.DataFrame(lime_data, columns=['Feature', 'Effect'])
                                
                                # Create bar chart
                                fig = px.bar(lime_df, 
                                            x='Effect', 
                                            y='Feature', 
                                            orientation='h',
                                            color='Effect',
                                            color_continuous_scale='RdYlGn',
                                            title='Feature Contributions to Prediction')
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show explanation text
                                st.markdown("**Explanation:**")
                                for feature, effect in lime_data:
                                    if effect > 0:
                                        st.markdown(f"- {feature} increases churn risk")
                                    else:
                                        st.markdown(f"- {feature} decreases churn risk")
                            else:
                                st.warning("Could not generate LIME explanation")
                        
                        with tab3:
                            st.markdown("#### What-If Scenario Analysis")
                            st.markdown("See how changing features affects churn risk")
                            
                            # Create sliders for what-if analysis
                            col1, col2 = st.columns(2)
                            with col1:
                                new_tenure = st.slider("Change Tenure (months)", 0, 72, tenure)
                                new_contract = st.selectbox("Change Contract", 
                                                          ["Month-to-month", "One year", "Two year"],
                                                          index=["Month-to-month", "One year", "Two year"].index(contract))
                            with col2:
                                new_monthly = st.slider("Change Monthly Charges ($)", 20.0, 150.0, monthly_charges)
                                new_payment = st.selectbox("Change Payment Method",
                                                          ["Electronic check", "Mailed check", 
                                                           "Bank transfer (automatic)", "Credit card (automatic)"],
                                                          index=["Electronic check", "Mailed check", 
                                                                 "Bank transfer (automatic)", "Credit card (automatic)"].index(payment_method))
                            
                            if st.button("Simulate Changes"):
                                # Create new input data
                                new_input_data = input_data.copy()
                                new_input_data.update({
                                    'tenure': new_tenure,
                                    'Contract': new_contract,
                                    'MonthlyCharges': new_monthly,
                                    'PaymentMethod': new_payment
                                })
                                
                                # Make new prediction
                                new_prediction, new_proba, _ = predict_churn(new_input_data)
                                if new_prediction is not None:
                                    new_churn_prob = new_proba[1] * 100
                                    delta = new_churn_prob - churn_prob
                                    
                                    st.metric("New Churn Probability", 
                                             f"{new_churn_prob:.1f}%", 
                                             f"{delta:+.1f}%",
                                             delta_color="inverse")
                                    
                                    if delta < 0:
                                        st.success("These changes would reduce churn risk!")
                                    elif delta > 0:
                                        st.warning("These changes would increase churn risk!")
                                    else:
                                        st.info("These changes would not affect churn risk")
                        
                        # Actionable recommendations
                        st.markdown("#### 🚀 Retention Strategies")
                        if risk_level == "high":
                            st.error("**Immediate action required!**")
                            st.markdown("""
                            - Personal outreach from account manager
                            - Customized retention offer (15-20% discount)
                            - Free premium service for 3 months
                            """)
                        elif risk_level == "medium":
                            st.warning("**Proactive measures recommended**")
                            st.markdown("""
                            - Targeted email campaign
                            - 10% discount on next bill
                            - Free service upgrade for 1 month
                            """)
                        else:
                            st.success("**Low risk - maintain engagement**")
                            st.markdown("""
                            - Regular satisfaction check-ins
                            - Early renewal bonus offer
                            - Referral program promotion
                            """)
                
                # Customer profile comparison
                st.markdown("---")
                st.markdown("### 📊 Similar Customer Profiles")
                
                # Simulated data - in real app you'd use your actual data
                similar_profiles = pd.DataFrame({
                    'Tenure': [tenure, 24, 6, 36],
                    'Monthly Charges': [monthly_charges, 65.0, 90.0, 45.0],
                    'Contract': [contract, "One year", "Month-to-month", "Two year"],
                    'Churn Risk': [f"{churn_prob:.1f}%", "22.5%", "78.3%", "15.2%"]
                })
                
                st.dataframe(
                    similar_profiles.style.highlight_max(axis=0, color='#fffd75'),
                    use_container_width=True
                )

if __name__ == "__main__":
    main()