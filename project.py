import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import joblib
import os
import cv2
import datetime
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ====================== PAGE CONFIG (CODE 1) ======================
st.set_page_config(
    page_title="BankAI Sentinel", 
    page_icon="🔒", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔒 BankAI Sentinel")
st.markdown("### AI-Powered Fraud Detection, Customer Insights & Credit Risk Engine")

# ====================== LOAD DATA (CODE 1) ======================
@st.cache_data
def load_bank_data():
    try:
        df = pd.read_csv('./bank_transactions.csv')
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
        df['Hour'] = df['TransactionDate'].dt.hour
        return df
    except:
        n = 1000
        df = pd.DataFrame({
            'AccountID': np.random.randint(1000, 2000, n),
            'TransactionAmount': np.random.uniform(100, 50000, n),
            'TransactionDate': pd.date_range(start='2026-01-01', periods=n, freq='H'),
            'LoginAttempts': np.random.randint(1, 6, n),
            'Channel': np.random.choice(['Mobile', 'Web', 'ATM'], n),
            'CustomerAge': np.random.randint(18, 75, n),
            'TransactionDuration': np.random.randint(10, 500, n)
        })
        df['Hour'] = df['TransactionDate'].dt.hour
        return df

@st.cache_data
def load_credit_data():
    try:
        df_credit = pd.read_excel("./credit_score_20000.xlsx", sheet_name="Credit_Data")
        if len(df_credit) > 12000:
            df_credit = df_credit.sample(8000, random_state=42).reset_index(drop=True)
        return df_credit
    except:
        np.random.seed(42)
        n = 8000
        df_credit = pd.DataFrame({
            'age': np.random.randint(22, 65, n),
            'annual_income': np.random.randint(15000, 250000, n),
            'employment_years': np.random.randint(0, 35, n),
            'home_ownership': np.random.choice(['RENT', 'MORTGAGE', 'OWN', 'OTHER'], n),
            'loan_amount': np.random.randint(5000, 450000, n),
            'loan_term_months': np.random            .choice([12, 24, 36, 48, 60, 120, 180, 240, 360], n),
            'interest_rate': np.round(np.random.uniform(5.0, 28.0, n), 2),
            'purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 
                                       'education', 'medical', 'major_purchase', 'other'], n),
            'num_late_payments': np.random.randint(0, 12, n),
            'num_bankruptcies': np.random.randint(0, 4, n),
            'credit_utilization_pct': np.round(np.random.uniform(5, 98, n), 1),
            'default': np.random.choice([0, 1], n, p=[0.78, 0.22])
        })
        return df_credit

df_bank = load_bank_data()
df_credit = load_credit_data()

# ====================== MODELS (CODE 1 & 2) ======================
@st.cache_resource
def get_credit_model():
    model_path = "crediguard_model.pkl"
    if os.path.exists(model_path):
        try: return joblib.load(model_path)
        except: pass
    data = df_credit.copy()
    data['loan_to_income'] = data['loan_amount'] / data['annual_income'].replace(0, 1)
    encoders = {}
    for col in ['home_ownership', 'purpose']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le
    feature_cols = ['age', 'annual_income', 'employment_years', 'home_ownership',
                    'loan_amount', 'loan_term_months', 'interest_rate', 'purpose',
                    'num_late_payments', 'num_bankruptcies', 'credit_utilization_pct',
                    'loan_to_income']
    X, y = data[feature_cols], data['default']
    model = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=5, 
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X, y)
    joblib.dump((model, encoders, feature_cols), model_path)
    return model, encoders, feature_cols

@st.cache_resource
def train_gb_model():
    np.random.seed(42)
    X = np.random.rand(2000, 3)
    X[:, 0] = X[:, 0] * 550 + 300 
    X[:, 1] = X[:, 1] * 0.8       
    X[:, 2] = X[:, 2] * 62 + 18    
    prob = 1 / (1 + np.exp((X[:, 0] - 620) / 60)) + (X[:, 1] * 0.4)
    y = np.random.binomial(1, np.clip(prob, 0, 1))
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05)
    model.fit(X, y)
    return model

model, encoders, feature_cols = get_credit_model()
gb_model = train_gb_model()

# ====================== SIDEBAR NAVIGATION ======================
page = st.sidebar.radio("Select Module", [
    "🏠 Home",
    "📊 Fraud Detection",
    "👥 Cust. Segmentation",
    "🔴 Live Fraud Detection",
    "🛡️ Credit Risk Scoring",
    "📸 KYC Selfie Verification",
    "🧪 Dynamic Risk Pipeline",
    "🕵️ AML Surveillance",
    "🧠 Sentiment Intelligence",
    "📈 Market Forecast",
    "🏢 InsurTech & Final Pitch",  # <--- Add this line
    "⚖️ Model Governance"
])

# ====================== 1. HOME (CODE 1) ======================
if page == "🏠 Home":
    st.header("Welcome to BankAI Sentinel")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Transactions", f"{len(df_bank):,}")
    with col2: st.metric("Unique Accounts", df_bank['AccountID'].nunique())
    with col3: st.metric("Avg Transaction", f"₹{df_bank['TransactionAmount'].mean():.2f}")
    with col4: st.metric("Default Rate", f"{df_credit['default'].mean()*100:.1f}%")

# ====================== 2. FRAUD DETECTION (CODE 1) ======================
elif page == "📊 Fraud Detection":
    st.header("📊 AI Fraud Detection System")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Avg Transaction", f"₹{df_bank['TransactionAmount'].mean():.2f}")
    with col2: st.metric("High Login Attempts (>3)", len(df_bank[df_bank['LoginAttempts'] > 3]))
    with col3: st.metric("Most Active Hour", f"{int(df_bank['Hour'].mode()[0]):02d}:00")
    fig = px.histogram(df_bank, x="TransactionAmount", color="Channel", nbins=50,
                      title="Transaction Amount Distribution by Channel")
    st.plotly_chart(fig, use_container_width=True)

# ====================== 3. CUSTOMER SEGMENTATION (CODE 1) ======================
elif page == "👥 Cust. Segmentation":
    st.header("👥 Customer Segmentation")
    try:
        customer_features = df_bank.groupby('AccountID').agg({
            'TransactionAmount': ['mean', 'sum', 'count'],
            'LoginAttempts': 'max',
            'CustomerAge': 'first',
            'TransactionDuration': 'mean'
        }).reset_index()
        customer_features.columns = ['AccountID', 'AvgAmount', 'TotalSpend', 'TransactionCount', 'MaxLoginAttempts', 'Age', 'AvgDuration']
        customer_features = customer_features.fillna(0)
        clustering_features = customer_features[['AvgAmount', 'TotalSpend', 'TransactionCount', 'MaxLoginAttempts', 'Age', 'AvgDuration']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(clustering_features)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
        customer_features['Cluster'] = kmeans.fit_predict(scaled_features).astype(str)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", len(customer_features))
        c2.metric("Avg Spend", f"₹{customer_features['TotalSpend'].mean():,.0f}")
        c3.metric("Avg Transactions", f"{customer_features['TransactionCount'].mean():.1f}")
        c4.metric("Highest Spending Cluster", customer_features.groupby('Cluster')['TotalSpend'].mean().idxmax())

        fig = px.scatter(customer_features, x="TotalSpend", y="AvgAmount", color="Cluster", size="TransactionCount",
            hover_data=['AccountID', 'Age', 'MaxLoginAttempts', 'AvgDuration'],
            title="Customer Segmentation Analysis", template="plotly_dark", height=650)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Cluster Summary")
        cluster_summary = customer_features.groupby('Cluster').agg({'TotalSpend': 'mean', 'AvgAmount': 'mean', 'TransactionCount': 'mean', 'Age': 'mean'}).round(2)
        st.dataframe(cluster_summary, use_container_width=True)
    except Exception as e: st.error(f"Error: {e}")

# ====================== 4. LIVE FRAUD DETECTION (CODE 1) ======================
elif page == "🔴 Live Fraud Detection":
    st.header("🔴 Live Fraud Detection Simulator")
    max_amount = float(df_bank['TransactionAmount'].max())
    default_amount = float(df_bank['TransactionAmount'].quantile(0.75))
    with st.form("fraud_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Transaction Amount (₹)", min_value=100.0, max_value=max_amount*1.5, value=default_amount)
            hour = st.slider("Hour of Transaction", 0, 23, int(df_bank['Hour'].mode()[0]))
            channel = st.selectbox("Channel", sorted(df_bank['Channel'].unique()))
        with col2:
            login_attempts = st.slider("Login Attempts", 1, 10, 1)
            duration = st.number_input("Transaction Duration (seconds)", 10, 1000, 120)
        submitted = st.form_submit_button("🔍 Analyze Transaction Risk", use_container_width=True)
    if submitted:
        base_risk = 0.15 + (0.35 if amount > df_bank['TransactionAmount'].quantile(0.9) else 0) + (0.4 if login_attempts >= 3 else 0)
        risk_prob = min(0.96, base_risk + np.random.uniform(-0.08, 0.08))
        risk_score = int(risk_prob * 100)
        st.markdown("### Risk Analysis Dashboard")
        col_v1, col_v2 = st.columns([3, 2])
        with col_v1:
            fig_hist = px.histogram(df_bank, x="TransactionAmount", nbins=60, title="Transaction Amount Distribution")
            fig_hist.add_vline(x=amount, line_dash="dash", line_color="red", annotation_text="Your Transaction")
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_v2:
            fig_gauge = go.Figure(go.Indicator(mode="gauge+number+delta", value=risk_score, title={'text': "Fraud Risk Score"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkred" if risk_score > 65 else "orange"}}))
            st.plotly_chart(fig_gauge, use_container_width=True)
        if risk_prob > 0.65: st.error(f"🚨 HIGH FRAUD RISK — {risk_prob:.1%}")
        elif risk_prob > 0.35: st.warning(f"⚠️ Suspicious — {risk_prob:.1%}")
        else: st.success(f"✅ Low Risk — {risk_prob:.1%}")

# ====================== 5. CREDIT RISK SCORING (STRICT RESTORE CODE 1) ======================
elif page == "🛡️ Credit Risk Scoring":
    st.header("🛡️ Credit Risk Scoring & Applicant Assessment")
    with st.form("applicant_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", "Yuvraj Singh")
            age = st.number_input("Age", 18, 80, 35)
            annual_income = st.number_input("Annual Income ($)", 10000, 500000, 65000)
            employment_years = st.number_input("Employment Years", 0, 50, 6)
            home_ownership = st.selectbox("Home Ownership", sorted(df_credit['home_ownership'].unique()))
        with col2:
            loan_amount = st.number_input("Loan Amount ($)", 1000, 500000, 25000)
            loan_term = st.number_input("Loan Term (months)", 12, 360, 36)
            interest_rate = st.number_input("Interest Rate (%)", 5.0, 30.0, 12.5)
            purpose = st.selectbox("Loan Purpose", sorted(df_credit['purpose'].unique()))
            num_late = st.number_input("Number of Late Payments", 0, 20, 1)
            num_bankrupt = st.number_input("Number of Bankruptcies", 0, 10, 0)
            credit_util = st.number_input("Credit Utilization (%)", 0.0, 100.0, 35.0)
        submitted = st.form_submit_button("Assess Risk & Generate Score", use_container_width=True)
    if submitted:
        try:
            input_data = pd.DataFrame([{
                'age': age, 'annual_income': annual_income, 'employment_years': employment_years,
                'home_ownership': home_ownership, 'loan_amount': loan_amount, 'loan_term_months': loan_term,
                'interest_rate': interest_rate, 'purpose': purpose, 'num_late_payments': num_late,
                'num_bankruptcies': num_bankrupt, 'credit_utilization_pct': credit_util,
                'loan_to_income': loan_amount / annual_income if annual_income > 0 else 0
            }])
            for col, encoder in encoders.items():
                val = input_data[col].astype(str).iloc[0]
                input_data[col] = encoder.transform([val])[0] if val in encoder.classes_ else 0
            prob_default = model.predict_proba(input_data[feature_cols])[0][1]
            credit_score = max(300, int(850 - (prob_default * 550)))
            decision = "✅ APPROVE" if prob_default < 0.15 else "⚠️ REVIEW" if prob_default < 0.35 else "❌ DECLINE"
            st.success(f"**Assessment Complete for {name}**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Credit Score", credit_score)
            col2.metric("Default Probability", f"{prob_default*100:.1f}%")
            col3.metric("Decision", decision)
            v1, v2 = st.columns(2)
            with v1:
                st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=prob_default*100, title={'text': "Default Risk (%)"}, gauge={'axis': {'range': [0, 100]}})), use_container_width=True)
            with v2:
                imp = pd.DataFrame({'Feature': ['Loan-to-Income','Late Payments','Credit Utilization','Interest Rate'], 'Importance': [0.28, 0.24, 0.20, 0.15]})
                st.plotly_chart(px.bar(imp, x='Importance', y='Feature', orientation='h', title="Risk Drivers"), use_container_width=True)
        except Exception as e: st.error(f"❌ Error: {str(e)}")

# ====================== KYC SELFIE VERIFICATION (FIXED TO CODE 1) ======================
elif page == "📸 KYC Selfie Verification":
    st.header("📸 KYC Selfie Verification & Credit Decision")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Your Selfie")
        method = st.radio("Input Method", ["Live Camera", "Upload Photo"])
        photo = None
        if method == "Live Camera":
            photo = st.camera_input("Take clear selfie")
        else:
            uploaded = st.file_uploader("Upload Selfie", type=["jpg", "jpeg", "png"])
            if uploaded:
                photo = uploaded
        
        if photo:
            image = Image.open(photo)
            st.image(image, caption="Captured Selfie", use_column_width=True)
    
    with col2:
        st.subheader("🔍 Verification Result")
        if photo:
            # OpenCV Face Detection Logic from Code 1
            opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                st.error("❌ No face detected.")
            else:
                st.success("✅ Face detected")
                
                # Randomized Scoring Logic from Code 1
                base_score = 720
                variation = np.random.randint(-35, 45)
                final_score = max(300, min(850, base_score + variation))
                prob_default = round(max(0.01, (850 - final_score) / 680), 3)
                
                if final_score >= 730:
                    dec, color = "✅ APPROVED", "green"
                    limits = [800000, 1000000, 1200000, 1500000]
                elif final_score >= 650:
                    dec, color = "⚠️ REVIEW", "orange"
                    limits = [300000, 450000, 600000]
                else:
                    dec, color = "❌ DECLINED", "red"
                    limits = [0]
                
                rec_limit = np.random.choice(limits)
                limit_str = f"₹{rec_limit:,}" if rec_limit > 0 else "Not Eligible"
                
                st.markdown(f"### Final Decision: <span style='color:{color}; font-size:1.5em'>{dec}</span>", 
                           unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                c1.metric("Final Credit  Score", final_score)
                c2.metric("Default Prob", f"{prob_default*100:.1f}%")
                
                st.metric("Recommended Limit", limit_str)
                
                if st.button("✅ Approve KYC"):
                    st.balloons()
                    st.success("KYC Approved Successfully!")


# ====================== 🧪 DYNAMIC RISK PIPELINE (STRICT RESTORE) ======================
elif page == "🧪 Dynamic Risk Pipeline":
    st.header("Personalized Borrower Risk Analysis")
    st.markdown("### Real-time Stress Testing & Expected Loss Calculation")
    
    col_in, col_out = st.columns([1, 2])
    
    with col_in:
        st.subheader("Variable Parameters")
        with st.form("dynamic_risk_form"):
            u_score = st.slider("Credit Score", 300, 850, 650)
            u_dti = st.slider("DTI Ratio (Debt-to-Income)", 0.0, 1.0, 0.45)
            u_age = st.number_input("Applicant Age", 18, 90, 35)
            
            st.markdown("---")
            u_ead = st.number_input("Exposure at Default (EAD) ₹", value=1000000)
            u_lgd = st.slider("Loss Given Default (LGD) %", 0, 100, 40) / 100.0
            
            run_btn = st.form_submit_button("🚀 Generate Dynamic Report")
            
    with col_out:
        if run_btn:
            # Prediction using the restored Gradient Boosting logic
            # Features: [Score, DTI, Age]
            current_pd = float(gb_model.predict_proba([[u_score, u_dti, u_age]])[0][1])
            expected_loss = current_pd * u_lgd * u_ead
            
            # KPI Metrics
            m1, m2 = st.columns(2)
            m1.metric("Predicted PD", f"{current_pd:.2%}")
            m2.metric("Expected Loss (EL)", f"₹{expected_loss:,.2f}")
            
            # Sensitivity Analysis Chart: PD vs DTI Trend
            st.subheader("Sensitivity Analysis")
            dti_range = np.linspace(0, 1, 20)
            # Simulate changing DTI while keeping Score and Age constant
            pd_trend = gb_model.predict_proba([[u_score, d, u_age] for d in dti_range])[:, 1]
            
            fig_trend = px.line(
                x=dti_range, 
                y=pd_trend, 
                title="Risk Sensitivity: PD vs. Debt-to-Income Ratio",
                labels={'x': 'DTI Ratio', 'y': 'Probability of Default'},
                template="plotly_dark"
            )
            fig_trend.add_vline(x=u_dti, line_dash="dash", line_color="yellow", annotation_text="Current DTI")
            st.plotly_chart(fig_trend, use_container_width=True)
            
            st.success("Simulation complete. Data points reflected for internal risk rating (IRR).")
        else:
            st.info("Adjust the parameters on the left and click 'Generate Report' to run the simulation.")
# ====================== 🕵️ AML SURVEILLANCE (UPDATED PAGE) ======================
elif page == "🕵️ AML Surveillance":
    st.header("🕵️ Risk Dashboard: AML Surveillance")
    st.markdown("### Comply, Monitor, and Stay Ahead of the Market")
    
    # Overview Metrics for Compliance
    m1, m2, m3, m4 = st.columns(4)
    structuring_count = len(df_bank[df_bank['TransactionAmount'] > 45000])
    velocity_alerts = len(df_bank.groupby('AccountID').filter(lambda x: len(x) > 5))
    
    m1.metric("AML Alerts", structuring_count, "+2")
    m2.metric("High Velocity", velocity_alerts, "Stable")
    m3.metric("SARs Filed", "14", "Audit Ready")
    m4.metric("Compliance Score", "98%", "🛡️")

    st.divider()

    col1, col2 = st.columns([2, 1])
    

    with col1:
        st.subheader("🔍 Pattern Analysis")
        # Visualization of Smurfing (Many small transactions from same Account)
        smurf_check = df_bank.groupby('AccountID').size().reset_index(name='Volume')
        fig_smurf = px.histogram(smurf_check, x="Volume", 
                                 title="Transaction Density (Smurfing Check)",
                                 labels={'Volume': 'Transactions per Account'},
                                 color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig_smurf, use_container_width=True)

    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        # Rapid Movement of Funds
        st.subheader("⚡ Rapid Fund Movement")
        fig_velocity = px.scatter(df_bank, x="TransactionDate", y="TransactionAmount", 
                                 size="TransactionAmount", color="Channel",
                                 title="Global Transaction Flow Velocity",
                                 template="plotly_dark")
        st.plotly_chart(fig_velocity, use_container_width=True)
        
    with c2:
        st.subheader("📡 Regional Risk Distribution")
        # Synthetic Geographic Risk (as per common AML dashboards)
        geo_risk = pd.DataFrame({
            'Channel': ['ATM', 'Mobile', 'Web'],
            'Risk_Level': [0.8, 0.4, 0.6]
        })
        fig_geo = px.bar(geo_risk, x='Channel', y='Risk_Level', 
                         color='Risk_Level', color_continuous_scale='Reds',
                         title="Risk Exposure by Channel")
        st.plotly_chart(fig_geo, use_container_width=True)

    if st.button("📄 Generate Regulatory AML Report"):
        st.success("Regulatory report generated and encrypted for compliance officer review.")
# ====================== 🧠 SENTIMENT INTELLIGENCE (RESTORED) ======================
elif page == "🧠 Sentiment Intelligence":
    st.header("🧠 Market & Brand Sentiment Analysis")
    
    # Core Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Customer NPS", "72", "+5")
    m2.metric("Market Sentiment", "Bullish", "High")
    m3.metric("Reputation Risk", "Low", "-2%")
    
    # Topic Sentiment Chart
    st.subheader("Topic-Wise Sentiment Breakdown")
    fig = px.bar(
        x=["App", "Rates", "Service", "Security"], 
        y=[0.8, -0.4, 0.6, 0.9], 
        color=[0.8, -0.4, 0.6, 0.9], 
        color_continuous_scale="RdYlGn", 
        title="Customer Sentiment by Category",
        labels={'x': 'Category', 'y': 'Sentiment Score'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment Area Trend
    st.subheader("Reputation Trend (30 Days)")

    dates = pd.date_range(end=datetime.datetime.now(), periods=30)
    sentiment_values = np.random.uniform(0.5, 0.9, 30).cumsum() / 10

    fig_sent = px.area(
        x=dates,
        y=sentiment_values,
        title="Historical Sentiment Stability Index",
        labels={
            "x": "Date",
            "y": "Sentiment Score"
        }
    )

    # Add value labels on points
    fig_sent.update_traces(
        mode="lines+markers+text",
        text=[f"{v:.2f}" for v in sentiment_values],
        textposition="top center"
    )

    # Optional layout improvements
    fig_sent.update_layout(
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        template="plotly_white"
    )

    st.plotly_chart(fig_sent, use_container_width=True)

# ====================== 📈 MARKET FORECAST (ENHANCED) ======================
elif page == "📈 Market Forecast":
    st.header("Predictive Market Analytics")
    ticker = st.selectbox("Select Asset", ["AAPL", "RELIANCE.NS", "TSLA", "GOOGL", "NVDA"])
    
    try:
        # Data Retrieval
        df = yf.download(ticker, period="1y")
        
        if not df.empty:
            # Data Processing
            prices = df['Close'].squeeze().astype(float)
            volume = df['Volume'].squeeze().astype(float)
            
            # --- 3 NEW KPIs + 1 ORIGINAL ---
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            
            # KPI 1: Current Price (Original)
            current_price = prices.iloc[-1]
            price_delta = current_price - prices.iloc[-2]
            kpi1.metric("Current Price", f"${current_price:,.2f}", f"{price_delta:+.2f}")
            
            # KPI 2: 52-Week High (New)
            high_52 = prices.max()
            kpi2.metric("52-Week High", f"${high_52:,.2f}")
            
            # KPI 3: Volatility (Annualized Std Dev) (New)
            volatility = (prices.pct_change().std() * np.sqrt(252)) * 100
            kpi3.metric("Volatility (Ann.)", f"{volatility:.2f}%")
            
            # KPI 4: RSI (14-Day) (New)
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
            kpi4.metric("RSI (14D)", f"{current_rsi:.1f}", rsi_status)

            st.divider()

            # --- CHART 1: MAIN FORECAST (STRICT RESTORE) ---
            model_ts = ExponentialSmoothing(prices, trend='add').fit()
            forecast_steps = 10
            forecast_values = model_ts.forecast(forecast_steps)
            forecast_dates = [prices.index[-1] + datetime.timedelta(days=i) for i in range(1, forecast_steps + 1)]
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=prices.index[-60:], y=prices.tail(60), name="History", line=dict(color='#1f77b4')))
            fig1.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, name="10D Forecast", line=dict(dash='dash', color='red')))
            fig1.update_layout(title=f"{ticker} Price Prediction", template="plotly_dark", height=400)
            st.plotly_chart(fig1, use_container_width=True)

            col_left, col_right = st.columns(2)

            with col_left:
                # --- CHART 2: RSI MOMENTUM (NEW) ---
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=rsi.index[-60:], y=rsi.tail(60), name="RSI", line=dict(color='#FFA500')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(title="RSI Momentum (14-Day)", template="plotly_dark", height=350, yaxis_range=[0, 100])
                st.plotly_chart(fig_rsi, use_container_width=True)

            with col_right:
                # --- CHART 3: PRICE VS VOLUME (NEW) ---
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(x=volume.index[-30:], y=volume.tail(30), name="Volume", marker_color='rgba(100, 100, 100, 0.5)'))
                fig_vol.update_layout(title="Trading Volume (Last 30 Days)", template="plotly_dark", height=350)
                st.plotly_chart(fig_vol, use_container_width=True)
            
        else:
            st.error("Market data not found for the selected ticker.")
            
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")

# ====================== 11. MODEL GOVERNANCE (CODE 2) ======================
elif page == "⚖️ Model Governance":
    st.header("MLOps Governance & Regulatory Compliance")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Model Health", "94.2%", "0.4%"); g2.metric("Data Drift (PSI)", "0.08", "-0.02", delta_color="inverse")
    g3.metric("Bias Variance", "0.012", "Stable"); g4.metric("Uptime", "99.99%", "Production")
    st.plotly_chart(px.bar(pd.DataFrame({"Feature": ["Credit Score", "DTI", "Age"], "PSI": [0.03, 0.09, 0.02]}), 
    x="Feature", y="PSI", title="PSI Score"), use_container_width=True)



# ====================== 10. INSURTECH & PRICE RISK (DYNAMIC) ======================
elif page == "🏢 InsurTech & Final Pitch":
    st.header("🏢 InsurTech & Price Risk Analytics")
    st.markdown("### Integrated Risk Modeling & Financial Projections")

    # --- SECTION 1: INPUT PARAMETERS ---
    with st.form("price_risk_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            policy_type = st.selectbox("Policy Segment", ["Term Life", "Health Plus", "Auto Guard", "Home Shield"])
            coverage = st.number_input("Requested Coverage (₹)", 500000, 50000000, 1000000, step=500000)
        with c2:
            market_vol = st.slider("Market Volatility Index", 0.0, 1.0, 0.4)
            past_claims = st.number_input("Historical Claims Count", 0, 10, 0)
        with c3:
            retention_target = st.slider("Target Retention Rate (%)", 50, 99, 85)
            st.caption("⚙️ Data synced with Market & Credit modules")
        
        calculate_btn = st.form_submit_button("📊 Calculate Risk & Projections", use_container_width=True)

    if calculate_btn:
        # --- ACTUARIAL LOGIC & CALCULATIONS ---
        base_rate = {"Term Life": 0.0012, "Health Plus": 0.025, "Auto Guard": 0.045, "Home Shield": 0.008}
        
        # Risk Multiplier based on volatility and claims
        risk_multiplier = (market_vol * 1.8) + (past_claims * 0.25)
        annual_premium = (coverage * base_rate[policy_type]) * (1 + risk_multiplier)
        
        # Derived Metrics
        loss_ratio = max(40, 95 - (retention_target * 0.5) + (market_vol * 20))
        margin_pct = 100 - loss_ratio - 15  # 15% fixed OpEx
        expected_profit = annual_premium * (margin_pct / 100)

        # --- SECTION 2: DYNAMIC KPI CARDS ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Risk-Adjusted Premium", f"₹{annual_premium:,.2f}", f"{risk_multiplier:+.1%} Risk")
        kpi2.metric("Projected Loss Ratio", f"{loss_ratio:.1f}%", f"{'High' if loss_ratio > 70 else 'Stable'}")
        kpi3.metric("Expected Net Margin", f"{margin_pct:.1f}%")
        kpi4.metric("Per-Policy Profit", f"₹{expected_profit:,.0f}")

        st.divider()

        # --- SECTION 3: 4-CHART DYNAMIC DASHBOARD ---
        r1_c1, r1_c2 = st.columns(2)
        r2_c1, r2_c2 = st.columns(2)

        with r1_c1:
            # CHART 1: Premium Sensitivity
            v_range = np.linspace(0, 1, 15)
            p_trend = [(coverage * base_rate[policy_type]) * (1 + (v * 1.8) + (past_claims * 0.25)) for v in v_range]
            fig1 = px.line(x=v_range, y=p_trend, title="Premium Sensitivity to Volatility",
                          labels={'x': 'Volatility Index', 'y': 'Premium (₹)'}, template="plotly_dark")
            fig1.add_vline(x=market_vol, line_dash="dash", line_color="yellow")
            st.plotly_chart(fig1, use_container_width=True)

        with r1_c2:
            # CHART 2: Margin Decay Analysis
            loss_scenarios = np.linspace(40, 90, 10)
            margin_scenarios = [100 - l - 15 for l in loss_scenarios]
            fig2 = px.area(x=loss_scenarios, y=margin_scenarios, title="Profit Margin vs. Loss Ratio",
                          labels={'x': 'Loss Ratio (%)', 'y': 'Margin (%)'}, template="plotly_dark", color_discrete_sequence=['#FF4B4B'])
            st.plotly_chart(fig2, use_container_width=True)

        with r2_c1:
            # CHART 3: Risk Probability Distribution
            risk_dist = np.random.normal(annual_premium, annual_premium * (market_vol * 0.5), 1000)
            fig3 = px.histogram(x=risk_dist, nbins=30, title="Price Risk Probability Spread",
                               template="plotly_dark", color_discrete_sequence=['#00CC96'])
            st.plotly_chart(fig3, use_container_width=True)

        with r2_c2:
            # CHART 4: Revenue Composition
            composition = pd.DataFrame({
                'Component': ['Expected Loss', 'Operating Cost', 'Net Profit'],
                'Value': [loss_ratio, 15, max(0, margin_pct)]
            })
            fig4 = px.pie(composition, values='Value', names='Component', title="Premium Breakout",
                         hole=0.4, color_discrete_sequence=px.colors.sequential.YlGnBu_r)
            st.plotly_chart(fig4, use_container_width=True)

        st.divider()

        # --- SECTION 4: FINAL PITCH ROI ---
        st.subheader("📈 5-Year Integrated Profit Forecast")
        years = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
        # Growth compounded by retention and premium size
        growth_factor = 1 + (retention_target / 100) - 0.5
        projections = [expected_profit * (growth_factor**i) / 100000 for i in range(5)] # In Lakhs
        
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Bar(x=years, y=projections, marker_color='#00CC96', name='Lakhs (₹)'))
        fig_roi.update_layout(title="Scaling Impact (Cumulative Profit in Lakhs)", template="plotly_dark", height=350)
        st.plotly_chart(fig_roi, use_container_width=True)

        if st.button("📢 Confirm Strategic Presentation"):
            st.balloons()
            st.success("Financial models locked and integrated into the Sentinel ecosystem.")
    else:
        st.info("💡 Adjust parameters and click 'Calculate' to generate the Price Risk analysis.")

st.sidebar.markdown("---")
st.sidebar.caption("BankAI Sentinel | Integrated Intelligence Suite")