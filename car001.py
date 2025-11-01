# ======================================================
# ENHANCED SMART CAR PRICING SYSTEM WITH ACCURATE PREDICTIONS
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from datetime import datetime

# ========================================
# ENHANCED PRICE PREDICTION ENGINE
# ========================================

class EnhancedCarPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_importance = {}
        
    def create_synthetic_training_data(self):
        """Create comprehensive synthetic training data for all car models"""
        np.random.seed(42)
        records = []
        
        current_year = datetime.now().year
        
        for brand in CAR_DATABASE:
            for i, model in enumerate(CAR_DATABASE[brand]['models']):
                car_type = CAR_DATABASE[brand]['car_types'][i]
                engine_cc = CAR_DATABASE[brand]['engine_cc'][i]
                power_hp = CAR_DATABASE[brand]['power_hp'][i]
                seats = CAR_DATABASE[brand]['seats'][i]
                
                # Get base price range for this model
                base_prices, _ = get_enhanced_live_prices(brand, model)
                base_price = base_prices[1]  # Use average price
                
                # Generate multiple records with variations
                for _ in range(50):  # 50 records per model
                    year = np.random.randint(max(1990, current_year-20), current_year+1)
                    age = current_year - year
                    
                    mileage = np.random.randint(1000, min(300000, 15000 * age))
                    
                    # Condition probabilities
                    condition_weights = [0.1, 0.2, 0.4, 0.2, 0.1]  # More 'Good' condition cars
                    condition = np.random.choice(CAR_CONDITIONS, p=condition_weights)
                    
                    owner_weights = [0.4, 0.3, 0.2, 0.1]  # More first owners
                    owner_type = np.random.choice(OWNER_TYPES, p=owner_weights)
                    
                    fuel_type = np.random.choice(FUEL_TYPES)
                    transmission = np.random.choice(TRANSMISSIONS)
                    
                    # Calculate price with realistic factors
                    price = self.calculate_realistic_price(
                        base_price, age, mileage, condition, owner_type, 
                        fuel_type, transmission, brand, car_type
                    )
                    
                    records.append({
                        'Brand': brand,
                        'Model': model,
                        'Car_Type': car_type,
                        'Year': year,
                        'Fuel_Type': fuel_type,
                        'Transmission': transmission,
                        'Mileage': mileage,
                        'Engine_cc': engine_cc,
                        'Power_HP': power_hp,
                        'Seats': seats,
                        'Condition': condition,
                        'Owner_Type': owner_type,
                        'Price': price
                    })
        
        return pd.DataFrame(records)
    
    def calculate_realistic_price(self, base_price, age, mileage, condition, owner_type, 
                                fuel_type, transmission, brand, car_type):
        """Calculate realistic price based on multiple factors"""
        
        # Age depreciation (non-linear)
        age_depreciation = 0.85 ** age  # 15% depreciation per year
        
        # Mileage depreciation
        mileage_factor = max(0.3, 1 - (mileage / 200000))
        
        # Condition multipliers
        condition_multipliers = {
            "Excellent": 1.15,
            "Very Good": 1.05,
            "Good": 1.0,
            "Fair": 0.85,
            "Poor": 0.65
        }
        
        # Owner type multipliers
        owner_multipliers = {
            "First": 1.08,
            "Second": 1.0,
            "Third": 0.92,
            "Fourth & Above": 0.82
        }
        
        # Fuel type adjustments
        fuel_adjustments = {
            "Petrol": 1.0,
            "Diesel": 1.05,
            "CNG": 0.9,
            "Electric": 1.15,
            "Hybrid": 1.1
        }
        
        # Transmission adjustments
        transmission_adjustments = {
            "Manual": 1.0,
            "Automatic": 1.08,
            "CVT": 1.05,
            "DCT": 1.1,
            "AMT": 1.02
        }
        
        # Brand premium factors
        brand_premium = {
            'Maruti Suzuki': 1.02, 'Hyundai': 1.01, 'Tata': 1.0, 'Mahindra': 1.01,
            'Toyota': 1.05, 'Honda': 1.03, 'Kia': 1.02, 'Volkswagen': 1.02,
            'Skoda': 1.01, 'Renault': 1.0, 'Nissan': 1.0, 'MG': 1.03,
            'Ford': 1.0, 'BMW': 1.25, 'Mercedes-Benz': 1.28, 'Audi': 1.26,
            'Lexus': 1.22, 'Jaguar': 1.2, 'Land Rover': 1.23, 'Porsche': 1.35,
            'Volvo': 1.18, 'Maserati': 1.3, 'Bentley': 1.4, 'Rolls-Royce': 1.5,
            'Lamborghini': 1.45, 'Ferrari': 1.48, 'Aston Martin': 1.38,
            'McLaren': 1.42, 'Bugatti': 1.6
        }
        
        # Calculate final price
        price = (base_price * age_depreciation * mileage_factor * 
                condition_multipliers[condition] * owner_multipliers[owner_type] *
                fuel_adjustments[fuel_type] * transmission_adjustments[transmission] *
                brand_premium.get(brand, 1.0))
        
        # Add some random variation (±8%)
        variation = np.random.uniform(0.92, 1.08)
        price *= variation
        
        return max(50000, int(price))
    
    def train_model(self):
        """Train the enhanced prediction model"""
        st.info("🔄 Training advanced price prediction model...")
        
        # Create comprehensive training data
        df = self.create_synthetic_training_data()
        
        # Prepare features
        features = ['Brand', 'Model', 'Car_Type', 'Year', 'Fuel_Type', 'Transmission',
                   'Mileage', 'Engine_cc', 'Power_HP', 'Seats', 'Condition', 'Owner_Type']
        
        X = df[features]
        y = df['Price']
        
        # Encode categorical variables
        categorical_features = ['Brand', 'Model', 'Car_Type', 'Fuel_Type', 'Transmission', 'Condition', 'Owner_Type']
        for feature in categorical_features:
            self.encoders[feature] = LabelEncoder()
            X[feature] = self.encoders[feature].fit_transform(X[feature])
        
        # Scale numerical features
        numerical_features = ['Year', 'Mileage', 'Engine_cc', 'Power_HP', 'Seats']
        X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        
        # Train ensemble model
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        )
        
        # Train models
        rf_model.fit(X, y)
        gb_model.fit(X, y)
        
        # Store feature importance
        self.feature_importance = dict(zip(features, rf_model.feature_importances_))
        
        # Use ensemble of both models
        self.model = {'rf': rf_model, 'gb': gb_model}
        
        # Evaluate model
        y_pred_rf = rf_model.predict(X)
        y_pred_gb = gb_model.predict(X)
        y_pred_ensemble = (y_pred_rf + y_pred_gb) / 2
        
        r2 = r2_score(y, y_pred_ensemble)
        mae = mean_absolute_error(y, y_pred_ensemble)
        
        st.success(f"✅ Model trained successfully! R² Score: {r2:.3f}, MAE: ₹{mae:,.0f}")
        
        return self.model
    
    def predict_price(self, input_data):
        """Predict car price with enhanced accuracy"""
        if self.model is None:
            self.train_model()
        
        # Prepare input features
        features = ['Brand', 'Model', 'Car_Type', 'Year', 'Fuel_Type', 'Transmission',
                   'Mileage', 'Engine_cc', 'Power_HP', 'Seats', 'Condition', 'Owner_Type']
        
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        for feature in ['Brand', 'Model', 'Car_Type', 'Fuel_Type', 'Transmission', 'Condition', 'Owner_Type']:
            if feature in self.encoders:
                try:
                    input_df[feature] = self.encoders[feature].transform([input_data[feature]])[0]
                except ValueError:
                    # Handle unseen labels
                    input_df[feature] = 0
        
        # Scale numerical features
        numerical_features = ['Year', 'Mileage', 'Engine_cc', 'Power_HP', 'Seats']
        input_df[numerical_features] = self.scaler.transform(input_df[numerical_features])
        
        # Ensure all features are present
        for feature in features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        input_df = input_df[features]
        
        # Get predictions from both models
        rf_pred = self.model['rf'].predict(input_df)[0]
        gb_pred = self.model['gb'].predict(input_df)[0]
        
        # Ensemble prediction
        final_prediction = (rf_pred + gb_pred) / 2
        
        # Apply additional business rules
        final_prediction = self.apply_business_rules(final_prediction, input_data)
        
        return max(50000, int(final_prediction))
    
    def apply_business_rules(self, predicted_price, input_data):
        """Apply business rules and domain knowledge"""
        adjusted_price = predicted_price
        
        # Age-based adjustment (non-linear depreciation)
        current_year = datetime.now().year
        age = current_year - input_data['Year']
        if age > 10:
            adjusted_price *= 0.9  # Additional discount for very old cars
        elif age < 3:
            adjusted_price *= 1.05  # Premium for nearly new cars
        
        # Mileage adjustment
        mileage = input_data['Mileage']
        if mileage > 100000:
            adjusted_price *= 0.92
        elif mileage < 20000:
            adjusted_price *= 1.03
        
        # Luxury car specific rules
        luxury_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Jaguar', 'Land Rover', 
                        'Porsche', 'Volvo', 'Maserati', 'Bentley', 'Rolls-Royce', 
                        'Lamborghini', 'Ferrari', 'Aston Martin', 'McLaren', 'Bugatti']
        
        if input_data['Brand'] in luxury_brands:
            # Luxury cars depreciate faster initially but hold value better later
            if age < 5:
                adjusted_price *= 0.95
            else:
                adjusted_price *= 1.02
        
        return adjusted_price

# ========================================
# ENHANCED PRICE PREDICTION INTERFACE
# ========================================

def show_enhanced_prediction_interface():
    """Show the enhanced price prediction interface"""
    st.subheader("🎯 Advanced Price Prediction")
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = EnhancedCarPricePredictor()
    
    # Manual input form
    input_data = show_manual_input_form()
    
    if input_data:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show real-time market prices
            brand = input_data['Brand']
            model = input_data['Model']
            
            if brand and model:
                with st.spinner('🔍 Analyzing market trends...'):
                    prices, sources = get_real_time_prices(brand, model)
                    min_price, avg_price, max_price = prices
                
                # Display market intelligence
                st.subheader("📊 Market Intelligence")
                
                market_col1, market_col2, market_col3 = st.columns(3)
                
                with market_col1:
                    st.metric("Current Market Low", f"₹{min_price:,.0f}")
                
                with market_col2:
                    st.metric("Market Average", f"₹{avg_price:,.0f}")
                
                with market_col3:
                    st.metric("Premium Range", f"₹{max_price:,.0f}")
        
        with col2:
            st.subheader("🤖 AI Prediction")
            
            if st.button("🎯 Get Accurate Price", type="primary", use_container_width=True):
                with st.spinner('🤖 Calculating optimal price...'):
                    # Get AI prediction
                    predicted_price = st.session_state.predictor.predict_price(input_data)
                    
                    # Show confidence factors
                    confidence = calculate_confidence(input_data)
                    
                    # Display result
                    st.success(f"**Recommended Price: ₹{predicted_price:,.0f}**")
                    st.metric("Confidence Level", f"{confidence}%")
                    
                    # Price justification
                    show_price_breakdown(input_data, predicted_price, avg_price)
                    
                    st.balloons()

def calculate_confidence(input_data):
    """Calculate prediction confidence based on data quality"""
    confidence = 85  # Base confidence
    
    # Increase confidence for popular brands
    popular_brands = ['Maruti Suzuki', 'Hyundai', 'Tata', 'Mahindra', 'Honda', 'Toyota']
    if input_data['Brand'] in popular_brands:
        confidence += 5
    
    # Increase confidence for newer cars
    current_year = datetime.now().year
    if current_year - input_data['Year'] <= 5:
        confidence += 3
    
    # Decrease confidence for high mileage
    if input_data['Mileage'] > 100000:
        confidence -= 5
    
    return min(95, max(70, confidence))

def show_price_breakdown(input_data, predicted_price, market_avg):
    """Show detailed price breakdown"""
    st.subheader("💰 Price Breakdown Analysis")
    
    # Calculate factors affecting price
    current_year = datetime.now().year
    age = current_year - input_data['Year']
    
    factors = {
        'Market Average': market_avg,
        'Age Adjustment (Depreciation)': predicted_price - market_avg,
        'Mileage Factor': -int(input_data['Mileage'] * 0.5),
        'Condition Premium': get_condition_premium(input_data['Condition']),
        'Owner History Impact': get_owner_impact(input_data['Owner_Type']),
        'Brand Value Factor': get_brand_factor(input_data['Brand'])
    }
    
    breakdown_df = pd.DataFrame({
        'Factor': factors.keys(),
        'Impact': factors.values(),
        'Description': [
            'Current market average for similar models',
            f'Depreciation for {age} year old car',
            f'Adjustment for {input_data["Mileage"]:,} km mileage',
            f'{input_data["Condition"]} condition premium/discount',
            f'{input_data["Owner_Type"]} owner impact',
            f'{input_data["Brand"]} brand value factor'
        ]
    })
    
    st.dataframe(breakdown_df, use_container_width=True)
    
    # Show feature importance
    if hasattr(st.session_state.predictor, 'feature_importance'):
        st.subheader("📈 Key Price Influencers")
        
        importance_df = pd.DataFrame({
            'Feature': list(st.session_state.predictor.feature_importance.keys()),
            'Importance': list(st.session_state.predictor.feature_importance.values())
        }).sort_values('Importance', ascending=False).head(8)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title='Most Important Factors Affecting Car Price')
        st.plotly_chart(fig, use_container_width=True)

def get_condition_premium(condition):
    """Get condition-based price adjustment"""
    premiums = {
        "Excellent": 50000,
        "Very Good": 25000,
        "Good": 0,
        "Fair": -20000,
        "Poor": -50000
    }
    return premiums.get(condition, 0)

def get_owner_impact(owner_type):
    """Get owner history impact"""
    impacts = {
        "First": 30000,
        "Second": 0,
        "Third": -15000,
        "Fourth & Above": -30000
    }
    return impacts.get(owner_type, 0)

def get_brand_factor(brand):
    """Get brand-specific adjustment"""
    factors = {
        'Maruti Suzuki': 20000,
        'Toyota': 25000,
        'Honda': 20000,
        'Hyundai': 15000,
        'Tata': 10000,
        'Mahindra': 12000,
        'BMW': 50000,
        'Mercedes-Benz': 60000,
        'Audi': 45000
    }
    return factors.get(brand, 0)

# ========================================
# UPDATE MAIN FUNCTION
# ========================================

def main():
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = EnhancedCarPricePredictor()
    
    st.set_page_config(
        page_title="Advanced Car Price Predictor", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    st.title("🚗 Advanced Car Price Prediction System")
    st.markdown("### **AI-Powered Accurate Price Estimation with Market Intelligence**")
    
    # Show brand statistics in sidebar
    show_brand_statistics()
    
    # Add search functionality
    search_cars()
    
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/48/car.png")
        st.title("Navigation")
        page = st.radio("Go to", ["Advanced Price Prediction", "Brand Explorer", "Market Analysis", "Model Training"])
        
        st.markdown("---")
        st.subheader("AI Features")
        st.success("✅ Ensemble Machine Learning")
        st.success("✅ Real-Time Market Data")
        st.success("✅ Price Breakdown Analysis")
        st.success("✅ Confidence Scoring")
        
        if page == "Model Training":
            if st.button("🔄 Train Enhanced Model", use_container_width=True):
                with st.spinner("Training advanced AI model..."):
                    st.session_state.predictor.train_model()
    
    if page == "Advanced Price Prediction":
        show_enhanced_prediction_interface()
    
    elif page == "Brand Explorer":
        show_brand_explorer()
    
    elif page == "Market Analysis":
        show_market_analysis()
    
    elif page == "Model Training":
        show_model_training()

def show_brand_explorer():
    """Enhanced brand explorer"""
    st.subheader("🔍 Car Brand & Model Explorer")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_brand = st.selectbox("Select Brand", list(CAR_DATABASE.keys()))
        
        if selected_brand in CAR_DATABASE:
            st.info(f"**{selected_brand}** has **{len(CAR_DATABASE[selected_brand]['models'])}** models")
            
            # Show price range for brand
            prices, _ = get_enhanced_live_prices(selected_brand, CAR_DATABASE[selected_brand]['models'][0])
            st.metric("Starting Price", f"₹{prices[0]:,.0f}")

def show_market_analysis():
    """Enhanced market analysis"""
    st.subheader("📈 Advanced Market Analysis")
    
    # Price distribution by brand
    brand_prices = {}
    for brand in list(CAR_DATABASE.keys())[:15]:  # Limit to first 15 brands for performance
        try:
            prices, _ = get_enhanced_live_prices(brand, CAR_DATABASE[brand]['models'][0])
            brand_prices[brand] = prices[1]  # Use average price
        except:
            continue
    
    if brand_prices:
        fig = px.bar(x=list(brand_prices.keys()), y=list(brand_prices.values()),
                    title="Average Price by Brand", labels={'x': 'Brand', 'y': 'Price (₹)'})
        st.plotly_chart(fig, use_container_width=True)

def show_model_training():
    """Model training interface"""
    st.subheader("🤖 AI Model Training")
    
    st.info("""
    **Enhanced Machine Learning Features:**
    - Ensemble Learning (Random Forest + Gradient Boosting)
    - Comprehensive Synthetic Training Data
    - Realistic Price Calculation Algorithms
    - Business Rules Integration
    - Confidence Scoring
    """)
    
    if st.button("🚀 Train Advanced Model", type="primary"):
        with st.spinner("Creating comprehensive training dataset and training AI models..."):
            st.session_state.predictor.train_model()

# ========================================
# RUN THE APPLICATION
# ========================================

if __name__ == "__main__":
    main()
