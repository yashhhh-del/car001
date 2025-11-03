import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime
import random

# Page config
st.set_page_config(page_title="Smart Car Pricing", page_icon="🚗", layout="wide")

# ========================================
# COMPREHENSIVE CAR DATABASE
# ========================================

CAR_DATABASE = {
    'Maruti Suzuki': {
        'models': ['Alto 800', 'Alto K10', 'WagonR', 'Swift', 'Baleno', 'Dzire', 'Ertiga', 'Brezza', 'Fronx', 'Grand Vitara'],
        'prices': [[150000, 250000], [300000, 450000], [300000, 450000], [400000, 600000], [450000, 650000], 
                  [400000, 600000], [550000, 800000], [600000, 900000], [500000, 750000], [900000, 1300000]]
    },
    'Hyundai': {
        'models': ['i10', 'i20', 'Aura', 'Verna', 'Venue', 'Creta', 'Alcazar', 'Tucson'],
        'prices': [[250000, 400000], [400000, 600000], [350000, 550000], [550000, 800000], 
                  [500000, 750000], [750000, 1100000], [950000, 1400000], [1500000, 2200000]]
    },
    'Tata': {
        'models': ['Tiago', 'Tigor', 'Altroz', 'Punch', 'Nexon', 'Harrier', 'Safari'],
        'prices': [[300000, 450000], [320000, 480000], [400000, 600000], [350000, 550000], 
                  [550000, 850000], [950000, 1400000], [1100000, 1600000]]
    },
    'Mahindra': {
        'models': ['Bolero', 'Scorpio', 'XUV300', 'XUV700', 'Thar', 'Scorpio N'],
        'prices': [[400000, 600000], [600000, 900000], [550000, 850000], [1000000, 1500000], 
                  [750000, 1100000], [900000, 1350000]]
    },
    'Honda': {
        'models': ['Amaze', 'City', 'Elevate', 'City Hybrid'],
        'prices': [[400000, 600000], [550000, 850000], [700000, 1000000], [950000, 1350000]]
    },
    'Toyota': {
        'models': ['Glanza', 'Urban Cruiser', 'Innova Crysta', 'Fortuner', 'Camry'],
        'prices': [[400000, 600000], [700000, 1000000], [1200000, 1800000], [1800000, 2600000], [2500000, 3500000]]
    },
    'Kia': {
        'models': ['Sonet', 'Seltos', 'Carens', 'Carnival'],
        'prices': [[500000, 750000], [750000, 1100000], [800000, 1200000], [2000000, 2800000]]
    },
    'Volkswagen': {
        'models': ['Polo', 'Virtus', 'Taigun'],
        'prices': [[400000, 600000], [650000, 950000], [750000, 1100000]]
    },
    'Skoda': {
        'models': ['Rapid', 'Slavia', 'Kushaq'],
        'prices': [[400000, 600000], [650000, 950000], [750000, 1100000]]
    },
    'Renault': {
        'models': ['Kwid', 'Triber', 'Kiger'],
        'prices': [[250000, 400000], [400000, 600000], [450000, 700000]]
    },
    'Nissan': {
        'models': ['Magnite', 'Kicks'],
        'prices': [[450000, 700000], [600000, 900000]]
    },
    'MG': {
        'models': ['Hector', 'Astor', 'Gloster', 'ZS EV'],
        'prices': [[800000, 1200000], [700000, 1050000], [1400000, 2000000], [1200000, 1700000]]
    },
    'BMW': {
        'models': ['2 Series', '3 Series', '5 Series', '7 Series', 'X1', 'X3', 'X5', 'X7'],
        'prices': [[2500000, 3500000], [3000000, 4500000], [4500000, 6500000], [9000000, 13000000],
                  [2800000, 4000000], [4000000, 6000000], [6500000, 9500000], [10000000, 14000000]]
    },
    'Mercedes-Benz': {
        'models': ['A-Class', 'C-Class', 'E-Class', 'S-Class', 'GLA', 'GLC', 'GLE', 'GLS'],
        'prices': [[2600000, 3800000], [3500000, 5200000], [5500000, 8000000], [10000000, 15000000],
                  [2800000, 4200000], [4500000, 6700000], [6500000, 9500000], [9000000, 13000000]]
    },
    'Audi': {
        'models': ['A3', 'A4', 'A6', 'A8', 'Q3', 'Q5', 'Q7', 'Q8', 'e-tron'],
        'prices': [[2500000, 3700000], [3200000, 4800000], [4800000, 7000000], [8500000, 12500000],
                  [3000000, 4500000], [4500000, 6500000], [7000000, 10000000], [8000000, 11500000], [7000000, 10000000]]
    },
    'Porsche': {
        'models': ['718', '911', 'Panamera', 'Cayenne', 'Macan', 'Taycan'],
        'prices': [[7000000, 10000000], [11000000, 16000000], [10000000, 14000000], 
                  [9000000, 13000000], [5500000, 8500000], [11000000, 16000000]]
    },
    'Jaguar': {
        'models': ['XE', 'XF', 'F-PACE', 'I-PACE'],
        'prices': [[3000000, 4500000], [3500000, 5200000], [4000000, 6000000], [6500000, 9500000]]
    },
    'Land Rover': {
        'models': ['Range Rover Evoque', 'Range Rover Velar', 'Range Rover Sport', 'Range Rover', 'Defender', 'Discovery'],
        'prices': [[3500000, 5200000], [4000000, 6000000], [6500000, 9500000], [12000000, 18000000], 
                  [5000000, 7500000], [5500000, 8200000]]
    },
    'Volvo': {
        'models': ['S60', 'S90', 'XC40', 'XC60', 'XC90'],
        'prices': [[3000000, 4500000], [3500000, 5200000], [2500000, 3800000], 
                  [3800000, 5600000], [5500000, 8200000]]
    },
    'Lamborghini': {
        'models': ['Huracan', 'Urus', 'Aventador', 'Revuelto'],
        'prices': [[22000000, 32000000], [24000000, 35000000], [38000000, 55000000], [50000000, 70000000]]
    },
    'Ferrari': {
        'models': ['Portofino', 'Roma', 'F8 Tributo', 'SF90', '296 GTB', 'Purosangue'],
        'prices': [[20000000, 30000000], [22000000, 32000000], [27000000, 40000000], 
                  [48000000, 68000000], [32000000, 47000000], [40000000, 58000000]]
    },
    'Rolls-Royce': {
        'models': ['Ghost', 'Phantom', 'Cullinan', 'Wraith', 'Dawn'],
        'prices': [[45000000, 65000000], [60000000, 85000000], [55000000, 78000000], 
                  [38000000, 55000000], [42000000, 60000000]]
    },
    'Bentley': {
        'models': ['Continental GT', 'Flying Spur', 'Bentayga'],
        'prices': [[25000000, 38000000], [28000000, 42000000], [30000000, 45000000]]
    }
}

FUEL_TYPES = ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
TRANSMISSIONS = ["Manual", "Automatic", "CVT", "DCT", "AMT"]
CONDITIONS = ["Excellent", "Very Good", "Good", "Fair", "Poor"]
OWNERS = ["First", "Second", "Third", "Fourth & Above"]
COLORS = ["White", "Black", "Silver", "Grey", "Red", "Blue", "Brown", "Other"]

# ========================================
# PRICE PREDICTOR CLASS
# ========================================

class CarPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.is_trained = False
        
    def create_training_data(self):
        """Create simple training data"""
        data = []
        current_year = datetime.now().year
        
        for brand, info in CAR_DATABASE.items():
            for idx, model in enumerate(info['models']):
                price_range = info['prices'][idx]
                avg_price = sum(price_range) / 2
                
                # Generate 20 samples per model
                for _ in range(20):
                    year = random.randint(max(2005, current_year - 15), current_year)
                    age = current_year - year
                    
                    # Generate mileage based on age
                    if age == 0:
                        mileage = random.randint(500, 3000)
                    elif age <= 2:
                        mileage = random.randint(5000, 25000)
                    else:
                        mileage = random.randint(10000, min(200000, age * 12000))
                    
                    condition = random.choice(CONDITIONS)
                    owner = random.choice(OWNERS)
                    fuel = random.choice(FUEL_TYPES)
                    transmission = random.choice(TRANSMISSIONS)
                    
                    # Calculate price with depreciation
                    price = avg_price * (0.88 ** age)  # 12% depreciation per year
                    
                    # Adjust for mileage
                    price *= max(0.4, 1 - (mileage / 300000))
                    
                    # Adjust for condition
                    condition_factors = {"Excellent": 1.1, "Very Good": 1.05, "Good": 1.0, "Fair": 0.9, "Poor": 0.75}
                    price *= condition_factors[condition]
                    
                    # Adjust for owner
                    owner_factors = {"First": 1.08, "Second": 1.0, "Third": 0.93, "Fourth & Above": 0.85}
                    price *= owner_factors[owner]
                    
                    # Random variation
                    price *= random.uniform(0.92, 1.08)
                    
                    data.append({
                        'Brand': brand,
                        'Model': model,
                        'Year': year,
                        'Mileage': mileage,
                        'Fuel_Type': fuel,
                        'Transmission': transmission,
                        'Condition': condition,
                        'Owner_Type': owner,
                        'Price': max(50000, int(price))
                    })
        
        return pd.DataFrame(data)
    
    def train(self):
        """Train the model"""
        with st.spinner("🔄 Training AI model..."):
            # Create data
            df = self.create_training_data()
            
            # Prepare features
            X = df[['Brand', 'Model', 'Year', 'Mileage', 'Fuel_Type', 'Transmission', 'Condition', 'Owner_Type']].copy()
            y = df['Price']
            
            # Encode categorical
            for col in ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Condition', 'Owner_Type']:
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col])
            
            # Scale numerical
            X[['Year', 'Mileage']] = self.scaler.fit_transform(X[['Year', 'Mileage']])
            
            # Train
            self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
            self.model.fit(X, y)
            
            self.is_trained = True
            
            # Evaluate
            y_pred = self.model.predict(X)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            st.success(f"✅ Model trained! R² = {r2:.3f}, MAE = ₹{mae:,.0f}")
            st.info(f"📊 Trained on {len(df)} samples from {len(CAR_DATABASE)} brands")
    
    def predict(self, brand, model, year, mileage, fuel, transmission, condition, owner):
        """Predict price"""
        if not self.is_trained:
            self.train()
        
        try:
            # Prepare input
            input_df = pd.DataFrame([{
                'Brand': brand,
                'Model': model,
                'Year': year,
                'Mileage': mileage,
                'Fuel_Type': fuel,
                'Transmission': transmission,
                'Condition': condition,
                'Owner_Type': owner
            }])
            
            # Encode categorical variables
            for col in ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Condition', 'Owner_Type']:
                if col in self.encoders and hasattr(self.encoders[col], 'classes_'):
                    try:
                        input_df[col] = self.encoders[col].transform(input_df[col])
                    except ValueError:
                        # Handle unseen labels
                        input_df[col] = 0
                else:
                    input_df[col] = 0
            
            # Scale numerical features
            try:
                input_df[['Year', 'Mileage']] = self.scaler.transform(input_df[['Year', 'Mileage']])
            except:
                pass
            
            # Predict
            prediction = self.model.predict(input_df)[0]
            
            return max(50000, int(prediction))
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            # Fallback: calculate manually
            return self.manual_price_calculation(brand, model, year, mileage, condition, owner)
    
    def manual_price_calculation(self, brand, model, year, mileage, condition, owner):
        """Fallback manual calculation"""
        # Get base price
        market_range = get_market_price(brand, model)
        base_price = sum(market_range) / 2
        
        # Age depreciation
        current_year = datetime.now().year
        age = current_year - year
        price = base_price * (0.88 ** age)
        
        # Mileage adjustment
        price *= max(0.4, 1 - (mileage / 300000))
        
        # Condition adjustment
        condition_factors = {"Excellent": 1.1, "Very Good": 1.05, "Good": 1.0, "Fair": 0.9, "Poor": 0.75}
        price *= condition_factors.get(condition, 1.0)
        
        # Owner adjustment
        owner_factors = {"First": 1.08, "Second": 1.0, "Third": 0.93, "Fourth & Above": 0.85}
        price *= owner_factors.get(owner, 1.0)
        
        return max(50000, int(price))

# ========================================
# UI FUNCTIONS
# ========================================

def get_market_price(brand, model):
    """Get market price range"""
    if brand in CAR_DATABASE:
        models = CAR_DATABASE[brand]['models']
        if model in models:
            idx = models.index(model)
            return CAR_DATABASE[brand]['prices'][idx]
    return [300000, 500000]

def show_input_form():
    """Show car input form"""
    st.subheader("🚗 Enter Car Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox("Brand", list(CAR_DATABASE.keys()))
        model = st.selectbox("Model", CAR_DATABASE[brand]['models'])
        
        current_year = datetime.now().year
        year = st.number_input("Year", min_value=2000, max_value=current_year, value=current_year-3)
        
        fuel = st.selectbox("Fuel Type", FUEL_TYPES)
        transmission = st.selectbox("Transmission", TRANSMISSIONS)
    
    with col2:
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=30000, step=1000)
        condition = st.selectbox("Condition", CONDITIONS)
        owner = st.selectbox("Owner Type", OWNERS)
        color = st.selectbox("Color", COLORS)
    
    return brand, model, year, mileage, fuel, transmission, condition, owner

def show_price_analysis(predicted, market_range, confidence):
    """Show price analysis"""
    st.subheader("💰 Price Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Predicted Price", f"₹{predicted:,.0f}")
    
    with col2:
        st.metric("Market Low", f"₹{market_range[0]:,.0f}")
    
    with col3:
        st.metric("Market High", f"₹{market_range[1]:,.0f}")
    
    with col4:
        st.metric("Confidence", f"{confidence}%")
    
    # Comparison
    market_avg = sum(market_range) / 2
    diff = predicted - market_avg
    diff_pct = (diff / market_avg) * 100
    
    if abs(diff_pct) < 5:
        st.success(f"✅ Fair Price! Within {abs(diff_pct):.1f}% of market average")
    elif diff < 0:
        st.info(f"📉 Below market by ₹{abs(diff):,.0f} ({abs(diff_pct):.1f}%)")
    else:
        st.warning(f"📈 Above market by ₹{abs(diff):,.0f} ({abs(diff_pct):.1f}%)")

def show_history():
    """Show search history"""
    st.subheader("📋 Search History")
    
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if not st.session_state.history:
        st.info("No searches yet. Start predicting prices!")
        return
    
    df = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(df, use_container_width=True)
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

def add_to_history(brand, model, year, predicted):
    """Add search to history"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    st.session_state.history.append({
        'Time': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'Brand': brand,
        'Model': model,
        'Year': year,
        'Predicted Price': f"₹{predicted:,.0f}"
    })
    
    # Keep last 50
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[-50:]

def show_statistics():
    """Show database statistics"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Database Stats")
    
    total_brands = len(CAR_DATABASE)
    total_models = sum(len(CAR_DATABASE[b]['models']) for b in CAR_DATABASE)
    
    st.sidebar.info(f"""
    **Coverage:**
    - 🚗 {total_brands} Brands
    - 🎯 {total_models} Models
    - ✅ All Available
    """)

# ========================================
# MAIN APP
# ========================================

def main():
    # Initialize
    if 'predictor' not in st.session_state:
        st.session_state.predictor = CarPricePredictor()
    
    # Header
    st.title("🚗 Smart Car Pricing System")
    st.markdown("### AI-Powered Car Price Prediction for Indian Market")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/car.png")
        page = st.radio("📍 Navigation", [
            "Price Prediction",
            "Search History",
            "Train Model"
        ])
        
        show_statistics()
    
    # Pages
    if page == "Price Prediction":
        brand, model, year, mileage, fuel, transmission, condition, owner = show_input_form()
        
        if st.button("🎯 Predict Price", type="primary", use_container_width=True):
            try:
                with st.spinner("Calculating..."):
                    # Get prediction
                    predicted = st.session_state.predictor.predict(
                        brand, model, year, mileage, fuel, transmission, condition, owner
                    )
                    
                    # Get market range
                    market_range = get_market_price(brand, model)
                    
                    # Calculate confidence
                    age = datetime.now().year - year
                    confidence = min(95, max(75, 90 - age * 2 - (mileage // 50000) * 3))
                    
                    # Show results
                    show_price_analysis(predicted, market_range, confidence)
                    
                    # Add to history
                    add_to_history(brand, model, year, predicted)
                    
                    st.balloons()
                    
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
                st.info("Please try training the model first from the 'Train Model' page.")
    
    elif page == "Search History":
        show_history()
    
    elif page == "Train Model":
        st.subheader("🤖 Train AI Model")
        
        st.info("""
        **Machine Learning Features:**
        - Random Forest Algorithm
        - Trained on comprehensive data
        - Factors: Age, Mileage, Condition, etc.
        """)
        
        if st.button("🚀 Train Now", type="primary"):
            st.session_state.predictor.train()

if __name__ == "__main__":
    main()
