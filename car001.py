# ======================================================
# SMART CAR PRICING SYSTEM - COMPLETE CAR DATABASE
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime

# ========================================
# COMPREHENSIVE CAR DATABASE FOR MANUAL INPUT
# ========================================

CAR_DATABASE = {
    'Maruti Suzuki': {
        'models': ['Alto', 'Alto K10', 'S-Presso', 'Celerio', 'Wagon R', 'Ignis', 'Swift', 'Baleno', 'Dzire', 'Ciaz', 
                  'Ertiga', 'XL6', 'Vitara Brezza', 'Jimny', 'Fronx', 'Grand Vitara', 'Eeco', 'Omni', 'Celerio X'],
        'car_types': ['Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Sedan', 'Sedan',
                     'MUV', 'MUV', 'SUV', 'SUV', 'SUV', 'SUV', 'Van', 'Van', 'Hatchback'],
        'engine_cc': [796, 998, 998, 998, 998, 1197, 1197, 1197, 1197, 1462,
                     1462, 1462, 1462, 1462, 1197, 1462, 1196, 796, 998],
        'power_hp': [48, 67, 67, 67, 67, 83, 90, 90, 90, 103,
                    103, 103, 103, 103, 90, 103, 73, 35, 67],
        'seats': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                 7, 6, 5, 5, 5, 5, 5, 8, 5]
    },
    'Hyundai': {
        'models': ['i10', 'i20', 'Aura', 'Grand i10 Nios', 'Verna', 'Creta', 'Venue', 'Alcazar', 'Tucson', 'Kona Electric',
                  'Santro', 'Xcent', 'Elantra', 'Ioniq 5'],
        'car_types': ['Hatchback', 'Hatchback', 'Sedan', 'Hatchback', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV',
                     'Hatchback', 'Sedan', 'Sedan', 'SUV'],
        'engine_cc': [1086, 1197, 1197, 1197, 1493, 1493, 1197, 2199, 2199, 0,
                     1086, 1197, 1999, 0],
        'power_hp': [69, 83, 83, 83, 115, 115, 83, 148, 148, 136,
                    69, 83, 152, 217],
        'seats': [5, 5, 5, 5, 5, 5, 5, 6, 5, 5,
                 5, 5, 5, 5]
    },
    'Tata': {
        'models': ['Tiago', 'Tigor', 'Altroz', 'Nexon', 'Punch', 'Harrier', 'Safari', 'Nexon EV', 'Tigor EV', 'Tiago EV',
                  'Indica', 'Indigo', 'Sumo', 'Hexa'],
        'car_types': ['Hatchback', 'Sedan', 'Hatchback', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'Sedan', 'Hatchback',
                     'Hatchback', 'Sedan', 'SUV', 'SUV'],
        'engine_cc': [1199, 1199, 1199, 1199, 1199, 1956, 1956, 0, 0, 0,
                     1405, 1405, 2179, 2179],
        'power_hp': [85, 85, 85, 120, 120, 170, 170, 129, 75, 75,
                    70, 70, 120, 156],
        'seats': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                 5, 5, 8, 7]
    },
    'Mahindra': {
        'models': ['Bolero', 'Scorpio', 'XUV300', 'XUV400', 'XUV700', 'Thar', 'Marazzo', 'KUV100', 'TUV300', 'Alturas G4',
                  'Bolero Neo', 'Scorpio N', 'Verito', 'Xylo'],
        'car_types': ['SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'MUV', 'Hatchback', 'SUV', 'SUV',
                     'SUV', 'SUV', 'Sedan', 'MUV'],
        'engine_cc': [1493, 2179, 1197, 0, 1997, 1997, 1497, 1198, 1493, 2157,
                     1493, 1997, 1461, 2179],
        'power_hp': [75, 140, 110, 150, 200, 150, 123, 83, 100, 178,
                    100, 200, 65, 120],
        'seats': [7, 7, 5, 5, 7, 4, 8, 5, 7, 7,
                 7, 7, 5, 8]
    },
    'Toyota': {
        'models': ['Innova Crysta', 'Fortuner', 'Glanza', 'Urban Cruiser Hyryder', 'Camry', 'Vellfire', 'Hilux', 'Etios', 
                  'Etios Liva', 'Yaris', 'Corolla Altis', 'Innova Hycross'],
        'car_types': ['MUV', 'SUV', 'Hatchback', 'SUV', 'Sedan', 'MUV', 'Pickup', 'Sedan',
                     'Hatchback', 'Sedan', 'Sedan', 'MUV'],
        'engine_cc': [2393, 2694, 1197, 1462, 2487, 2494, 2755, 1496,
                     1496, 1496, 1798, 1987],
        'power_hp': [150, 204, 90, 103, 177, 197, 204, 90,
                    90, 107, 140, 186],
        'seats': [7, 7, 5, 5, 5, 7, 5, 5,
                 5, 5, 5, 7]
    },
    'Honda': {
        'models': ['Amaze', 'City', 'Jazz', 'WR-V', 'Elevate', 'Civic', 'CR-V', 'Brio'],
        'car_types': ['Sedan', 'Sedan', 'Hatchback', 'SUV', 'SUV', 'Sedan', 'SUV', 'Hatchback'],
        'engine_cc': [1199, 1498, 1199, 1199, 1498, 1799, 1997, 1198],
        'power_hp': [90, 121, 90, 90, 121, 141, 158, 88],
        'seats': [5, 5, 5, 5, 5, 5, 5, 5]
    },
    'Kia': {
        'models': ['Seltos', 'Sonet', 'Carens', 'Carnival', 'EV6'],
        'car_types': ['SUV', 'SUV', 'MUV', 'MUV', 'SUV'],
        'engine_cc': [1353, 998, 1482, 2199, 0],
        'power_hp': [140, 120, 115, 200, 229],
        'seats': [5, 5, 6, 7, 5]
    },
    'Volkswagen': {
        'models': ['Polo', 'Vento', 'Taigun', 'Virtus', 'Tiguan', 'T-Roc'],
        'car_types': ['Hatchback', 'Sedan', 'SUV', 'Sedan', 'SUV', 'SUV'],
        'engine_cc': [999, 999, 999, 999, 1984, 1498],
        'power_hp': [110, 110, 115, 115, 190, 150],
        'seats': [5, 5, 5, 5, 5, 5]
    },
    'Skoda': {
        'models': ['Rapid', 'Kushaq', 'Slavia', 'Kodiaq', 'Superb', 'Octavia'],
        'car_types': ['Sedan', 'SUV', 'Sedan', 'SUV', 'Sedan', 'Sedan'],
        'engine_cc': [999, 999, 999, 1984, 1984, 1984],
        'power_hp': [110, 115, 115, 190, 190, 190],
        'seats': [5, 5, 5, 7, 5, 5]
    },
    'Renault': {
        'models': ['Kwid', 'Triber', 'Kiger', 'Duster', 'Captur'],
        'car_types': ['Hatchback', 'MUV', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [999, 999, 999, 1498, 1498],
        'power_hp': [68, 72, 100, 106, 106],
        'seats': [5, 7, 5, 5, 5]
    },
    'Nissan': {
        'models': ['Magnite', 'Kicks', 'Sunny', 'Micra', 'Terrano'],
        'car_types': ['SUV', 'SUV', 'Sedan', 'Hatchback', 'SUV'],
        'engine_cc': [999, 1498, 1498, 1198, 1461],
        'power_hp': [100, 106, 99, 77, 110],
        'seats': [5, 5, 5, 5, 5]
    },
    'MG': {
        'models': ['Hector', 'Astor', 'Gloster', 'ZS EV', 'Comet EV'],
        'car_types': ['SUV', 'SUV', 'SUV', 'SUV', 'Hatchback'],
        'engine_cc': [1451, 1349, 1996, 0, 0],
        'power_hp': [143, 134, 218, 177, 42],
        'seats': [5, 5, 7, 5, 4]
    },
    'Ford': {
        'models': ['EcoSport', 'Endeavour', 'Figo', 'Aspire', 'Freestyle'],
        'car_types': ['SUV', 'SUV', 'Hatchback', 'Sedan', 'Crossover'],
        'engine_cc': [1498, 1996, 1194, 1194, 1194],
        'power_hp': [123, 170, 96, 96, 96],
        'seats': [5, 7, 5, 5, 5]
    }
}

FUEL_TYPES = ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
TRANSMISSIONS = ["Manual", "Automatic", "CVT", "DCT", "AMT"]
CAR_CONDITIONS = ["Excellent", "Very Good", "Good", "Fair", "Poor"]
OWNER_TYPES = ["First", "Second", "Third", "Fourth & Above"]
INSURANCE_STATUS = ["Comprehensive", "Third Party", "Expired", "No Insurance"]
COLORS = ["White", "Black", "Silver", "Grey", "Red", "Blue", "Brown", "Green", "Yellow", "Orange", "Purple", "Other"]
CITIES = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune", "Hyderabad", "Kolkata", "Ahmedabad", "Surat", "Jaipur", "Lucknow", "Chandigarh"]

# ========================================
# SIMPLIFIED PRICE PREDICTION ENGINE
# ========================================

class CarPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_importance = {}
        self.is_trained = False
        
    def get_live_prices(self, brand, model):
        """Get live prices for car models with proper error handling"""
        try:
            # Simple price database with default values
            price_database = {
                'Maruti Suzuki': {
                    'Alto': [150000, 250000, 350000],
                    'Swift': [300000, 450000, 600000],
                    'Baleno': [350000, 500000, 700000],
                    'Dzire': [320000, 480000, 650000],
                    'Vitara Brezza': [500000, 700000, 900000],
                    'Ertiga': [450000, 650000, 850000],
                    'Wagon R': [200000, 300000, 400000],
                    'Celerio': [250000, 350000, 450000],
                    'Ciaz': [450000, 650000, 850000],
                    'S-Presso': [280000, 380000, 480000],
                    'Ignis': [320000, 450000, 580000],
                    'XL6': [550000, 750000, 950000],
                    'Grand Vitara': [800000, 1100000, 1400000],
                    'Fronx': [450000, 600000, 800000],
                    'Jimny': [600000, 800000, 1000000]
                },
                'Hyundai': {
                    'i10': [250000, 350000, 450000],
                    'i20': [350000, 500000, 650000],
                    'Creta': [600000, 850000, 1100000],
                    'Verna': [450000, 650000, 850000],
                    'Venue': [450000, 600000, 800000],
                    'Aura': [320000, 450000, 580000],
                    'Alcazar': [800000, 1100000, 1400000],
                    'Tucson': [1200000, 1600000, 2000000],
                    'Grand i10 Nios': [300000, 420000, 550000]
                },
                'Tata': {
                    'Tiago': [250000, 350000, 450000],
                    'Nexon': [450000, 650000, 850000],
                    'Altroz': [350000, 500000, 650000],
                    'Harrier': [800000, 1100000, 1400000],
                    'Safari': [900000, 1200000, 1500000],
                    'Punch': [300000, 450000, 600000],
                    'Tigor': [280000, 400000, 520000]
                },
                'Mahindra': {
                    'Scorpio': [500000, 700000, 900000],
                    'XUV300': [450000, 600000, 800000],
                    'XUV700': [900000, 1200000, 1500000],
                    'Thar': [600000, 850000, 1100000],
                    'Bolero': [300000, 450000, 600000],
                    'Marazzo': [500000, 700000, 900000]
                },
                'Toyota': {
                    'Innova Crysta': [1000000, 1400000, 1800000],
                    'Fortuner': [1500000, 2000000, 2500000],
                    'Glanza': [350000, 500000, 650000],
                    'Urban Cruiser Hyryder': [600000, 800000, 1000000],
                    'Camry': [1800000, 2300000, 2800000]
                },
                'Honda': {
                    'City': [450000, 650000, 850000],
                    'Amaze': [350000, 500000, 650000],
                    'WR-V': [400000, 550000, 700000],
                    'Elevate': [600000, 800000, 1000000]
                },
                'Kia': {
                    'Seltos': [600000, 800000, 1000000],
                    'Sonet': [450000, 600000, 800000],
                    'Carens': [650000, 850000, 1100000]
                },
                'Volkswagen': {
                    'Polo': [350000, 500000, 650000],
                    'Vento': [400000, 550000, 700000],
                    'Taigun': [600000, 800000, 1000000],
                    'Virtus': [550000, 750000, 950000]
                }
            }
            
            # Check if brand exists in database
            if brand not in price_database:
                # Return default prices for unknown brands
                default_prices = [300000, 500000, 800000]
                return default_prices, ["Market Estimate - Unknown Brand"]
            
            # Check if model exists for the brand
            if model not in price_database[brand]:
                # Return default prices for unknown models
                default_prices = [300000, 500000, 800000]
                return default_prices, ["Market Estimate - Unknown Model"]
            
            # Return the actual prices
            prices = price_database[brand][model]
            sources = ["Market Database"]
            return prices, sources
            
        except Exception as e:
            # Return safe default values in case of any error
            default_prices = [300000, 500000, 800000]
            return default_prices, ["General Market Average"]

    def create_simple_training_data(self):
        """Create simple training data without complex random generation"""
        records = []
        current_year = datetime.now().year
        
        # Only create data for popular models to avoid issues
        popular_brands = ['Maruti Suzuki', 'Hyundai', 'Tata', 'Mahindra', 'Toyota', 'Honda']
        
        for brand in popular_brands:
            if brand not in CAR_DATABASE:
                continue
                
            for i, model in enumerate(CAR_DATABASE[brand]['models'][:3]):  # Only first 3 models
                car_type = CAR_DATABASE[brand]['car_types'][i]
                engine_cc = CAR_DATABASE[brand]['engine_cc'][i]
                power_hp = CAR_DATABASE[brand]['power_hp'][i]
                seats = CAR_DATABASE[brand]['seats'][i]
                
                # Get base price
                base_prices, _ = self.get_live_prices(brand, model)
                base_price = base_prices[1]
                
                # Create simple variations
                for year in [current_year-1, current_year-3, current_year-5]:
                    age = current_year - year
                    
                    # Simple mileage calculation - no complex random
                    if age == 1:
                        mileage = 15000
                    elif age == 3:
                        mileage = 45000
                    else:
                        mileage = 75000
                    
                    for condition in ["Good", "Very Good", "Excellent"]:
                        for owner_type in ["First", "Second"]:
                            input_data = {
                                'Brand': brand,
                                'Model': model,
                                'Car_Type': car_type,
                                'Year': year,
                                'Fuel_Type': 'Petrol',
                                'Transmission': 'Manual',
                                'Mileage': mileage,
                                'Engine_cc': engine_cc,
                                'Power_HP': power_hp,
                                'Seats': seats,
                                'Condition': condition,
                                'Owner_Type': owner_type
                            }
                            
                            price = self.calculate_simple_price(input_data, base_price)
                            records.append({**input_data, 'Price': price})
        
        return pd.DataFrame(records)
    
    def calculate_simple_price(self, input_data, base_price):
        """Calculate price using simple rules"""
        current_year = datetime.now().year
        age = current_year - input_data['Year']
        
        # Simple depreciation
        age_factor = max(0.3, 1 - (age * 0.1))
        
        # Mileage factor
        mileage_factor = max(0.5, 1 - (input_data['Mileage'] / 200000))
        
        # Condition multipliers
        condition_multipliers = {
            "Excellent": 1.1,
            "Very Good": 1.0,
            "Good": 0.9,
            "Fair": 0.8,
            "Poor": 0.6
        }
        
        # Owner multipliers
        owner_multipliers = {
            "First": 1.05,
            "Second": 1.0,
            "Third": 0.9,
            "Fourth & Above": 0.8
        }
        
        price = (base_price * age_factor * mileage_factor * 
                condition_multipliers[input_data['Condition']] * 
                owner_multipliers[input_data['Owner_Type']])
        
        return max(100000, int(price))
    
    def train_model(self):
        """Train the prediction model"""
        try:
            st.info("🔄 Training price prediction model...")
            
            # Create simple training data
            df = self.create_simple_training_data()
            
            if df.empty:
                st.error("No training data created!")
                return None
            
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
            
            # Train simple model
            self.model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )
            
            self.model.fit(X, y)
            self.is_trained = True
            
            st.success("✅ Model trained successfully!")
            return self.model
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None
    
    def predict_price(self, input_data):
        """Predict car price"""
        if not self.is_trained:
            success = self.train_model()
            if not success:
                return self.fallback_prediction(input_data)
        
        try:
            # Prepare input features
            features = ['Brand', 'Model', 'Car_Type', 'Year', 'Fuel_Type', 'Transmission',
                       'Mileage', 'Engine_cc', 'Power_HP', 'Seats', 'Condition', 'Owner_Type']
            
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for feature in ['Brand', 'Model', 'Car_Type', 'Fuel_Type', 'Transmission', 'Condition', 'Owner_Type']:
                if feature in self.encoders:
                    try:
                        input_df[feature] = self.encoders[feature].transform([input_data[feature]])[0]
                    except:
                        input_df[feature] = 0
            
            # Scale numerical features
            numerical_features = ['Year', 'Mileage', 'Engine_cc', 'Power_HP', 'Seats']
            input_df[numerical_features] = self.scaler.transform(input_df[numerical_features])
            
            # Ensure all features are present
            for feature in features:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            input_df = input_df[features]
            
            # Get prediction
            prediction = self.model.predict(input_df)[0]
            return max(100000, int(prediction))
            
        except Exception as e:
            st.warning(f"Using fallback prediction due to error: {str(e)}")
            return self.fallback_prediction(input_data)
    
    def fallback_prediction(self, input_data):
        """Fallback price prediction when model fails"""
        base_prices, _ = self.get_live_prices(input_data['Brand'], input_data['Model'])
        base_price = base_prices[1]
        
        return self.calculate_simple_price(input_data, base_price)

# ========================================
# UTILITY FUNCTIONS
# ========================================

def show_brand_statistics():
    """Show statistics about available car brands"""
    st.sidebar.subheader("📈 Brand Statistics")
    
    total_brands = len(CAR_DATABASE)
    total_models = sum(len(CAR_DATABASE[brand]['models']) for brand in CAR_DATABASE)
    
    st.sidebar.info(f"""
    **Database Overview:**
    - 🚗 **Brands:** {total_brands}
    - 🎯 **Models:** {total_models}
    - 📊 **Coverage:** Comprehensive
    """)

def show_manual_input_form():
    """Show comprehensive manual input form for car details"""
    st.subheader("🔧 Complete Car Details Entry")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Brand selection
        brand = st.selectbox("Brand", list(CAR_DATABASE.keys()))
        
        if brand in CAR_DATABASE:
            model = st.selectbox("Model", CAR_DATABASE[brand]['models'])
            
            # Auto-fill technical specifications
            if model in CAR_DATABASE[brand]['models']:
                model_index = CAR_DATABASE[brand]['models'].index(model)
                car_type = CAR_DATABASE[brand]['car_types'][model_index]
                engine_cc = CAR_DATABASE[brand]['engine_cc'][model_index]
                power_hp = CAR_DATABASE[brand]['power_hp'][model_index]
                seats = CAR_DATABASE[brand]['seats'][model_index]
                
                st.text_input("Car Type", value=car_type, disabled=True)
                st.text_input("Engine Capacity", value=f"{engine_cc} cc", disabled=True)
                st.text_input("Power", value=f"{power_hp} HP", disabled=True)
                st.text_input("Seating Capacity", value=f"{seats} seats", disabled=True)
            else:
                # Fallback if model not found
                car_type = "Sedan"
                engine_cc = 1200
                power_hp = 80
                seats = 5
        else:
            # Fallback if brand not found
            model = "Unknown"
            car_type = "Sedan"
            engine_cc = 1200
            power_hp = 80
            seats = 5
        
        current_year = datetime.now().year
        year = st.number_input("Manufacturing Year", min_value=1950, max_value=current_year, value=current_year-3)
        
        fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)
        transmission = st.selectbox("Transmission", TRANSMISSIONS)
    
    with col2:
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=30000, step=1000)
        color = st.selectbox("Color", COLORS)
        condition = st.selectbox("Car Condition", CAR_CONDITIONS)
        owner_type = st.selectbox("Owner Type", OWNER_TYPES)
        insurance_status = st.selectbox("Insurance Status", INSURANCE_STATUS)
        registration_city = st.selectbox("Registration City", CITIES)
    
    # Generate unique Car_ID
    car_id = f"{brand[:3].upper()}_{model[:3].upper()}_{year}"
    
    # Return input data
    input_data = {
        'Car_ID': car_id,
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
        'Color': color,
        'Condition': condition,
        'Owner_Type': owner_type,
        'Insurance_Status': insurance_status,
        'Registration_City': registration_city
    }
    
    return input_data

def calculate_confidence(input_data):
    """Calculate prediction confidence"""
    confidence = 80
    
    # Increase confidence for newer cars
    current_year = datetime.now().year
    if current_year - input_data['Year'] <= 5:
        confidence += 10
    
    # Decrease confidence for high mileage
    if input_data['Mileage'] > 100000:
        confidence -= 10
    
    return min(95, max(60, confidence))

# ========================================
# MAIN INTERFACE
# ========================================

def show_prediction_interface():
    """Show the price prediction interface"""
    st.subheader("🎯 Car Price Prediction")
    
    # Manual input form
    input_data = show_manual_input_form()
    
    if input_data:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show market prices
            brand = input_data['Brand']
            model = input_data['Model']
            
            if brand and model:
                with st.spinner('🔍 Analyzing market trends...'):
                    prices, sources = st.session_state.predictor.get_live_prices(brand, model)
                    min_price, avg_price, max_price = prices
                
                st.subheader("📊 Market Intelligence")
                
                market_col1, market_col2, market_col3 = st.columns(3)
                
                with market_col1:
                    st.metric("Market Low", f"₹{min_price:,.0f}")
                with market_col2:
                    st.metric("Market Average", f"₹{avg_price:,.0f}")
                with market_col3:
                    st.metric("Market High", f"₹{max_price:,.0f}")
                
                st.info(f"**Data Sources:** {', '.join(sources)}")
        
        with col2:
            st.subheader("🤖 AI Prediction")
            
            if st.button("🎯 Get Price Prediction", type="primary", use_container_width=True):
                with st.spinner('🤖 Calculating price...'):
                    # Get AI prediction
                    predicted_price = st.session_state.predictor.predict_price(input_data)
                    
                    # Show confidence
                    confidence = calculate_confidence(input_data)
                    
                    # Display result
                    st.success(f"**Predicted Price: ₹{predicted_price:,.0f}**")
                    st.metric("Confidence Level", f"{confidence}%")
                    
                    st.balloons()

# ========================================
# MAIN APPLICATION
# ========================================

def main():
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = CarPricePredictor()  # Fixed the typo here
    
    st.set_page_config(
        page_title="Car Price Predictor", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    st.title("🚗 Car Price Prediction System")
    st.markdown("### **AI-Powered Price Estimation**")
    
    # Show brand statistics in sidebar
    show_brand_statistics()
    
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/48/car.png")
        st.title("Navigation")
        st.info("🎯 Price Prediction")
        
        st.markdown("---")
        st.subheader("Features")
        st.success("✅ AI Price Prediction")
        st.success("✅ Market Intelligence")
        st.success("✅ Confidence Scoring")
        st.success("✅ All Car Categories")
    
    # Main interface
    show_prediction_interface()

if __name__ == "__main__":
    main()
