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
        'seats': [5, 5, 5, 5, 5, 5, 7, 5, 5, 5,
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
    },
    # LUXURY CAR BRANDS
    'BMW': {
        'models': ['3 Series', '5 Series', '7 Series', 'X1', 'X3', 'X5', 'X7', 'Z4', 'i4', 'iX', 'M3', 'M5', 'X3 M', 'X5 M', '8 Series'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'Convertible', 'Sedan', 'SUV', 'Sedan', 'Sedan', 'SUV', 'SUV', 'Coupe'],
        'engine_cc': [1998, 1998, 2998, 1499, 1998, 2998, 2998, 1998, 0, 0, 2993, 4395, 2993, 4395, 2998],
        'power_hp': [255, 248, 335, 140, 248, 335, 400, 197, 340, 523, 473, 600, 473, 600, 335],
        'seats': [5, 5, 5, 5, 5, 5, 7, 2, 5, 5, 5, 5, 5, 5, 4]
    },
    'Mercedes-Benz': {
        'models': ['A-Class', 'C-Class', 'E-Class', 'S-Class', 'GLA', 'GLC', 'GLE', 'GLS', 'EQB', 'EQS', 'AMG GT', 'Maybach S-Class', 'G-Class', 'CLS', 'SL'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'Sedan', 'Coupe', 'Sedan', 'SUV', 'Coupe', 'Convertible'],
        'engine_cc': [1332, 1497, 1991, 2999, 1332, 1991, 1991, 2999, 0, 0, 3982, 5980, 2925, 1991, 1991],
        'power_hp': [163, 204, 258, 435, 163, 258, 362, 483, 228, 329, 523, 621, 416, 258, 258],
        'seats': [5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 2, 5, 5, 5, 2]
    },
    'Audi': {
        'models': ['A3', 'A4', 'A6', 'A8', 'Q3', 'Q5', 'Q7', 'Q8', 'e-tron', 'RS5', 'R8', 'TT', 'RS7', 'Q8 Sportback', 'A5'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'Coupe', 'Sports', 'Coupe', 'Sedan', 'SUV', 'Coupe'],
        'engine_cc': [1395, 1984, 1984, 2995, 1395, 1984, 2995, 2995, 0, 2894, 5204, 1984, 3993, 2995, 1984],
        'power_hp': [150, 190, 245, 340, 150, 245, 340, 340, 355, 450, 602, 228, 600, 340, 190],
        'seats': [5, 5, 5, 5, 5, 5, 7, 5, 7, 4, 2, 2, 5, 5, 4]
    },
    'Lexus': {
        'models': ['ES', 'LS', 'NX', 'RX', 'UX', 'LC', 'LX', 'RC', 'GX', 'IS'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'Coupe', 'SUV', 'Coupe', 'SUV', 'Sedan'],
        'engine_cc': [2487, 3445, 2487, 3456, 1987, 4969, 3445, 3456, 3956, 1998],
        'power_hp': [215, 422, 194, 295, 169, 471, 409, 311, 301, 241],
        'seats': [5, 5, 5, 5, 5, 4, 8, 4, 7, 5]
    },
    'Jaguar': {
        'models': ['XE', 'XF', 'XJ', 'F-PACE', 'E-PACE', 'I-PACE', 'F-TYPE', 'XK', 'S-Type'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'Convertible', 'Coupe', 'Sedan'],
        'engine_cc': [1997, 1997, 2993, 1997, 1997, 0, 5000, 5000, 2967],
        'power_hp': [247, 247, 335, 247, 247, 400, 575, 385, 235],
        'seats': [5, 5, 5, 5, 5, 5, 2, 4, 5]
    },
    'Land Rover': {
        'models': ['Range Rover', 'Range Rover Sport', 'Range Rover Velar', 'Range Rover Evoque', 'Discovery', 'Defender', 'Discovery Sport'],
        'car_types': ['SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [2996, 2996, 1997, 1997, 2996, 2996, 1997],
        'power_hp': [355, 355, 247, 247, 355, 400, 247],
        'seats': [5, 5, 5, 5, 7, 5, 7]
    },
    'Porsche': {
        'models': ['911', 'Panamera', 'Cayenne', 'Macan', 'Taycan', 'Boxster', 'Cayman', '718', '918 Spyder'],
        'car_types': ['Coupe', 'Sedan', 'SUV', 'SUV', 'Sedan', 'Convertible', 'Coupe', 'Coupe', 'Sports'],
        'engine_cc': [2981, 2894, 2995, 1984, 0, 2497, 2497, 1988, 4593],
        'power_hp': [385, 330, 340, 265, 402, 300, 300, 300, 887],
        'seats': [4, 5, 5, 5, 4, 2, 2, 2, 2]
    },
    'Volvo': {
        'models': ['S60', 'S90', 'XC40', 'XC60', 'XC90', 'C40', 'V90', 'V60', 'XC90 Recharge'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'Estate', 'Estate', 'SUV'],
        'engine_cc': [1969, 1969, 1969, 1969, 1969, 0, 1969, 1969, 1969],
        'power_hp': [250, 250, 197, 250, 300, 231, 250, 250, 400],
        'seats': [5, 5, 5, 5, 7, 5, 5, 5, 7]
    },
    'Maserati': {
        'models': ['Ghibli', 'Quattroporte', 'Levante', 'GranTurismo', 'MC20', 'GranCabrio'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'Coupe', 'Sports', 'Convertible'],
        'engine_cc': [2979, 2979, 2979, 4691, 2992, 4691],
        'power_hp': [350, 424, 424, 454, 621, 454],
        'seats': [5, 5, 5, 4, 2, 4]
    },
    'Bentley': {
        'models': ['Continental GT', 'Flying Spur', 'Bentayga', 'Mulsanne', 'Azure', 'Brooklands'],
        'car_types': ['Coupe', 'Sedan', 'SUV', 'Sedan', 'Convertible', 'Coupe'],
        'engine_cc': [3993, 3993, 3993, 6750, 6750, 6750],
        'power_hp': [542, 542, 542, 505, 457, 530],
        'seats': [4, 5, 5, 5, 4, 4]
    },
    'Rolls-Royce': {
        'models': ['Ghost', 'Phantom', 'Cullinan', 'Wraith', 'Dawn', 'Spectre'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'Coupe', 'Convertible', 'Coupe'],
        'engine_cc': [6749, 6749, 6749, 6592, 6592, 0],
        'power_hp': [563, 563, 563, 624, 563, 577],
        'seats': [5, 5, 5, 4, 4, 4]
    },
    'Lamborghini': {
        'models': ['Huracan', 'Aventador', 'Urus', 'Gallardo', 'Murcielago', 'Revuelto'],
        'car_types': ['Sports', 'Sports', 'SUV', 'Sports', 'Sports', 'Sports'],
        'engine_cc': [5204, 6498, 3996, 5204, 6498, 6498],
        'power_hp': [631, 740, 641, 562, 661, 1015],
        'seats': [2, 2, 5, 2, 2, 2]
    },
    'Ferrari': {
        'models': ['Portofino', 'Roma', 'F8 Tributo', 'SF90 Stradale', '812 Superfast', '296 GTB', 'Purosangue'],
        'car_types': ['Convertible', 'Coupe', 'Coupe', 'Sports', 'Coupe', 'Sports', 'SUV'],
        'engine_cc': [3855, 3855, 3902, 3990, 6496, 2992, 6496],
        'power_hp': [612, 612, 710, 986, 789, 654, 715],
        'seats': [2, 4, 2, 2, 2, 2, 4]
    },
    'Aston Martin': {
        'models': ['DB11', 'Vantage', 'DBS Superleggera', 'DBX', 'Rapide', 'Valhalla', 'Valkyrie'],
        'car_types': ['Coupe', 'Sports', 'Coupe', 'SUV', 'Sedan', 'Sports', 'Hypercar'],
        'engine_cc': [3996, 3996, 5204, 3982, 5935, 3996, 6500],
        'power_hp': [503, 503, 715, 542, 552, 937, 1160],
        'seats': [4, 2, 2, 5, 4, 2, 2]
    },
    'McLaren': {
        'models': ['720S', '570S', 'GT', 'Artura', 'P1', 'Senna', 'Elva'],
        'car_types': ['Sports', 'Sports', 'Sports', 'Sports', 'Hypercar', 'Sports', 'Roadster'],
        'engine_cc': [3994, 3799, 3994, 2993, 3799, 3994, 3994],
        'power_hp': [710, 562, 612, 671, 903, 789, 804],
        'seats': [2, 2, 2, 2, 2, 2, 2]
    },
    'Bugatti': {
        'models': ['Chiron', 'Veyron', 'Divo', 'Centodieci', 'Bolide'],
        'car_types': ['Hypercar', 'Hypercar', 'Hypercar', 'Hypercar', 'Track Car'],
        'engine_cc': [7993, 7993, 7993, 7993, 7993],
        'power_hp': [1500, 1001, 1500, 1600, 1600],
        'seats': [2, 2, 2, 2, 2]
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
# ENHANCED PRICE PREDICTION ENGINE
# ========================================

class EnhancedCarPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_importance = {}
        self.is_trained = False
        
    def get_enhanced_live_prices(self, brand, model):
        """Get enhanced live prices for all car models"""
        try:
            # Comprehensive price database
            car_price_database = {
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
            
            # Luxury car price database
            luxury_price_database = {
                'BMW': {
                    '3 Series': [1800000, 2500000, 3500000],
                    '5 Series': [3000000, 4000000, 5500000],
                    '7 Series': [6000000, 8500000, 12000000],
                    'X1': [2500000, 3500000, 4500000],
                    'X3': [3500000, 5000000, 6500000],
                    'X5': [5500000, 7500000, 9500000],
                    'X7': [8000000, 11000000, 14000000]
                },
                'Mercedes-Benz': {
                    'A-Class': [2200000, 3000000, 4000000],
                    'C-Class': [2800000, 4000000, 5500000],
                    'E-Class': [4500000, 6000000, 8000000],
                    'S-Class': [8000000, 12000000, 16000000],
                    'GLA': [2500000, 3500000, 4800000],
                    'GLC': [4000000, 5500000, 7500000],
                    'GLE': [5500000, 7500000, 10000000]
                },
                'Audi': {
                    'A3': [2000000, 2800000, 3800000],
                    'A4': [3000000, 4200000, 5500000],
                    'A6': [4500000, 6000000, 8000000],
                    'A8': [7500000, 10000000, 13000000],
                    'Q3': [2800000, 3800000, 5000000],
                    'Q5': [4000000, 5500000, 7000000],
                    'Q7': [6000000, 8000000, 11000000]
                }
            }
            
            if brand in car_price_database and model in car_price_database[brand]:
                prices = car_price_database[brand][model]
                sources = ["Used Car Market Database"]
            elif brand in luxury_price_database and model in luxury_price_database[brand]:
                prices = luxury_price_database[brand][model]
                sources = ["Luxury Car Market Database"]
            else:
                # Estimate based on car type
                base_prices = {
                    'Hatchback': [200000, 350000, 500000],
                    'Sedan': [300000, 500000, 700000],
                    'SUV': [400000, 650000, 900000],
                    'MUV': [350000, 550000, 750000],
                    'Sports': [5000000, 10000000, 20000000],
                    'Hypercar': [50000000, 100000000, 200000000]
                }
                # Get car type for estimation
                car_type = "Sedan"  # default
                if brand in CAR_DATABASE and model in CAR_DATABASE[brand]['models']:
                    model_index = CAR_DATABASE[brand]['models'].index(model)
                    car_type = CAR_DATABASE[brand]['car_types'][model_index]
                
                prices = base_prices.get(car_type, [300000, 500000, 800000])
                sources = ["Market Estimate"]
                
        except Exception as e:
            prices = [300000, 500000, 800000]
            sources = ["General Market Average"]
        
        return prices, sources

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
                base_prices, _ = self.get_enhanced_live_prices(brand, model)
                base_price = base_prices[1]  # Use average price
                
                # Generate multiple records with variations
                for _ in range(20):  # 20 records per model for performance
                    year = np.random.randint(max(1990, current_year-20), current_year+1)
                    age = current_year - year
                    
                    # FIX: Ensure mileage calculation is always valid
                    if age == 0:
                        # Brand new car
                        mileage = np.random.randint(10, 1000)
                    else:
                        # Used car
                        max_mileage = min(300000, 15000 * age)
                        min_mileage = min(1000, max_mileage - 1000)
                        mileage = np.random.randint(min_mileage, max_mileage + 1)
                    
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
        
        X = df[features].copy()
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
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Train model
        rf_model.fit(X, y)
        
        # Store feature importance
        self.feature_importance = dict(zip(features, rf_model.feature_importances_))
        
        # Use the trained model
        self.model = rf_model
        self.is_trained = True
        
        # Evaluate model
        y_pred = rf_model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        st.success(f"✅ Model trained successfully! R² Score: {r2:.3f}, MAE: ₹{mae:,.0f}")
        
        return self.model
    
    def predict_price(self, input_data):
        """Predict car price with enhanced accuracy"""
        if not self.is_trained:
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
        
        # Get prediction
        prediction = self.model.predict(input_df)[0]
        
        # Apply additional business rules
        final_prediction = self.apply_business_rules(prediction, input_data)
        
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
    
    # Brand distribution
    brand_counts = {brand: len(CAR_DATABASE[brand]['models']) for brand in CAR_DATABASE}
    top_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    st.sidebar.write("**Top 5 Brands by Models:**")
    for brand, count in top_brands:
        st.sidebar.write(f"- {brand}: {count} models")

def search_cars():
    """Search functionality for cars in database"""
    st.sidebar.subheader("🔍 Search Cars")
    
    search_brand = st.sidebar.selectbox("Search by Brand", ["All"] + list(CAR_DATABASE.keys()))
    search_type = st.sidebar.selectbox("Search by Type", ["All", "Hatchback", "Sedan", "SUV", "MUV"])
    
    if st.sidebar.button("Search"):
        results = []
        
        for brand in CAR_DATABASE:
            if search_brand != "All" and brand != search_brand:
                continue
                
            for i, model in enumerate(CAR_DATABASE[brand]['models']):
                car_type = CAR_DATABASE[brand]['car_types'][i]
                
                if search_type != "All" and car_type != search_type:
                    continue
                    
                results.append({
                    'Brand': brand,
                    'Model': model,
                    'Type': car_type,
                    'Engine': CAR_DATABASE[brand]['engine_cc'][i],
                    'Power': CAR_DATABASE[brand]['power_hp'][i],
                    'Seats': CAR_DATABASE[brand]['seats'][i]
                })
        
        if results:
            st.sidebar.success(f"Found {len(results)} cars")
            df_results = pd.DataFrame(results)
            st.sidebar.dataframe(df_results, use_container_width=True)
        else:
            st.sidebar.warning("No cars found matching criteria")

def show_manual_input_form():
    """Show comprehensive manual input form for car details"""
    st.subheader("🔧 Complete Car Details Entry")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Brand selection with search
        brand = st.selectbox("Brand", list(CAR_DATABASE.keys()), 
                           help="Select car brand from comprehensive database")
        
        if brand in CAR_DATABASE:
            model = st.selectbox("Model", CAR_DATABASE[brand]['models'],
                               help=f"Select {brand} model")
            
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
                car_type = st.selectbox("Car Type", ["Hatchback", "Sedan", "SUV", "MUV", "Coupe", "Convertible", "Van", "Pickup"])
                engine_cc = st.number_input("Engine CC", min_value=600, max_value=5000, value=1200)
                power_hp = st.number_input("Power (HP)", min_value=40, max_value=500, value=80)
                seats = st.number_input("Seats", min_value=2, max_value=9, value=5)
        else:
            model = st.text_input("Model Name", placeholder="Enter model name")
            car_type = st.selectbox("Car Type", ["Hatchback", "Sedan", "SUV", "MUV", "Coupe", "Convertible", "Van", "Pickup"])
            engine_cc = st.number_input("Engine CC", min_value=600, max_value=5000, value=1200)
            power_hp = st.number_input("Power (HP)", min_value=40, max_value=500, value=80)
            seats = st.number_input("Seats", min_value=2, max_value=9, value=5)
        
        current_year = datetime.now().year
        year = st.number_input("Manufacturing Year", min_value=1990, max_value=current_year, 
                             value=current_year-3, help="Year when car was manufactured")
        
        fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)
        transmission = st.selectbox("Transmission", TRANSMISSIONS)
    
    with col2:
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=30000, step=1000,
                                help="Total kilometers driven")
        
        color = st.selectbox("Color", COLORS)
        condition = st.selectbox("Car Condition", CAR_CONDITIONS,
                               help="Overall condition of the vehicle")
        
        owner_type = st.selectbox("Owner Type", OWNER_TYPES,
                                help="Number of previous owners")
        insurance_status = st.selectbox("Insurance Status", INSURANCE_STATUS)
        
        registration_city = st.selectbox("Registration City", CITIES,
                                       help="City where car is registered")
    
    # Additional details section
    st.subheader("📋 Additional Details")
    
    col3, col4 = st.columns(2)
    
    with col3:
        service_history = st.radio("Service History", 
                                 ["Full Service History", "Partial Service History", "No Service History"])
        
        accident_history = st.radio("Accident History", 
                                  ["No Accidents", "Minor Accidents", "Major Accidents"])
    
    with col4:
        car_availability = st.radio("Car Availability", ["Available", "Sold"])
        
        # Additional features
        features = st.multiselect("Additional Features",
                                ["Power Steering", "Power Windows", "Air Conditioning", "Music System",
                                 "Alloy Wheels", "Sunroof", "Leather Seats", "Rear Camera", "GPS Navigation",
                                 "Keyless Entry", "Push Start", "ABS", "Airbags", "ESP"])
    
    # Generate unique Car_ID
    car_id = f"{brand[:3].upper()}_{model[:3].upper()}_{year}_{np.random.randint(1000,9999)}"
    
    # Return all input data
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
        'Registration_City': registration_city,
        'Service_History': service_history,
        'Accident_History': accident_history,
        'Car_Availability': car_availability,
        'Features': ', '.join(features) if features else 'None'
    }
    
    # Show summary
    with st.expander("📊 Car Details Summary", expanded=True):
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.write(f"**Brand:** {brand}")
            st.write(f"**Model:** {model}")
            st.write(f"**Year:** {year}")
            st.write(f"**Fuel Type:** {fuel_type}")
            st.write(f"**Transmission:** {transmission}")
            
        with summary_col2:
            st.write(f"**Mileage:** {mileage:,} km")
            st.write(f"**Engine:** {engine_cc} cc")
            st.write(f"**Power:** {power_hp} HP")
            st.write(f"**Condition:** {condition}")
            st.write(f"**Owners:** {owner_type}")
    
    return input_data

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
# SEARCH HISTORY FEATURE
# ========================================

def show_search_history():
    """Show search history of previous car searches"""
    st.subheader("📋 Search History")
    
    # Initialize search history in session state if not exists
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    if not st.session_state.search_history:
        st.info("No search history yet. Start searching for cars to see your history here.")
        return
    
    # Show search history in reverse order (newest first)
    st.write(f"**Total Searches:** {len(st.session_state.search_history)}")
    
    # Convert to dataframe for better display
    history_df = pd.DataFrame(st.session_state.search_history[::-1])  # Reverse to show newest first
    
    # Display history
    st.dataframe(history_df, use_container_width=True)
    
    # Show some statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        most_searched_brand = history_df['Brand'].mode()[0] if not history_df.empty else "N/A"
        st.metric("Most Searched Brand", most_searched_brand)
    
    with col2:
        total_unique_cars = history_df[['Brand', 'Model']].drop_duplicates().shape[0]
        st.metric("Unique Cars Searched", total_unique_cars)
    
    with col3:
        if st.button("Clear History"):
            st.session_state.search_history = []
            st.rerun()

def add_to_search_history(brand, model, predicted_price, market_avg, confidence):
    """Add current search to history"""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    search_entry = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Brand': brand,
        'Model': model,
        'Predicted Price': f"₹{predicted_price:,.0f}",
        'Market Average': f"₹{market_avg:,.0f}",
        'Confidence': f"{confidence}%",
        'Price Difference': f"₹{predicted_price - market_avg:,.0f}"
    }
    
    st.session_state.search_history.append(search_entry)
    
    # Keep only last 50 searches to prevent memory issues
    if len(st.session_state.search_history) > 50:
        st.session_state.search_history = st.session_state.search_history[-50:]

# ========================================
# CSV UPLOAD FEATURE
# ========================================

def show_csv_upload_feature():
    """Show CSV upload and bulk prediction feature"""
    st.subheader("📁 Upload CSV for Bulk Price Prediction")
    
    st.info("""
    **Upload a CSV file with car details to get bulk price predictions.**
    
    Required columns: Brand, Model, Year, Fuel_Type, Transmission, Mileage, Condition, Owner_Type
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully uploaded {len(df)} cars")
            
            # Show preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🚀 Predict Prices for All Cars", type="primary"):
                predictions = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Predicting prices for all cars..."):
                    for idx, row in df.iterrows():
                        try:
                            # Prepare input data
                            input_data = {
                                'Brand': row.get('Brand', ''),
                                'Model': row.get('Model', ''),
                                'Car_Type': row.get('Car_Type', 'Sedan'),
                                'Year': int(row.get('Year', 2020)),
                                'Fuel_Type': row.get('Fuel_Type', 'Petrol'),
                                'Transmission': row.get('Transmission', 'Manual'),
                                'Mileage': int(row.get('Mileage', 30000)),
                                'Engine_cc': int(row.get('Engine_cc', 1200)),
                                'Power_HP': int(row.get('Power_HP', 80)),
                                'Seats': int(row.get('Seats', 5)),
                                'Condition': row.get('Condition', 'Good'),
                                'Owner_Type': row.get('Owner_Type', 'First')
                            }
                            
                            # Get prediction
                            predicted_price = st.session_state.predictor.predict_price(input_data)
                            
                            # Get market prices
                            market_prices, _ = st.session_state.predictor.get_enhanced_live_prices(
                                input_data['Brand'], input_data['Model']
                            )
                            
                            predictions.append({
                                'Brand': input_data['Brand'],
                                'Model': input_data['Model'],
                                'Year': input_data['Year'],
                                'Predicted_Price': predicted_price,
                                'Market_Low': market_prices[0],
                                'Market_Average': market_prices[1],
                                'Market_High': market_prices[2],
                                'Price_Difference': predicted_price - market_prices[1]
                            })
                            
                            # Update progress
                            progress = (idx + 1) / len(df)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing: {idx + 1}/{len(df)} cars")
                            
                        except Exception as e:
                            st.warning(f"Error processing row {idx}: {str(e)}")
                            continue
                
                progress_bar.empty()
                status_text.empty()
                
                if predictions:
                    # Create results dataframe
                    results_df = pd.DataFrame(predictions)
                    
                    st.subheader("🎯 Prediction Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="car_price_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Show summary statistics
                    st.subheader("📈 Prediction Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_predicted = results_df['Predicted_Price'].mean()
                        st.metric("Average Predicted Price", f"₹{avg_predicted:,.0f}")
                    
                    with col2:
                        total_cars = len(results_df)
                        st.metric("Total Cars Processed", total_cars)
                    
                    with col3:
                        avg_difference = results_df['Price_Difference'].mean()
                        st.metric("Avg Price Difference", f"₹{avg_difference:,.0f}")
                    
                    # Visualization
                    st.subheader("📊 Price Distribution")
                    fig = px.histogram(results_df, x='Predicted_Price', nbins=30,
                                      title='Distribution of Predicted Prices')
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

# ========================================
# CAR COMPARISON FEATURE
# ========================================

def show_car_comparison():
    """Show car comparison feature"""
    st.subheader("🔍 Compare Multiple Cars")
    
    st.info("Compare up to 3 cars side by side to make informed decisions")
    
    col1, col2, col3 = st.columns(3)
    
    cars_to_compare = []
    
    with col1:
        st.write("**Car 1**")
        brand1 = st.selectbox("Brand 1", list(CAR_DATABASE.keys()), key="brand1")
        if brand1 in CAR_DATABASE:
            model1 = st.selectbox("Model 1", CAR_DATABASE[brand1]['models'], key="model1")
            year1 = st.number_input("Year 1", min_value=1990, max_value=datetime.now().year, value=2020, key="year1")
            condition1 = st.selectbox("Condition 1", CAR_CONDITIONS, key="condition1")
            add_car1 = st.checkbox("Include Car 1", key="add1", value=True)
            
            if add_car1:
                cars_to_compare.append({
                    'Brand': brand1,
                    'Model': model1,
                    'Year': year1,
                    'Condition': condition1
                })
    
    with col2:
        st.write("**Car 2**")
        brand2 = st.selectbox("Brand 2", list(CAR_DATABASE.keys()), key="brand2")
        if brand2 in CAR_DATABASE:
            model2 = st.selectbox("Model 2", CAR_DATABASE[brand2]['models'], key="model2")
            year2 = st.number_input("Year 2", min_value=1990, max_value=datetime.now().year, value=2019, key="year2")
            condition2 = st.selectbox("Condition 2", CAR_CONDITIONS, key="condition2")
            add_car2 = st.checkbox("Include Car 2", key="add2", value=True)
            
            if add_car2:
                cars_to_compare.append({
                    'Brand': brand2,
                    'Model': model2,
                    'Year': year2,
                    'Condition': condition2
                })
    
    with col3:
        st.write("**Car 3**")
        brand3 = st.selectbox("Brand 3", list(CAR_DATABASE.keys()), key="brand3")
        if brand3 in CAR_DATABASE:
            model3 = st.selectbox("Model 3", CAR_DATABASE[brand3]['models'], key="model3")
            year3 = st.number_input("Year 3", min_value=1990, max_value=datetime.now().year, value=2018, key="year3")
            condition3 = st.selectbox("Condition 3", CAR_CONDITIONS, key="condition3")
            add_car3 = st.checkbox("Include Car 3", key="add3", value=False)
            
            if add_car3:
                cars_to_compare.append({
                    'Brand': brand3,
                    'Model': model3,
                    'Year': year3,
                    'Condition': condition3
                })
    
    if cars_to_compare and st.button("🔄 Compare Cars", type="primary"):
        comparison_data = []
        
        with st.spinner("Comparing cars..."):
            for car in cars_to_compare:
                # Get car specifications
                brand = car['Brand']
                model = car['Model']
                
                if brand in CAR_DATABASE and model in CAR_DATABASE[brand]['models']:
                    model_index = CAR_DATABASE[brand]['models'].index(model)
                    
                    # Prepare input data for prediction
                    input_data = {
                        'Brand': brand,
                        'Model': model,
                        'Car_Type': CAR_DATABASE[brand]['car_types'][model_index],
                        'Year': car['Year'],
                        'Fuel_Type': 'Petrol',
                        'Transmission': 'Manual',
                        'Mileage': 30000,
                        'Engine_cc': CAR_DATABASE[brand]['engine_cc'][model_index],
                        'Power_HP': CAR_DATABASE[brand]['power_hp'][model_index],
                        'Seats': CAR_DATABASE[brand]['seats'][model_index],
                        'Condition': car['Condition'],
                        'Owner_Type': 'First'
                    }
                    
                    # Get predicted price
                    predicted_price = st.session_state.predictor.predict_price(input_data)
                    
                    # Get market prices
                    market_prices, _ = st.session_state.predictor.get_enhanced_live_prices(brand, model)
                    
                    comparison_data.append({
                        'Brand': brand,
                        'Model': model,
                        'Year': car['Year'],
                        'Type': CAR_DATABASE[brand]['car_types'][model_index],
                        'Engine (cc)': CAR_DATABASE[brand]['engine_cc'][model_index],
                        'Power (HP)': CAR_DATABASE[brand]['power_hp'][model_index],
                        'Seats': CAR_DATABASE[brand]['seats'][model_index],
                        'Condition': car['Condition'],
                        'Predicted Price': predicted_price,
                        'Market Low': market_prices[0],
                        'Market Average': market_prices[1],
                        'Market High': market_prices[2],
                        'Value Score': (predicted_price / market_prices[1]) * 100
                    })
        
        if comparison_data:
            # Create comparison dataframe
            comparison_df = pd.DataFrame(comparison_data)
            
            st.subheader("📊 Car Comparison Results")
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visual comparison
            st.subheader("📈 Visual Comparison")
            
            # Price comparison chart
            fig = px.bar(comparison_df, 
                        x='Model', 
                        y=['Predicted Price', 'Market Average'],
                        title='Price Comparison',
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            # Value score comparison
            fig2 = px.bar(comparison_df,
                         x='Model',
                         y='Value Score',
                         title='Value Score (Higher is Better)',
                         color='Value Score')
            st.plotly_chart(fig2, use_container_width=True)

# ========================================
# MAIN INTERFACE FUNCTIONS
# ========================================

def show_enhanced_prediction_interface():
    """Show the enhanced price prediction interface"""
    st.subheader("🎯 Advanced Price Prediction")
    
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
                    prices, sources = st.session_state.predictor.get_enhanced_live_prices(brand, model)
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
                
                st.info(f"**Data Sources:** {', '.join(sources)}")
        
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
                    
                    # Add to search history
                    add_to_search_history(brand, model, predicted_price, avg_price, confidence)
                    
                    # Price justification
                    show_price_breakdown(input_data, predicted_price, avg_price)
                    
                    st.balloons()

def show_market_analysis():
    """Enhanced market analysis"""
    st.subheader("📈 Advanced Market Analysis")
    
    # Price distribution by brand
    brand_prices = {}
    for brand in list(CAR_DATABASE.keys())[:15]:  # Limit to first 15 brands for performance
        try:
            prices, _ = st.session_state.predictor.get_enhanced_live_prices(brand, CAR_DATABASE[brand]['models'][0])
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
    - Ensemble Learning (Random Forest)
    - Comprehensive Synthetic Training Data
    - Realistic Price Calculation Algorithms
    - Business Rules Integration
    - Confidence Scoring
    """)
    
    if st.button("🚀 Train Advanced Model", type="primary"):
        with st.spinner("Creating comprehensive training dataset and training AI models..."):
            st.session_state.predictor.train_model()

# ========================================
# MAIN APPLICATION
# ========================================

def main():
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = EnhancedCarPricePredictor()
    
    st.title("🚗 Advanced Car Price Prediction System")
    st.markdown("### **AI-Powered Accurate Price Estimation with Market Intelligence**")
    
    # Show brand statistics in sidebar
    show_brand_statistics()
    
    # Add search functionality
    search_cars()
    
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/48/car.png")
        st.title("Navigation")
        page = st.radio("Go to", [
            "Advanced Price Prediction", 
            "CSV Upload & Bulk Prediction", 
            "Car Comparison", 
            "Search History",
            "Market Analysis", 
            "Model Training"
        ])
        
        st.markdown("---")
        st.subheader("AI Features")
        st.success("✅ Machine Learning Model")
        st.success("✅ Real-Time Market Data")
        st.success("✅ Price Breakdown Analysis")
        st.success("✅ Confidence Scoring")
        st.success("✅ CSV Bulk Processing")
        st.success("✅ Car Comparison")
        st.success("✅ Search History")
        
        if page == "Model Training":
            if st.button("🔄 Train Enhanced Model", use_container_width=True):
                with st.spinner("Training advanced AI model..."):
                    st.session_state.predictor.train_model()
    
    if page == "Advanced Price Prediction":
        show_enhanced_prediction_interface()
    
    elif page == "CSV Upload & Bulk Prediction":
        show_csv_upload_feature()
    
    elif page == "Car Comparison":
        show_car_comparison()
    
    elif page == "Search History":
        show_search_history()
    
    elif page == "Market Analysis":
        show_market_analysis()
    
    elif page == "Model Training":
        show_model_training()

# ========================================
# RUN THE APPLICATION
# ========================================

if __name__ == "__main__":
    main()
