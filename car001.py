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
import requests
from bs4 import BeautifulSoup
import re
import joblib
import os
import time
import io
from fake_useragent import UserAgent

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
        'seats': [5, 5, 5, 5, 5, 5, 7, 2, 5, 5, 5, 5, 5, 5, 4]
    },
    'Audi': {
        'models': ['A3', 'A4', 'A6', 'A8', 'Q3', 'Q5', 'Q7', 'Q8', 'e-tron', 'RS5', 'R8', 'TT', 'RS7', 'Q8 Sportback', 'A5'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'Coupe', 'Sports', 'Coupe', 'Sedan', 'SUV', 'Coupe'],
        'engine_cc': [1395, 1984, 1984, 2995, 1395, 1984, 2995, 2995, 0, 2894, 5204, 1984, 3993, 2995, 1984],
        'power_hp': [150, 190, 245, 340, 150, 245, 340, 340, 355, 450, 602, 228, 600, 340, 190],
        'seats': [5, 5, 5, 5, 5, 5, 7, 2, 5, 5, 5, 5, 5, 5, 4]
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
        'seats': [5, 5, 5, 5, 5, 7, 5, 7]
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
    # SUPER LUXURY CAR BRANDS
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
        'seats': [2, 2, 2, 2, 2, 2]
    },
    # VINTAGE CAR BRANDS
    'Hindustan Motors': {
        'models': ['Ambassador', 'Contessa', 'Landmaster', 'Trekker'],
        'car_types': ['Sedan', 'Coupe', 'SUV', 'SUV'],
        'engine_cc': [1489, 1489, 1817, 1817],
        'power_hp': [50, 55, 60, 60],
        'seats': [5, 4, 5, 5]
    },
    'Premier': {
        'models': ['Padmini', '118NE', 'Sigma'],
        'car_types': ['Hatchback', 'Sedan', 'Sedan'],
        'engine_cc': [770, 1171, 1360],
        'power_hp': [37, 55, 68],
        'seats': [4, 5, 5]
    },
    'Standard': {
        'models': ['Herald', 'Vanguard', 'Eight', 'Ten'],
        'car_types': ['Convertible', 'Sedan', 'Sedan', 'Sedan'],
        'engine_cc': [948, 2088, 1006, 1206],
        'power_hp': [39, 68, 26, 33],
        'seats': [4, 5, 4, 5]
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
        """Get live prices for car models"""
        try:
            # Simple price database
            price_database = {
                'Maruti Suzuki': {
                    'Alto': [150000, 250000, 350000],
                    'Swift': [300000, 450000, 600000],
                    'Baleno': [350000, 500000, 700000],
                    'Dzire': [320000, 480000, 650000],
                    'Vitara Brezza': [500000, 700000, 900000],
                    'Ertiga': [450000, 650000, 850000],
                    'Wagon R': [200000, 300000, 400000],
                    'Celerio': [250000, 350000, 450000]
                },
                'Hyundai': {
                    'i10': [250000, 350000, 450000],
                    'i20': [350000, 500000, 650000],
                    'Creta': [600000, 850000, 1100000],
                    'Verna': [450000, 650000, 850000],
                    'Venue': [450000, 600000, 800000]
                },
                'Tata': {
                    'Tiago': [250000, 350000, 450000],
                    'Nexon': [450000, 650000, 850000],
                    'Altroz': [350000, 500000, 650000],
                    'Harrier': [800000, 1100000, 1400000],
                    'Safari': [900000, 1200000, 1500000]
                },
                'Mahindra': {
                    'Scorpio': [500000, 700000, 900000],
                    'XUV300': [450000, 600000, 800000],
                    'XUV700': [900000, 1200000, 1500000],
                    'Thar': [600000, 850000, 1100000]
                },
                'Toyota': {
                    'Innova Crysta': [1000000, 1400000, 1800000],
                    'Fortuner': [1500000, 2000000, 2500000],
                    'Glanza': [350000, 500000, 650000]
                },
                'Honda': {
                    'City': [450000, 650000, 850000],
                    'Amaze': [350000, 500000, 650000]
                },
                'BMW': {
                    '3 Series': [1800000, 2500000, 3500000],
                    '5 Series': [3000000, 4000000, 5500000],
                    'X1': [2500000, 3500000, 4500000]
                },
                'Mercedes-Benz': {
                    'A-Class': [2200000, 3000000, 4000000],
                    'C-Class': [2800000, 4000000, 5500000],
                    'GLA': [2500000, 3500000, 4800000]
                },
                'Audi': {
                    'A3': [2000000, 2800000, 3800000],
                    'A4': [3000000, 4200000, 5500000],
                    'Q3': [2800000, 3800000, 5000000]
                }
            }
            
            if brand in price_database and model in price_database[brand]:
                prices = price_database[brand][model]
                sources = ["Market Database"]
            else:
                # Estimate based on car type
                base_prices = {
                    'Hatchback': [200000, 350000, 500000],
                    'Sedan': [300000, 500000, 700000],
                    'SUV': [400000, 650000, 900000],
                    'MUV': [350000, 550000, 750000],
                    'Sports': [5000000, 10000000, 20000000]
                }
                car_type = "Sedan"
                if brand in CAR_DATABASE and model in CAR_DATABASE[brand]['models']:
                    model_index = CAR_DATABASE[brand]['models'].index(model)
                    car_type = CAR_DATABASE[brand]['car_types'][model_index]
                
                prices = base_prices.get(car_type, [300000, 500000, 800000])
                sources = ["Market Estimate"]
                
        except Exception as e:
            prices = [300000, 500000, 800000]
            sources = ["General Market Average"]
        
        return prices, sources

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
        st.session_state.predictor = CarPricePredictor()
    
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
