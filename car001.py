# ======================================================
# SMART CAR PRICING SYSTEM - ALL CARS COMPREHENSIVE DATABASE
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime

# ========================================
# COMPREHENSIVE CAR DATABASE - ALL BRANDS & MODELS
# ========================================

CAR_DATABASE = {
    'Maruti Suzuki': {
        'models': ['Alto 800', 'Alto K10', 'S-Presso', 'Celerio', 'Wagon R', 'Ignis', 'Swift', 'Baleno', 
                  'Dzire', 'Ciaz', 'Ertiga', 'XL6', 'Vitara Brezza', 'Jimny', 'Fronx', 'Grand Vitara', 
                  'Eeco', 'Omni', 'Celerio X', 'Swift Dzire', 'Estilo', 'A-Star', 'Zen', 'Alto LXi'],
        'car_types': ['Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 
                     'Hatchback', 'Hatchback', 'Sedan', 'Sedan', 'MUV', 'MUV', 'SUV', 'SUV', 'SUV', 
                     'SUV', 'Van', 'Van', 'Hatchback', 'Sedan', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback'],
        'engine_cc': [796, 998, 998, 998, 998, 1197, 1197, 1197, 1197, 1462, 1462, 1462, 1462, 1462, 
                     1197, 1462, 1196, 796, 998, 1197, 998, 998, 993, 796],
        'power_hp': [48, 67, 67, 67, 67, 83, 90, 90, 90, 103, 103, 103, 103, 103, 90, 103, 73, 35, 
                    67, 90, 63, 68, 60, 48],
        'seats': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 6, 5, 5, 5, 5, 5, 8, 5, 5, 5, 5, 5, 5]
    },
    'Hyundai': {
        'models': ['Santro', 'i10', 'Grand i10', 'Grand i10 Nios', 'i20', 'i20 N Line', 'Aura', 'Verna', 
                  'Elantra', 'Venue', 'Creta', 'Alcazar', 'Tucson', 'Kona Electric', 'Ioniq 5', 'Xcent', 
                  'Eon', 'Accent'],
        'car_types': ['Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 'Hatchback', 
                     'Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'Sedan', 
                     'Hatchback', 'Sedan'],
        'engine_cc': [1086, 1086, 1197, 1197, 1197, 998, 1197, 1493, 1999, 1197, 1493, 2199, 2199, 0, 
                     0, 1197, 814, 1495],
        'power_hp': [69, 69, 83, 83, 83, 120, 83, 115, 152, 83, 115, 148, 148, 136, 217, 83, 56, 94],
        'seats': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5]
    },
    'Tata': {
        'models': ['Tiago', 'Tiago EV', 'Tigor', 'Tigor EV', 'Altroz', 'Altroz EV', 'Punch', 'Punch EV',
                  'Nexon', 'Nexon EV', 'Harrier', 'Safari', 'Hexa', 'Indica', 'Indigo', 'Sumo', 'Nano',
                  'Bolt', 'Zest', 'Vista'],
        'car_types': ['Hatchback', 'Hatchback', 'Sedan', 'Sedan', 'Hatchback', 'Hatchback', 'SUV', 'SUV',
                     'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'Hatchback', 'Sedan', 'SUV', 'Hatchback', 
                     'Hatchback', 'Sedan', 'Hatchback'],
        'engine_cc': [1199, 0, 1199, 0, 1199, 0, 1199, 0, 1199, 0, 1956, 1956, 2179, 1405, 1405, 2179,
                     624, 1193, 1193, 1172],
        'power_hp': [85, 75, 85, 75, 85, 75, 120, 120, 120, 129, 170, 170, 156, 70, 70, 120, 38, 90, 
                    90, 75],
        'seats': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 5, 5, 8, 4, 5, 5, 5]
    },
    'Mahindra': {
        'models': ['Bolero', 'Bolero Neo', 'Scorpio', 'Scorpio N', 'Scorpio Classic', 'XUV300', 'XUV400',
                  'XUV500', 'XUV700', 'Thar', 'Thar Roxx', 'Marazzo', 'TUV300', 'KUV100', 'Alturas G4',
                  'Verito', 'Xylo', 'Quanto', 'XUV 3XO'],
        'car_types': ['SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'MUV',
                     'SUV', 'Hatchback', 'SUV', 'Sedan', 'MUV', 'SUV', 'SUV'],
        'engine_cc': [1493, 1493, 2179, 1997, 2179, 1197, 0, 2179, 1997, 1997, 2184, 1497, 1493, 1198,
                     2157, 1461, 2179, 1493, 1197],
        'power_hp': [75, 100, 140, 200, 140, 110, 150, 155, 200, 150, 162, 123, 100, 83, 178, 65, 120,
                    100, 110],
        'seats': [7, 7, 7, 7, 7, 5, 5, 7, 7, 4, 5, 8, 7, 5, 7, 5, 8, 5, 5]
    },
    'Toyota': {
        'models': ['Glanza', 'Urban Cruiser Hyryder', 'Fortuner', 'Fortuner Legender', 'Innova Crysta',
                  'Innova Hycross', 'Camry', 'Vellfire', 'Hilux', 'Land Cruiser', 'Rumion', 'Etios',
                  'Etios Liva', 'Corolla Altis', 'Yaris', 'Prius'],
        'car_types': ['Hatchback', 'SUV', 'SUV', 'SUV', 'MUV', 'MUV', 'Sedan', 'MUV', 'Pickup', 'SUV',
                     'MUV', 'Sedan', 'Hatchback', 'Sedan', 'Sedan', 'Sedan'],
        'engine_cc': [1197, 1462, 2694, 2755, 2393, 1987, 2487, 2494, 2755, 4461, 1462, 1496, 1496,
                     1798, 1496, 1798],
        'power_hp': [90, 103, 204, 204, 150, 186, 177, 197, 204, 304, 103, 90, 90, 140, 107, 138],
        'seats': [5, 5, 7, 7, 7, 7, 5, 7, 5, 7, 7, 5, 5, 5, 5, 5]
    },
    'Honda': {
        'models': ['Amaze', 'City', 'City e:HEV', 'Elevate', 'WR-V', 'Jazz', 'Civic', 'CR-V', 'Accord',
                  'Brio', 'Mobilio'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'Hatchback', 'Sedan', 'SUV', 'Sedan',
                     'Hatchback', 'MUV'],
        'engine_cc': [1199, 1498, 1498, 1498, 1199, 1199, 1799, 1997, 1993, 1198, 1497],
        'power_hp': [90, 121, 126, 121, 90, 90, 141, 158, 177, 88, 118],
        'seats': [5, 5, 5, 5, 5, 5, 5, 7, 5, 5, 7]
    },
    'Kia': {
        'models': ['Sonet', 'Seltos', 'Carens', 'Carnival', 'EV6', 'EV9'],
        'car_types': ['SUV', 'SUV', 'MUV', 'MUV', 'SUV', 'SUV'],
        'engine_cc': [998, 1353, 1482, 2199, 0, 0],
        'power_hp': [120, 140, 115, 200, 229, 384],
        'seats': [5, 5, 6, 7, 5, 6]
    },
    'Volkswagen': {
        'models': ['Polo', 'Vento', 'Virtus', 'Taigun', 'Tiguan', 'T-Roc'],
        'car_types': ['Hatchback', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [999, 999, 999, 999, 1984, 1498],
        'power_hp': [110, 110, 115, 115, 190, 150],
        'seats': [5, 5, 5, 5, 5, 5]
    },
    'Skoda': {
        'models': ['Rapid', 'Slavia', 'Kushaq', 'Kodiaq', 'Superb', 'Octavia'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'SUV', 'Sedan', 'Sedan'],
        'engine_cc': [999, 999, 999, 1984, 1984, 1984],
        'power_hp': [110, 115, 115, 190, 190, 190],
        'seats': [5, 5, 5, 7, 5, 5]
    },
    'Renault': {
        'models': ['Kwid', 'Triber', 'Kiger', 'Duster', 'Captur', 'Lodgy', 'Fluence'],
        'car_types': ['Hatchback', 'MUV', 'SUV', 'SUV', 'SUV', 'MUV', 'Sedan'],
        'engine_cc': [999, 999, 999, 1498, 1498, 1461, 1461],
        'power_hp': [68, 72, 100, 106, 106, 85, 110],
        'seats': [5, 7, 5, 5, 5, 7, 5]
    },
    'Nissan': {
        'models': ['Magnite', 'Kicks', 'X-Trail', 'Sunny', 'Micra', 'Terrano', 'GT-R'],
        'car_types': ['SUV', 'SUV', 'SUV', 'Sedan', 'Hatchback', 'SUV', 'Sports'],
        'engine_cc': [999, 1498, 2488, 1498, 1198, 1461, 3799],
        'power_hp': [100, 106, 169, 99, 77, 110, 565],
        'seats': [5, 5, 7, 5, 5, 5, 4]
    },
    'MG': {
        'models': ['Hector', 'Hector Plus', 'Astor', 'Gloster', 'ZS EV', 'Comet EV', 'Windsor EV'],
        'car_types': ['SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'Hatchback', 'MUV'],
        'engine_cc': [1451, 1451, 1349, 1996, 0, 0, 0],
        'power_hp': [143, 143, 134, 218, 177, 42, 136],
        'seats': [5, 6, 5, 7, 5, 4, 5]
    },
    'Ford': {
        'models': ['EcoSport', 'Endeavour', 'Figo', 'Aspire', 'Freestyle', 'Mustang'],
        'car_types': ['SUV', 'SUV', 'Hatchback', 'Sedan', 'Crossover', 'Sports'],
        'engine_cc': [1498, 1996, 1194, 1194, 1194, 5038],
        'power_hp': [123, 170, 96, 96, 96, 450],
        'seats': [5, 7, 5, 5, 5, 4]
    },
    'Jeep': {
        'models': ['Compass', 'Meridian', 'Wrangler', 'Grand Cherokee'],
        'car_types': ['SUV', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [1956, 1956, 1995, 2995],
        'power_hp': [170, 170, 268, 286],
        'seats': [5, 7, 5, 5]
    },
    'Citroen': {
        'models': ['C3', 'C5 Aircross', 'eC3', 'C3 Aircross'],
        'car_types': ['Hatchback', 'SUV', 'Hatchback', 'SUV'],
        'engine_cc': [1199, 1997, 0, 1199],
        'power_hp': [82, 177, 57, 110],
        'seats': [5, 5, 5, 5]
    },
    'BMW': {
        'models': ['2 Series Gran Coupe', '3 Series', '3 Series Gran Limousine', '5 Series', '7 Series',
                  'X1', 'X3', 'X4', 'X5', 'X6', 'X7', 'Z4', 'i4', 'iX', 'iX1', 'M2', 'M3', 'M4', 'M5',
                  'M8', 'X3 M', 'X5 M', 'X6 M'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV',
                     'SUV', 'Convertible', 'Sedan', 'SUV', 'SUV', 'Coupe', 'Sedan', 'Coupe', 'Sedan',
                     'Coupe', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [1998, 1998, 1998, 1998, 2998, 1499, 1998, 2998, 2998, 2998, 2998, 1998, 0, 0, 0,
                     2993, 2993, 2993, 4395, 4395, 2993, 4395, 4395],
        'power_hp': [190, 258, 258, 248, 335, 140, 248, 265, 335, 335, 400, 197, 340, 523, 313, 460,
                    473, 473, 600, 625, 473, 600, 600],
        'seats': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 2, 5, 5, 5, 4, 5, 4, 5, 4, 5, 5, 5]
    },
    'Mercedes-Benz': {
        'models': ['A-Class Limousine', 'C-Class', 'E-Class', 'S-Class', 'GLA', 'GLB', 'GLC', 'GLE',
                  'GLS', 'G-Class', 'EQB', 'EQE', 'EQS', 'EQS SUV', 'CLA', 'CLS', 'AMG GT', 'SL',
                  'Maybach S-Class', 'Maybach GLS'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV',
                     'SUV', 'Sedan', 'Sedan', 'SUV', 'Sedan', 'Coupe', 'Coupe', 'Convertible', 'Sedan',
                     'SUV'],
        'engine_cc': [1332, 1497, 1991, 2999, 1332, 1332, 1991, 1991, 2999, 2925, 0, 0, 0, 0, 1332,
                     1991, 3982, 2999, 5980, 3982],
        'power_hp': [163, 204, 258, 435, 163, 163, 258, 362, 483, 416, 228, 292, 329, 544, 163, 258,
                    523, 430, 621, 558],
        'seats': [5, 5, 5, 5, 5, 7, 5, 7, 7, 5, 7, 5, 5, 7, 5, 5, 2, 2, 5, 7]
    },
    'Audi': {
        'models': ['A3', 'A4', 'A6', 'A8', 'Q3', 'Q3 Sportback', 'Q5', 'Q7', 'Q8', 'e-tron', 'e-tron GT',
                  'RS Q8', 'RS5', 'RS7', 'R8', 'TT', 'S5', 'A5'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV',
                     'Sedan', 'SUV', 'Coupe', 'Sedan', 'Sports', 'Coupe', 'Coupe', 'Coupe'],
        'engine_cc': [1395, 1984, 1984, 2995, 1395, 1395, 1984, 2995, 2995, 0, 0, 3993, 2894, 3993,
                     5204, 1984, 2894, 1984],
        'power_hp': [150, 190, 245, 340, 150, 150, 245, 340, 340, 355, 530, 600, 450, 600, 602, 228,
                    349, 190],
        'seats': [5, 5, 5, 5, 5, 5, 5, 7, 5, 7, 4, 5, 4, 5, 2, 2, 4, 4]
    },
    'Lexus': {
        'models': ['ES', 'LS', 'NX', 'RX', 'LX', 'UX', 'LC', 'RC'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'Coupe', 'Coupe'],
        'engine_cc': [2487, 3445, 2487, 3456, 3445, 1987, 4969, 3456],
        'power_hp': [215, 422, 194, 295, 409, 169, 471, 311],
        'seats': [5, 5, 5, 5, 8, 5, 4, 4]
    },
    'Jaguar': {
        'models': ['XE', 'XF', 'XJ', 'F-PACE', 'E-PACE', 'I-PACE', 'F-TYPE'],
        'car_types': ['Sedan', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'Sports'],
        'engine_cc': [1997, 1997, 2993, 1997, 1997, 0, 5000],
        'power_hp': [247, 247, 335, 247, 247, 400, 575],
        'seats': [5, 5, 5, 5, 5, 5, 2]
    },
    'Land Rover': {
        'models': ['Range Rover', 'Range Rover Sport', 'Range Rover Velar', 'Range Rover Evoque',
                  'Discovery', 'Discovery Sport', 'Defender'],
        'car_types': ['SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV', 'SUV'],
        'engine_cc': [2996, 2996, 1997, 1997, 2996, 1997, 2996],
        'power_hp': [355, 355, 247, 247, 355, 247, 400],
        'seats': [5, 5, 5, 5, 7, 7, 5]
    },
    'Porsche': {
        'models': ['718 Boxster', '718 Cayman', '911', 'Panamera', 'Cayenne', 'Macan', 'Taycan'],
        'car_types': ['Convertible', 'Coupe', 'Sports', 'Sedan', 'SUV', 'SUV', 'Sedan'],
        'engine_cc': [1988, 1988, 2981, 2894, 2995, 1984, 0],
        'power_hp': [300, 300, 385, 330, 340, 265, 402],
        'seats': [2, 2, 4, 5, 5, 5, 4]
    },
    'Volvo': {
        'models': ['S60', 'S90', 'XC40', 'XC60', 'XC90', 'C40', 'V90'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'SUV', 'SUV', 'SUV', 'Estate'],
        'engine_cc': [1969, 1969, 1969, 1969, 1969, 0, 1969],
        'power_hp': [250, 250, 197, 250, 300, 231, 250],
        'seats': [5, 5, 5, 5, 7, 5, 5]
    },
    'Maserati': {
        'models': ['Ghibli', 'Quattroporte', 'Levante', 'GranTurismo', 'MC20', 'GranCabrio'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'Coupe', 'Sports', 'Convertible'],
        'engine_cc': [2979, 2979, 2979, 4691, 2992, 4691],
        'power_hp': [350, 424, 424, 454, 621, 454],
        'seats': [5, 5, 5, 4, 2, 4]
    },
    'Bentley': {
        'models': ['Continental GT', 'Flying Spur', 'Bentayga', 'Mulsanne'],
        'car_types': ['Coupe', 'Sedan', 'SUV', 'Sedan'],
        'engine_cc': [3993, 3993, 3993, 6750],
        'power_hp': [542, 542, 542, 505],
        'seats': [4, 5, 5, 5]
    },
    'Rolls-Royce': {
        'models': ['Ghost', 'Phantom', 'Cullinan', 'Wraith', 'Dawn', 'Spectre'],
        'car_types': ['Sedan', 'Sedan', 'SUV', 'Coupe', 'Convertible', 'Coupe'],
        'engine_cc': [6749, 6749, 6749, 6592, 6592, 0],
        'power_hp': [563, 563, 563, 624, 563, 577],
        'seats': [5, 5, 5, 4, 4, 4]
    },
    'Lamborghini': {
        'models': ['Huracan', 'Huracan Sterrato', 'Aventador', 'Urus', 'Revuelto'],
        'car_types': ['Sports', 'Sports', 'Sports', 'SUV', 'Sports'],
        'engine_cc': [5204, 5204, 6498, 3996, 6498],
        'power_hp': [631, 610, 740, 641, 1015],
        'seats': [2, 2, 2, 5, 2]
    },
    'Ferrari': {
        'models': ['Portofino', 'Roma', 'F8 Tributo', 'SF90 Stradale', '812 Superfast', '296 GTB',
                  'Purosangue', 'Daytona SP3'],
        'car_types': ['Convertible', 'Coupe', 'Coupe', 'Sports', 'Coupe', 'Sports', 'SUV', 'Sports'],
        'engine_cc': [3855, 3855, 3902, 3990, 6496, 2992, 6496, 6496],
        'power_hp': [612, 612, 710, 986, 789, 654, 715, 840],
        'seats': [2, 4, 2, 2, 2, 2, 4, 2]
    },
    'Aston Martin': {
        'models': ['DB11', 'DB12', 'Vantage', 'DBS', 'DBX', 'Valhalla'],
        'car_types': ['Coupe', 'Coupe', 'Sports', 'Coupe', 'SUV', 'Sports'],
        'engine_cc': [3996, 3982, 3996, 5204, 3982, 3996],
        'power_hp': [503, 671, 503, 715, 542, 937],
        'seats': [4, 4, 2, 2, 5, 2]
    },
    'McLaren': {
        'models': ['GT', 'Artura', '720S', '750S', '765LT'],
        'car_types': ['Sports', 'Sports', 'Sports', 'Sports', 'Sports'],
        'engine_cc': [3994, 2993, 3994, 3994, 3994],
        'power_hp': [612, 671, 710, 740, 765],
        'seats': [2, 2, 2, 2, 2]
    },
    'Bugatti': {
        'models': ['Chiron', 'Chiron Super Sport', 'Chiron Pur Sport', 'Bolide', 'Mistral'],
        'car_types': ['Hypercar', 'Hypercar', 'Hypercar', 'Track Car', 'Roadster'],
        'engine_cc': [7993, 7993, 7993, 7993, 7993],
        'power_hp': [1500, 1578, 1500, 1600, 1578],
        'seats': [2, 2, 2, 2, 2]
    }
}

FUEL_TYPES = ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
TRANSMISSIONS = ["Manual", "Automatic", "CVT", "DCT", "AMT", "iMT"]
CAR_CONDITIONS = ["Excellent", "Very Good", "Good", "Fair", "Poor"]
OWNER_TYPES = ["First", "Second", "Third", "Fourth & Above"]
INSURANCE_STATUS = ["Comprehensive", "Third Party", "Expired", "No Insurance"]
COLORS = ["White", "Black", "Silver", "Grey", "Red", "Blue", "Brown", "Green", "Yellow", "Orange", "Purple", "Other"]
CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad", "Surat", "Jaipur", "Lucknow", "Chandigarh"]

# ========================================
# PRICE DATABASE FOR ALL CARS
# ========================================

def get_base_price_range(brand, model, car_type):
    """Get base price range for any car"""
    
    # Comprehensive price database
    price_database = {
        'Maruti Suzuki': {
            'Alto 800': [120000, 180000, 250000],
            'Alto K10': [280000, 380000, 480000],
            'S-Presso': [280000, 380000, 500000],
            'Celerio': [300000, 420000, 550000],
            'Wagon R': [280000, 400000, 550000],
            'Ignis': [320000, 450000, 580000],
            'Swift': [350000, 500000, 700000],
            'Baleno': [400000, 550000, 750000],
            'Dzire': [350000, 500000, 700000],
            'Ciaz': [450000, 650000, 900000],
            'Ertiga': [500000, 700000, 950000],
            'XL6': [600000, 800000, 1100000],
            'Vitara Brezza': [500000, 700000, 1000000],
            'Jimny': [700000, 950000, 1200000],
            'Fronx': [480000, 650000, 850000],
            'Grand Vitara': [850000, 1150000, 1500000],
            'Eeco': [250000, 350000, 450000],
            'Omni': [100000, 150000, 200000],
        },
        'Hyundai': {
            'Santro': [200000, 300000, 400000],
            'i10': [220000, 320000, 450000],
            'Grand i10': [250000, 350000, 500000],
            'Grand i10 Nios': [320000, 450000, 600000],
            'i20': [400000, 550000, 750000],
            'i20 N Line': [550000, 750000, 1000000],
            'Aura': [350000, 500000, 650000],
            'Verna': [500000, 700000, 950000],
            'Elantra': [800000, 1100000, 1500000],
            'Venue': [450000, 620000, 850000],
            'Creta': [700000, 950000, 1300000],
            'Alcazar': [900000, 1200000, 1600000],
            'Tucson': [1400000, 1900000, 2500000],
            'Kona Electric': [1200000, 1600000, 2100000],
            'Ioniq 5': [2800000, 3500000, 4500000],
        },
        'Tata': {
            'Tiago': [280000, 380000, 520000],
            'Tiago EV': [450000, 600000, 850000],
            'Tigor': [300000, 420000, 580000],
            'Tigor EV': [550000, 750000, 1000000],
            'Altroz': [380000, 520000, 720000],
            'Altroz EV': [650000, 850000, 1150000],
            'Punch': [350000, 480000, 650000],
            'Punch EV': [600000, 800000, 1100000],
            'Nexon': [500000, 700000, 950000],
            'Nexon EV': [800000, 1100000, 1500000],
            'Harrier': [950000, 1300000, 1750000],
            'Safari': [1100000, 1450000, 1950000],
            'Hexa': [600000, 850000, 1200000],
            'Nano': [50000, 100000, 150000],
        },
        'Mahindra': {
            'Bolero': [350000, 500000, 700000],
            'Bolero Neo': [450000, 600000, 800000],
            'Scorpio': [550000, 750000, 1000000],
            'Scorpio N': [800000, 1100000, 1500000],
            'XUV300': [500000, 700000, 950000],
            'XUV400': [900000, 1200000, 1600000],
            'XUV500': [650000, 900000, 1250000],
            'XUV700': [1000000, 1400000, 1900000],
            'Thar': [700000, 950000, 1300000],
            'Thar Roxx': [850000, 1150000, 1550000],
            'Marazzo': [500000, 700000, 950000],
            'TUV300': [350000, 500000, 700000],
            'KUV100': [250000, 350000, 500000],
            'Alturas G4': [1200000, 1650000, 2200000],
            'XUV 3XO': [550000, 750000, 1000000],
        },
        'Toyota': {
            'Glanza': [380000, 520000, 700000],
            'Urban Cruiser Hyryder': [700000, 950000, 1300000],
            'Fortuner': [1800000, 2400000, 3200000],
            'Fortuner Legender': [2200000, 2800000, 3600000],
            'Innova Crysta': [1100000, 1500000, 2000000],
            'Innova Hycross': [1200000, 1650000, 2200000],
            'Camry': [2200000, 2800000, 3600000],
            'Vellfire': [5500000, 7000000, 9000000],
            'Hilux': [1800000, 2400000, 3200000],
            'Land Cruiser': [8000000, 10500000, 14000000],
            'Rumion': [600000, 800000, 1100000],
            'Etios': [250000, 350000, 500000],
        },
        'Honda': {
            'Amaze': [380000, 520000, 700000],
            'City': [500000, 700000, 950000],
            'City e:HEV': [900000, 1200000, 1600000],
            'Elevate': [650000, 850000, 1200000],
            'WR-V': [450000, 620000, 850000],
            'Jazz': [350000, 500000, 700000],
            'Civic': [900000, 1250000, 1700000],
            'CR-V': [1800000, 2400000, 3200000],
            'Accord': [1500000, 2000000, 2700000],
        },
        'Kia': {
            'Sonet': [480000, 650000, 900000],
            'Seltos': [700000, 950000, 1300000],
            'Carens': [750000, 1000000, 1400000],
            'Carnival': [1800000, 2400000, 3200000],
            'EV6': [3500000, 4500000, 6000000],
            'EV9': [6500000, 8500000, 11000000],
        },
        'Volkswagen': {
            'Polo': [350000, 500000, 700000],
            'Vento': [400000, 550000, 750000],
            'Virtus': [600000, 800000, 1100000],
            'Taigun': [700000, 950000, 1300000],
            'Tiguan': [1800000, 2400000, 3200000],
            'T-Roc': [1200000, 1600000, 2200000],
        },
        'BMW': {
            '2 Series Gran Coupe': [2000000, 2700000, 3600000],
            '3 Series': [2500000, 3500000, 4800000],
            '5 Series': [4000000, 5500000, 7500000],
            '7 Series': [8000000, 11000000, 15000000],
            'X1': [2500000, 3500000, 4800000],
            'X3': [4000000, 5500000, 7500000],
            'X5': [6000000, 8500000, 12000000],
            'X7': [9000000, 12500000, 17000000],
            'Z4': [4500000, 6000000, 8500000],
            'i4': [4000000, 5500000, 7500000],
            'iX': [7000000, 9500000, 13000000],
            'M3': [7000000, 9500000, 13000000],
            'M5': [9000000, 12500000, 17000000],
        },
        'Mercedes-Benz': {
            'A-Class Limousine': [2300000, 3200000, 4300000],
            'C-Class': [3500000, 4800000, 6500000],
            'E-Class': [5500000, 7500000, 10000000],
            'S-Class': [10000000, 14000000, 19000000],
            'GLA': [2500000, 3500000, 4800000],
            'GLC': [4500000, 6200000, 8500000],
            'GLE': [6000000, 8500000, 12000000],
            'GLS': [8500000, 12000000, 16500000],
            'G-Class': [12000000, 17000000, 23000000],
            'EQB': [3500000, 4800000, 6500000],
            'EQS': [9000000, 12500000, 17000000],
            'AMG GT': [15000000, 20000000, 28000000],
            'Maybach S-Class': [18000000, 25000000, 35000000],
        },
        'Audi': {
            'A3': [2200000, 3000000, 4100000],
            'A4': [3200000, 4500000, 6200000],
            'A6': [4800000, 6500000, 9000000],
            'A8': [8000000, 11000000, 15000000],
            'Q3': [2800000, 3900000, 5300000],
            'Q5': [4200000, 5800000, 8000000],
            'Q7': [6500000, 9000000, 12500000],
            'Q8': [7500000, 10500000, 14500000],
            'e-tron': [6500000, 9000000, 12500000],
            'e-tron GT': [11000000, 15000000, 20000000],
            'RS Q8': [13000000, 18000000, 25000000],
            'R8': [18000000, 25000000, 35000000],
        },
        'Porsche': {
            '718 Boxster': [6000000, 8500000, 12000000],
            '718 Cayman': [6500000, 9000000, 12500000],
            '911': [10000000, 14000000, 19000000],
            'Panamera': [9000000, 12500000, 17000000],
            'Cayenne': [8000000, 11000000, 15000000],
            'Macan': [5000000, 7000000, 9500000],
            'Taycan': [10000000, 14000000, 19000000],
        },
        'Lamborghini': {
            'Huracan': [20000000, 28000000, 38000000],
            'Huracan Sterrato': [25000000, 35000000, 48000000],
            'Aventador': [35000000, 48000000, 65000000],
            'Urus': [22000000, 30000000, 42000000],
            'Revuelto': [45000000, 62000000, 85000000],
        },
        'Ferrari': {
            'Portofino': [18000000, 25000000, 35000000],
            'Roma': [20000000, 28000000, 38000000],
            'F8 Tributo': [25000000, 35000000, 48000000],
            'SF90 Stradale': [45000000, 62000000, 85000000],
            '812 Superfast': [35000000, 48000000, 65000000],
            '296 GTB': [30000000, 42000000, 58000000],
            'Purosangue': [38000000, 52000000, 72000000],
        },
        'Rolls-Royce': {
            'Ghost': [40000000, 55000000, 75000000],
            'Phantom': [55000000, 75000000, 105000000],
            'Cullinan': [50000000, 70000000, 95000000],
            'Wraith': [35000000, 48000000, 65000000],
            'Dawn': [38000000, 52000000, 72000000],
            'Spectre': [45000000, 62000000, 85000000],
        },
        'Bentley': {
            'Continental GT': [22000000, 30000000, 42000000],
            'Flying Spur': [25000000, 35000000, 48000000],
            'Bentayga': [28000000, 38000000, 52000000],
            'Mulsanne': [35000000, 48000000, 65000000],
        },
        'Bugatti': {
            'Chiron': [150000000, 200000000, 280000000],
            'Chiron Super Sport': [180000000, 250000000, 350000000],
            'Chiron Pur Sport': [170000000, 230000000, 320000000],
            'Bolide': [250000000, 350000000, 500000000],
            'Mistral': [200000000, 280000000, 400000000],
        }
    }
    
    # Check if specific brand-model combination exists
    if brand in price_database and model in price_database[brand]:
        return price_database[brand][model]
    
    # Fallback: estimate based on car type
    type_based_prices = {
        'Hatchback': [250000, 400000, 600000],
        'Sedan': [350000, 550000, 800000],
        'SUV': [500000, 750000, 1100000],
        'MUV': [400000, 650000, 950000],
        'Crossover': [450000, 650000, 900000],
        'Van': [200000, 350000, 500000],
        'Pickup': [800000, 1200000, 1700000],
        'Coupe': [3000000, 5000000, 8000000],
        'Convertible': [3500000, 5500000, 9000000],
        'Sports': [15000000, 25000000, 40000000],
        'Hypercar': [100000000, 180000000, 300000000],
        'Track Car': [80000000, 150000000, 250000000],
        'Roadster': [20000000, 35000000, 55000000],
        'Estate': [2000000, 3500000, 5500000]
    }
    
    # Apply luxury brand multiplier
    luxury_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Jaguar', 'Land Rover', 
                    'Porsche', 'Volvo', 'Maserati', 'Bentley', 'Rolls-Royce', 
                    'Lamborghini', 'Ferrari', 'Aston Martin', 'McLaren', 'Bugatti']
    
    base_prices = type_based_prices.get(car_type, [300000, 500000, 800000])
    
    if brand in luxury_brands:
        multiplier = 3.5 if brand in ['Rolls-Royce', 'Bugatti'] else 2.5 if brand in ['Ferrari', 'Lamborghini'] else 1.8
        base_prices = [int(p * multiplier) for p in base_prices]
    
    return base_prices

# ========================================
# ENHANCED PRICE PREDICTOR
# ========================================

class EnhancedCarPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_importance = {}
        self.is_trained = False
        
    def create_synthetic_training_data(self):
        """Create comprehensive training data for all cars"""
        np.random.seed(42)
        records = []
        
        current_year = datetime.now().year
        
        for brand in CAR_DATABASE:
            for i, model in enumerate(CAR_DATABASE[brand]['models']):
                car_type = CAR_DATABASE[brand]['car_types'][i]
                engine_cc = CAR_DATABASE[brand]['engine_cc'][i]
                power_hp = CAR_DATABASE[brand]['power_hp'][i]
                seats = CAR_DATABASE[brand]['seats'][i]
                
                # Get base prices
                base_prices = get_base_price_range(brand, model, car_type)
                base_price = base_prices[1]  # Use average
                
                # Generate 15 records per model
                for _ in range(15):
                    # Year generation
                    min_year = max(2000, current_year - 20)
                    year = int(np.random.uniform(min_year, current_year + 0.99))
                    age = current_year - year
                    
                    # Mileage generation with safety checks
                    if age <= 0:
                        mileage = int(np.random.uniform(100, 2000))
                    elif age == 1:
                        mileage = int(np.random.uniform(2000, 20000))
                    else:
                        min_mileage = 5000
                        max_mileage = max(min_mileage + 5000, min(300000, 15000 * age))
                        mileage = int(np.random.uniform(min_mileage, max_mileage))
                    
                    # Condition and owner type
                    condition = np.random.choice(CAR_CONDITIONS, p=[0.1, 0.25, 0.4, 0.2, 0.05])
                    owner_type = np.random.choice(OWNER_TYPES, p=[0.5, 0.3, 0.15, 0.05])
                    
                    fuel_type = np.random.choice(FUEL_TYPES)
                    transmission = np.random.choice(TRANSMISSIONS)
                    
                    # Calculate realistic price
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
        """Calculate realistic price with multiple factors"""
        
        # Age depreciation
        age_factor = 0.87 ** age
        
        # Mileage factor
        mileage_factor = max(0.35, 1 - (mileage / 250000))
        
        # Condition multipliers
        condition_mult = {
            "Excellent": 1.12,
            "Very Good": 1.05,
            "Good": 1.0,
            "Fair": 0.88,
            "Poor": 0.70
        }
        
        # Owner multipliers
        owner_mult = {
            "First": 1.08,
            "Second": 1.0,
            "Third": 0.93,
            "Fourth & Above": 0.84
        }
        
        # Fuel type
        fuel_mult = {
            "Petrol": 1.0,
            "Diesel": 1.05,
            "CNG": 0.92,
            "Electric": 1.15,
            "Hybrid": 1.12
        }
        
        # Transmission
        trans_mult = {
            "Manual": 1.0,
            "Automatic": 1.07,
            "CVT": 1.05,
            "DCT": 1.09,
            "AMT": 1.03,
            "iMT": 1.04
        }
        
        # Calculate price
        price = (base_price * age_factor * mileage_factor *
                condition_mult[condition] * owner_mult[owner_type] *
                fuel_mult[fuel_type] * trans_mult[transmission])
        
        # Add variation
        variation = np.random.uniform(0.93, 1.07)
        price *= variation
        
        return max(50000, int(price))
    
    def train_model(self):
        """Train the ML model"""
        st.info("🔄 Training AI model with comprehensive car database...")
        
        # Create training data
        df = self.create_synthetic_training_data()
        
        st.success(f"✅ Created {len(df)} training samples from {len(CAR_DATABASE)} brands")
        
        # Prepare features
        features = ['Brand', 'Model', 'Car_Type', 'Year', 'Fuel_Type', 'Transmission',
                   'Mileage', 'Engine_cc', 'Power_HP', 'Seats', 'Condition', 'Owner_Type']
        
        X = df[features].copy()
        y = df['Price']
        
        # Encode categorical
        categorical_features = ['Brand', 'Model', 'Car_Type', 'Fuel_Type', 'Transmission', 'Condition', 'Owner_Type']
        for feature in categorical_features:
            self.encoders[feature] = LabelEncoder()
            X[feature] = self.encoders[feature].fit_transform(X[feature])
        
        # Scale numerical
        numerical_features = ['Year', 'Mileage', 'Engine_cc', 'Power_HP', 'Seats']
        X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=120,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        
        # Feature importance
        self.feature_importance = dict(zip(features, self.model.feature_importances_))
        
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        st.success(f"✅ Model trained! R² Score: {r2:.3f}, MAE: ₹{mae:,.0f}")
        
        return self.model
    
    def predict_price(self, input_data):
        """Predict car price"""
        if not self.is_trained:
            self.train_model()
        
        features = ['Brand', 'Model', 'Car_Type', 'Year', 'Fuel_Type', 'Transmission',
                   'Mileage', 'Engine_cc', 'Power_HP', 'Seats', 'Condition', 'Owner_Type']
        
        input_df = pd.DataFrame([input_data])
        
        # Encode
        for feature in ['Brand', 'Model', 'Car_Type', 'Fuel_Type', 'Transmission', 'Condition', 'Owner_Type']:
            if feature in self.encoders:
                try:
                    input_df[feature] = self.encoders[feature].transform([input_data[feature]])[0]
                except ValueError:
                    input_df[feature] = 0
        
        # Scale
        numerical_features = ['Year', 'Mileage', 'Engine_cc', 'Power_HP', 'Seats']
        input_df[numerical_features] = self.scaler.transform(input_df[numerical_features])
        
        # Ensure all features
        for feature in features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        input_df = input_df[features]
        
        # Predict
        prediction = self.model.predict(input_df)[0]
        
        # Apply business rules
        final_prediction = self.apply_business_rules(prediction, input_data)
        
        return max(50000, int(final_prediction))
    
    def apply_business_rules(self, predicted_price, input_data):
        """Apply business logic"""
        adjusted = predicted_price
        
        current_year = datetime.now().year
        age = current_year - input_data['Year']
        
        # Age adjustments
        if age > 12:
            adjusted *= 0.92
        elif age < 2:
            adjusted *= 1.04
        
        # Mileage adjustments
        if input_data['Mileage'] > 120000:
            adjusted *= 0.94
        elif input_data['Mileage'] < 15000:
            adjusted *= 1.03
        
        # Luxury brand handling
        luxury_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Jaguar', 'Land Rover',
                        'Porsche', 'Volvo', 'Maserati', 'Bentley', 'Rolls-Royce',
                        'Lamborghini', 'Ferrari', 'Aston Martin', 'McLaren', 'Bugatti']
        
        if input_data['Brand'] in luxury_brands:
            if age < 5:
                adjusted *= 0.96
            else:
                adjusted *= 1.01
        
        return adjusted

# ========================================
# UI FUNCTIONS
# ========================================

def show_manual_input_form():
    """Complete car details entry form"""
    st.subheader("🚗 Enter Car Details - All Models Available")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Brand selection
        brand = st.selectbox("Select Brand", list(CAR_DATABASE.keys()),
                           help="Choose from comprehensive database of all brands")
        
        if brand in CAR_DATABASE:
            model = st.selectbox("Select Model", CAR_DATABASE[brand]['models'],
                               help=f"All {brand} models available")
            
            # Auto-fill specs
            if model in CAR_DATABASE[brand]['models']:
                model_index = CAR_DATABASE[brand]['models'].index(model)
                car_type = CAR_DATABASE[brand]['car_types'][model_index]
                engine_cc = CAR_DATABASE[brand]['engine_cc'][model_index]
                power_hp = CAR_DATABASE[brand]['power_hp'][model_index]
                seats = CAR_DATABASE[brand]['seats'][model_index]
                
                st.text_input("Car Type", value=car_type, disabled=True)
                st.text_input("Engine", value=f"{engine_cc} cc", disabled=True)
                st.text_input("Power", value=f"{power_hp} HP", disabled=True)
                st.text_input("Seats", value=f"{seats}", disabled=True)
        
        current_year = datetime.now().year
        year = st.number_input("Manufacturing Year", min_value=2000, max_value=current_year,
                             value=current_year-3, help="Year of manufacture")
        
        fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)
        transmission = st.selectbox("Transmission", TRANSMISSIONS)
    
    with col2:
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000,
                                value=30000, step=1000, help="Total kilometers driven")
        
        color = st.selectbox("Color", COLORS)
        condition = st.selectbox("Condition", CAR_CONDITIONS)
        owner_type = st.selectbox("Owner Type", OWNER_TYPES)
        insurance_status = st.selectbox("Insurance", INSURANCE_STATUS)
        city = st.selectbox("City", CITIES)
    
    # Return input data
    input_data = {
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
        'City': city
    }
    
    # Show summary
    with st.expander("📊 Car Summary", expanded=True):
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.write(f"**Brand:** {brand}")
            st.write(f"**Model:** {model}")
            st.write(f"**Year:** {year}")
            st.write(f"**Fuel:** {fuel_type}")
            st.write(f"**Transmission:** {transmission}")
        
        with summary_col2:
            st.write(f"**Mileage:** {mileage:,} km")
            st.write(f"**Engine:** {engine_cc} cc")
            st.write(f"**Power:** {power_hp} HP")
            st.write(f"**Condition:** {condition}")
            st.write(f"**Owner:** {owner_type}")
    
    return input_data

def calculate_confidence(input_data):
    """Calculate prediction confidence"""
    confidence = 88
    
    # Popular brands boost
    popular_brands = ['Maruti Suzuki', 'Hyundai', 'Tata', 'Mahindra', 'Honda', 'Toyota']
    if input_data['Brand'] in popular_brands:
        confidence += 4
    
    # New car boost
    current_year = datetime.now().year
    if current_year - input_data['Year'] <= 5:
        confidence += 3
    
    # Low mileage boost
    if input_data['Mileage'] < 50000:
        confidence += 2
    
    # High mileage penalty
    if input_data['Mileage'] > 100000:
        confidence -= 4
    
    return min(96, max(75, confidence))

def show_price_breakdown(input_data, predicted_price, market_prices):
    """Show detailed price breakdown"""
    st.subheader("💰 Price Analysis Breakdown")
    
    current_year = datetime.now().year
    age = current_year - input_data['Year']
    
    market_avg = market_prices[1]
    
    factors = {
        'Market Average': market_avg,
        'Age Impact': -(age * 35000),
        'Mileage Impact': -int(input_data['Mileage'] * 0.8),
        'Condition Premium': get_condition_premium(input_data['Condition']),
        'Owner History': get_owner_impact(input_data['Owner_Type']),
        'Brand Value': get_brand_factor(input_data['Brand'])
    }
    
    breakdown_df = pd.DataFrame({
        'Factor': factors.keys(),
        'Impact (₹)': factors.values()
    })
    
    st.dataframe(breakdown_df, use_container_width=True)
    
    # Show feature importance
    if hasattr(st.session_state.predictor, 'feature_importance'):
        st.subheader("📊 Key Price Factors")
        
        importance_df = pd.DataFrame({
            'Feature': list(st.session_state.predictor.feature_importance.keys()),
            'Importance': list(st.session_state.predictor.feature_importance.values())
        }).sort_values('Importance', ascending=False).head(8)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title='Most Important Factors')
        st.plotly_chart(fig, use_container_width=True)

def get_condition_premium(condition):
    """Get condition premium"""
    premiums = {
        "Excellent": 55000,
        "Very Good": 28000,
        "Good": 0,
        "Fair": -22000,
        "Poor": -55000
    }
    return premiums.get(condition, 0)

def get_owner_impact(owner_type):
    """Get owner impact"""
    impacts = {
        "First": 35000,
        "Second": 0,
        "Third": -18000,
        "Fourth & Above": -35000
    }
    return impacts.get(owner_type, 0)

def get_brand_factor(brand):
    """Get brand factor"""
    factors = {
        'Maruti Suzuki': 18000,
        'Toyota': 28000,
        'Honda': 22000,
        'Hyundai': 18000,
        'Tata': 12000,
        'Mahindra': 15000,
        'BMW': 65000,
        'Mercedes-Benz': 75000,
        'Audi': 55000,
        'Porsche': 120000,
        'Ferrari': 500000,
        'Lamborghini': 450000,
        'Rolls-Royce': 800000,
        'Bugatti': 2000000
    }
    return factors.get(brand, 0)

def add_to_search_history(brand, model, predicted_price, market_avg, confidence):
    """Add to search history"""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    search_entry = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Brand': brand,
        'Model': model,
        'Predicted Price': f"₹{predicted_price:,.0f}",
        'Market Average': f"₹{market_avg:,.0f}",
        'Confidence': f"{confidence}%",
        'Difference': f"₹{predicted_price - market_avg:,.0f}"
    }
    
    st.session_state.search_history.append(search_entry)
    
    # Keep last 50
    if len(st.session_state.search_history) > 50:
        st.session_state.search_history = st.session_state.search_history[-50:]

def show_search_history():
    """Show search history"""
    st.subheader("📋 Search History")
    
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    if not st.session_state.search_history:
        st.info("No search history yet. Start searching to see history here.")
        return
    
    st.write(f"**Total Searches:** {len(st.session_state.search_history)}")
    
    history_df = pd.DataFrame(st.session_state.search_history[::-1])
    st.dataframe(history_df, use_container_width=True)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        most_searched = history_df['Brand'].mode()[0] if not history_df.empty else "N/A"
        st.metric("Most Searched Brand", most_searched)
    
    with col2:
        total_unique = history_df[['Brand', 'Model']].drop_duplicates().shape[0]
        st.metric("Unique Cars", total_unique)
    
    with col3:
        if st.button("Clear History"):
            st.session_state.search_history = []
            st.rerun()

def show_csv_upload():
    """CSV upload for bulk prediction"""
    st.subheader("📁 CSV Bulk Prediction")
    
    st.info("""
    **Upload CSV with columns:** Brand, Model, Year, Fuel_Type, Transmission, Mileage, Condition, Owner_Type
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Uploaded {len(df)} cars")
            
            st.subheader("📊 Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🎯 Predict All Prices", type="primary"):
                predictions = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in df.iterrows():
                    try:
                        # Get car specs
                        brand = row.get('Brand', '')
                        model = row.get('Model', '')
                        
                        if brand in CAR_DATABASE and model in CAR_DATABASE[brand]['models']:
                            model_index = CAR_DATABASE[brand]['models'].index(model)
                            
                            input_data = {
                                'Brand': brand,
                                'Model': model,
                                'Car_Type': CAR_DATABASE[brand]['car_types'][model_index],
                                'Year': int(row.get('Year', 2020)),
                                'Fuel_Type': row.get('Fuel_Type', 'Petrol'),
                                'Transmission': row.get('Transmission', 'Manual'),
                                'Mileage': int(row.get('Mileage', 30000)),
                                'Engine_cc': CAR_DATABASE[brand]['engine_cc'][model_index],
                                'Power_HP': CAR_DATABASE[brand]['power_hp'][model_index],
                                'Seats': CAR_DATABASE[brand]['seats'][model_index],
                                'Condition': row.get('Condition', 'Good'),
                                'Owner_Type': row.get('Owner_Type', 'First')
                            }
                            
                            predicted_price = st.session_state.predictor.predict_price(input_data)
                            market_prices = get_base_price_range(brand, model, input_data['Car_Type'])
                            
                            predictions.append({
                                'Brand': brand,
                                'Model': model,
                                'Year': input_data['Year'],
                                'Predicted_Price': predicted_price,
                                'Market_Low': market_prices[0],
                                'Market_Average': market_prices[1],
                                'Market_High': market_prices[2],
                                'Difference': predicted_price - market_prices[1]
                            })
                        
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {idx + 1}/{len(df)}")
                        
                    except Exception as e:
                        st.warning(f"Error row {idx}: {str(e)}")
                        continue
                
                progress_bar.empty()
                status_text.empty()
                
                if predictions:
                    results_df = pd.DataFrame(predictions)
                    
                    st.subheader("🎯 Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Summary
                    st.subheader("📊 Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_pred = results_df['Predicted_Price'].mean()
                        st.metric("Avg Predicted", f"₹{avg_pred:,.0f}")
                    
                    with col2:
                        total = len(results_df)
                        st.metric("Total Cars", total)
                    
                    with col3:
                        avg_diff = results_df['Difference'].mean()
                        st.metric("Avg Difference", f"₹{avg_diff:,.0f}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

def show_car_comparison():
    """Compare multiple cars"""
    st.subheader("🔍 Compare Cars")
    
    st.info("Compare up to 3 cars side by side")
    
    col1, col2, col3 = st.columns(3)
    
    cars_to_compare = []
    
    with col1:
        st.write("**Car 1**")
        brand1 = st.selectbox("Brand", list(CAR_DATABASE.keys()), key="b1")
        model1 = st.selectbox("Model", CAR_DATABASE[brand1]['models'], key="m1")
        year1 = st.number_input("Year", 2000, datetime.now().year, 2020, key="y1")
        include1 = st.checkbox("Include", True, key="i1")
        
        if include1:
            cars_to_compare.append({'Brand': brand1, 'Model': model1, 'Year': year1})
    
    with col2:
        st.write("**Car 2**")
        brand2 = st.selectbox("Brand", list(CAR_DATABASE.keys()), key="b2")
        model2 = st.selectbox("Model", CAR_DATABASE[brand2]['models'], key="m2")
        year2 = st.number_input("Year", 2000, datetime.now().year, 2019, key="y2")
        include2 = st.checkbox("Include", True, key="i2")
        
        if include2:
            cars_to_compare.append({'Brand': brand2, 'Model': model2, 'Year': year2})
    
    with col3:
        st.write("**Car 3**")
        brand3 = st.selectbox("Brand", list(CAR_DATABASE.keys()), key="b3")
        model3 = st.selectbox("Model", CAR_DATABASE[brand3]['models'], key="m3")
        year3 = st.number_input("Year", 2000, datetime.now().year, 2018, key="y3")
        include3 = st.checkbox("Include", False, key="i3")
        
        if include3:
            cars_to_compare.append({'Brand': brand3, 'Model': model3, 'Year': year3})
    
    if cars_to_compare and st.button("🔄 Compare", type="primary"):
        comparison_data = []
        
        for car in cars_to_compare:
            brand = car['Brand']
            model = car['Model']
            
            if model in CAR_DATABASE[brand]['models']:
                model_index = CAR_DATABASE[brand]['models'].index(model)
                
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
                    'Condition': 'Good',
                    'Owner_Type': 'First'
                }
                
                predicted = st.session_state.predictor.predict_price(input_data)
                market_prices = get_base_price_range(brand, model, input_data['Car_Type'])
                
                comparison_data.append({
                    'Brand': brand,
                    'Model': model,
                    'Year': car['Year'],
                    'Type': input_data['Car_Type'],
                    'Engine': input_data['Engine_cc'],
                    'Power': input_data['Power_HP'],
                    'Seats': input_data['Seats'],
                    'Predicted': predicted,
                    'Market_Avg': market_prices[1],
                    'Value_Score': round((predicted / market_prices[1]) * 100, 1)
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            st.subheader("📊 Comparison Results")
            st.dataframe(comp_df, use_container_width=True)
            
            # Visual comparison
            fig = px.bar(comp_df, x='Model', y=['Predicted', 'Market_Avg'],
                        title='Price Comparison', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = px.bar(comp_df, x='Model', y='Value_Score',
                         title='Value Score', color='Value_Score')
            st.plotly_chart(fig2, use_container_width=True)

def show_market_analysis():
    """Market analysis dashboard"""
    st.subheader("📈 Market Analysis")
    
    # Price by brand
    brand_prices = {}
    sample_brands = list(CAR_DATABASE.keys())[:15]
    
    for brand in sample_brands:
        try:
            model = CAR_DATABASE[brand]['models'][0]
            car_type = CAR_DATABASE[brand]['car_types'][0]
            prices = get_base_price_range(brand, model, car_type)
            brand_prices[brand] = prices[1]
        except:
            continue
    
    if brand_prices:
        fig = px.bar(x=list(brand_prices.keys()), y=list(brand_prices.values()),
                    title="Average Price by Brand", labels={'x': 'Brand', 'y': 'Price (₹)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Car type distribution
    type_counts = {}
    for brand in CAR_DATABASE:
        for car_type in CAR_DATABASE[brand]['car_types']:
            type_counts[car_type] = type_counts.get(car_type, 0) + 1
    
    fig2 = px.pie(values=list(type_counts.values()), names=list(type_counts.keys()),
                  title="Car Type Distribution")
    st.plotly_chart(fig2, use_container_width=True)

def show_brand_statistics():
    """Show brand statistics"""
    st.sidebar.subheader("📈 Database Stats")
    
    total_brands = len(CAR_DATABASE)
    total_models = sum(len(CAR_DATABASE[brand]['models']) for brand in CAR_DATABASE)
    
    st.sidebar.success(f"""
    **Complete Database:**
    - 🚗 Brands: {total_brands}
    - 🎯 Models: {total_models}
    - ✅ All Cars Available
    """)

def show_prediction_interface():
    """Main prediction interface"""
    st.subheader("🎯 Car Price Prediction - All Models")
    
    # Input form
    input_data = show_manual_input_form()
    
    if input_data:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Market prices
            brand = input_data['Brand']
            model = input_data['Model']
            car_type = input_data['Car_Type']
            
            market_prices = get_base_price_range(brand, model, car_type)
            
            st.subheader("📊 Market Intelligence")
            
            m_col1, m_col2, m_col3 = st.columns(3)
            
            with m_col1:
                st.metric("Market Low", f"₹{market_prices[0]:,.0f}")
            
            with m_col2:
                st.metric("Market Average", f"₹{market_prices[1]:,.0f}")
            
            with m_col3:
                st.metric("Market High", f"₹{market_prices[2]:,.0f}")
        
        with col2:
            st.subheader("🤖 AI Prediction")
            
            if st.button("🎯 Get Price", type="primary", use_container_width=True):
                with st.spinner("Calculating..."):
                    predicted_price = st.session_state.predictor.predict_price(input_data)
                    confidence = calculate_confidence(input_data)
                    
                    st.success(f"**₹{predicted_price:,.0f}**")
                    st.metric("Confidence", f"{confidence}%")
                    
                    # Add to history
                    add_to_search_history(brand, model, predicted_price, market_prices[1], confidence)
                    
                    # Show breakdown
                    show_price_breakdown(input_data, predicted_price, market_prices)
                    
                    st.balloons()

# ========================================
# MAIN APPLICATION
# ========================================

def main():
    # Initialize
    if 'predictor' not in st.session_state:
        st.session_state.predictor = EnhancedCarPricePredictor()
    
    st.title("🚗 Smart Car Pricing System")
    st.markdown("### **All Cars Available - Complete Indian Market Coverage**")
    
    # Sidebar
    show_brand_statistics()
    
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/car.png")
        st.title("Navigation")
        page = st.radio("Go to", [
            "Price Prediction",
            "CSV Bulk Upload",
            "Car Comparison",
            "Search History",
            "Market Analysis",
            "Model Training"
        ])
        
        st.markdown("---")
        st.subheader("✨ Features")
        st.success("✅ All Brands & Models")
        st.success("✅ AI Price Prediction")
        st.success("✅ Market Intelligence")
        st.success("✅ Bulk CSV Upload")
        st.success("✅ Car Comparison")
        st.success("✅ Search History")
    
    # Route to pages
    if page == "Price Prediction":
        show_prediction_interface()
    
    elif page == "CSV Bulk Upload":
        show_csv_upload()
    
    elif page == "Car Comparison":
        show_car_comparison()
    
    elif page == "Search History":
        show_search_history()
    
    elif page == "Market Analysis":
        show_market_analysis()
    
    elif page == "Model Training":
        st.subheader("🤖 AI Model Training")
        
        st.info("""
        **Advanced ML Features:**
        - Random Forest Algorithm
        - Comprehensive Training Data
        - All Brands & Models Coverage
        - Realistic Price Calculations
        """)
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training AI model..."):
                st.session_state.predictor.train_model()

if __name__ == "__main__":
    main()
