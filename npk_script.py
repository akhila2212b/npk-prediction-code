import streamlit as st
import ee
import geemap
import joblib
import numpy as np

# Load the trained model
ensemble_model = joblib.load('ensemble_model.pkl')

# Extract individual models
rf_base = ensemble_model['rf_base']
xgb_base = ensemble_model['xgb_base']
lgbm_base = ensemble_model['lgbm_base']
rf_meta_model = ensemble_model['rf_meta_model']

# Authenticate & initialize GEE
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Function to compute vegetation indices
def compute_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    evi = image.expression(
        '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
        {'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2')}
    ).rename('EVI')
    savi = image.expression(
        '(1 + L) * (NIR - RED) / (NIR + RED + L)',
        {'NIR': image.select('B8'), 'RED': image.select('B4'), 'L': 0.5}
    ).rename('SAVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')
    return image.addBands([ndvi, evi, savi, ndwi, gndvi])

# Function to get NPK predictions at a given point
def get_npk_predictions(lat, lon, start_date, end_date):
    point = ee.Geometry.Point(lon, lat)
    image_collection = (
        ee.ImageCollection('COPERNICUS/S2')
        .filterBounds(point)
        .filterDate(start_date, end_date)
    )
    if image_collection.size().getInfo() == 0:
        return "No imagery found for the specified date range and location."
    image = image_collection.first()
    image = compute_indices(image)
    sample = image.sampleRegions(
        collection=ee.FeatureCollection(point),
        scale=10
    ).first().getInfo()
    indices = {
        'NDVI': sample['properties']['NDVI'],
        'EVI': sample['properties']['EVI'],
        'SAVI': sample['properties']['SAVI'],
        'NDWI': sample['properties']['NDWI'],
        'GNDVI': sample['properties']['GNDVI']
    }
    input_data = [[indices['NDVI'], indices['EVI'], indices['SAVI'], indices['NDWI'], indices['GNDVI']]]
    rf_pred = rf_base.predict(input_data)
    xgb_pred = xgb_base.predict(input_data)
    lgbm_pred = lgbm_base.predict(input_data)
    stack_input = np.column_stack((rf_pred, xgb_pred, lgbm_pred))
    npk_prediction = rf_meta_model.predict(stack_input)
    return {
        'N (Nitrogen)': npk_prediction[0][0],
        'P (Phosphorus)': npk_prediction[0][1],
        'K (Potassium)': npk_prediction[0][2]
    }

# Streamlit UI
def streamlit_input():
    st.title("NPK Prediction using Vegetation Indices")
    latitude = st.number_input("Enter Latitude:", -90.0, 90.0, value=0.0)
    longitude = st.number_input("Enter Longitude:", -180.0, 180.0, value=0.0)
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    if end_date < start_date:
        st.error("End date cannot be earlier than start date.")
        return
    if st.button("Get NPK Predictions"):
        npk_values = get_npk_predictions(latitude, longitude, str(start_date), str(end_date))
        if isinstance(npk_values, dict):
            st.write(f"N (Nitrogen): {npk_values['N (Nitrogen)']}")
            st.write(f"P (Phosphorus): {npk_values['P (Phosphorus)']}")
            st.write(f"K (Potassium): {npk_values['K (Potassium)']}")
        else:
            st.write(npk_values)

if __name__ == "__main__":
    streamlit_input()
