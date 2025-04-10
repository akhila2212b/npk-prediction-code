{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "153814e9-3c1a-437e-9611-fe25daff31eb",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-04-01 16:19:33.014 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Akhi\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import ee\n",
    "import geemap\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# Load the trained model (this assumes you've saved it as a .pkl file)\n",
    "ensemble_model = joblib.load('ensemble_model.pkl')\n",
    "\n",
    "# Extract individual models from the loaded dictionary\n",
    "rf_base = ensemble_model['rf_base']\n",
    "xgb_base = ensemble_model['xgb_base']\n",
    "lgbm_base = ensemble_model['lgbm_base']\n",
    "rf_meta_model = ensemble_model['rf_meta_model']\n",
    "\n",
    "# Authenticate & initialize GEE\n",
    "try:\n",
    "    ee.Initialize()\n",
    "except Exception as e:\n",
    "    ee.Authenticate()\n",
    "    ee.Initialize()\n",
    "\n",
    "# Function to compute vegetation indices\n",
    "def compute_indices(image):\n",
    "    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')\n",
    "    evi = image.expression(\n",
    "        '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',\n",
    "        {\n",
    "            'NIR': image.select('B8'),\n",
    "            'RED': image.select('B4'),\n",
    "            'BLUE': image.select('B2'),\n",
    "        }\n",
    "    ).rename('EVI')\n",
    "    savi = image.expression(\n",
    "        '(1 + L) * (NIR - RED) / (NIR + RED + L)',\n",
    "        {'NIR': image.select('B8'), 'RED': image.select('B4'), 'L': 0.5}\n",
    "    ).rename('SAVI')\n",
    "    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')\n",
    "    gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')\n",
    "\n",
    "    return image.addBands([ndvi, evi, savi, ndwi, gndvi])\n",
    "\n",
    "# Function to get NPK predictions at a given point\n",
    "def get_npk_predictions(lat, lon, start_date, end_date):\n",
    "    point = ee.Geometry.Point(lon, lat)\n",
    "    \n",
    "    # Filter Sentinel-2 imagery within the date range\n",
    "    image_collection = (\n",
    "        ee.ImageCollection('COPERNICUS/S2')\n",
    "        .filterBounds(point)\n",
    "        .filterDate(start_date, end_date)\n",
    "    )\n",
    "\n",
    "    # Check if the image collection has any imagery\n",
    "    if image_collection.size().getInfo() == 0:\n",
    "        return \"No imagery found for the specified date range and location.\"\n",
    "    \n",
    "    # Get the first image from the collection\n",
    "    image = image_collection.first()\n",
    "    \n",
    "    # Compute vegetation indices\n",
    "    image = compute_indices(image)\n",
    "\n",
    "    # Sample the point and extract index values\n",
    "    sample = image.sampleRegions(\n",
    "        collection=ee.FeatureCollection(point),\n",
    "        scale=10\n",
    "    ).first().getInfo()\n",
    "\n",
    "    # Extract the indices (NDVI, EVI, SAVI, NDWI, GNDVI)\n",
    "    indices = {\n",
    "        'NDVI': sample['properties']['NDVI'],\n",
    "        'EVI': sample['properties']['EVI'],\n",
    "        'SAVI': sample['properties']['SAVI'],\n",
    "        'NDWI': sample['properties']['NDWI'],\n",
    "        'GNDVI': sample['properties']['GNDVI']\n",
    "    }\n",
    "\n",
    "    # Prepare the input data for prediction\n",
    "    input_data = [[indices['NDVI'], indices['EVI'], indices['SAVI'], indices['NDWI'], indices['GNDVI']]]\n",
    "\n",
    "    # Predict NPK values using the base models\n",
    "    rf_pred = rf_base.predict(input_data)\n",
    "    xgb_pred = xgb_base.predict(input_data)\n",
    "    lgbm_pred = lgbm_base.predict(input_data)\n",
    "\n",
    "    # Stack the predictions from the base models\n",
    "    stack_input = np.column_stack((rf_pred, xgb_pred, lgbm_pred))\n",
    "\n",
    "    # Use the Random Forest meta-model to make the final prediction\n",
    "    npk_prediction = rf_meta_model.predict(stack_input)\n",
    "\n",
    "    # Return the predicted NPK values (e.g., N, P, K)\n",
    "    return {\n",
    "        'N (Nitrogen)': npk_prediction[0][0],\n",
    "        'P (Phosphorus)': npk_prediction[0][1],\n",
    "        'K (Potassium)': npk_prediction[0][2]\n",
    "    }\n",
    "\n",
    "# Streamlit UI for manual input\n",
    "def streamlit_input():\n",
    "    st.title(\"NPK Prediction using Vegetation Indices\")\n",
    "\n",
    "    # Accept input for latitude, longitude, and date range\n",
    "    latitude = st.number_input(\"Enter Latitude:\", -90.0, 90.0, value=0.0)\n",
    "    longitude = st.number_input(\"Enter Longitude:\", -180.0, 180.0, value=0.0)\n",
    "    start_date = st.date_input(\"Start Date\")\n",
    "    end_date = st.date_input(\"End Date\")\n",
    "\n",
    "    # Ensure end date is after start date\n",
    "    if end_date < start_date:\n",
    "        st.error(\"End date cannot be earlier than start date.\")\n",
    "        return\n",
    "\n",
    "    # Get NPK predictions when the button is pressed\n",
    "    if st.button(\"Get NPK Predictions\"):\n",
    "        npk_values = get_npk_predictions(latitude, longitude, str(start_date), str(end_date))\n",
    "        \n",
    "        # Display the predicted NPK values\n",
    "        if isinstance(npk_values, dict):\n",
    "            st.write(f\"N (Nitrogen): {npk_values['N (Nitrogen)']}\")\n",
    "            st.write(f\"P (Phosphorus): {npk_values['P (Phosphorus)']}\")\n",
    "            st.write(f\"K (Potassium): {npk_values['K (Potassium)']}\")\n",
    "        else:\n",
    "            st.write(npk_values)\n",
    "\n",
    "# Run the Streamlit app\n",
    "if __name__ == \"__main__\":\n",
    "    streamlit_input()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "5476d25f-0ad6-4886-be09-a5c8074d0c1c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
