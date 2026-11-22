
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
import traceback
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Load the trained model
try:
    model = joblib.load("superkart_model.joblib")
except Exception as e:
    model = f"Error loading model: {str(e)}"

# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System"

# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    try:
        if isinstance(model, str):
            return jsonify({'error': model}), 500
            
        # Get JSON data from the request
        data = request.get_json()

        # Extract relevant features from the input data
        sample = {
            'Product_Weight': data['Product_Weight'],
            'Product_Sugar_Content': data['Product_Sugar_Content'],
            'Product_Allocated_Area': data['Product_Allocated_Area'],
            'Product_MRP': data['Product_MRP'],
            'Store_Size': data['Store_Size'],
            'Store_Location_City_Type': data['Store_Location_City_Type'],
            'Store_Type': data['Store_Type'],
            'Product_Id_char': data['Product_Id_char'],
            'Store_Age_Years': data['Store_Age_Years'],
            'Product_Type_Category': data['Product_Type_Category']
        }

        # Convert the extracted data into a DataFrame
        input_data = pd.DataFrame([sample])

        # Make a prediction using the trained model
        prediction = model.predict(input_data).tolist()[0]

        # Return the prediction as a JSON response
        return jsonify({'Sales': prediction})
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Define an endpoint to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    try:
        if isinstance(model, str):
            return jsonify({'error': model}), 500
            
        # Get the uploaded CSV file from the request
        file = request.files['file']

        # Read the file into a DataFrame
        input_data = pd.read_csv(file)

        # Make predictions for the batch data
        predictions = model.predict(input_data).tolist()

        # Create an output dictionary mapping row index to predicted sales
        output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

        return jsonify(output_dict)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
