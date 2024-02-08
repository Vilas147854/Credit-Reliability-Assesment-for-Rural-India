from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)

class CreditWorthinessModel:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model

    def predict_loan_amount(self, features):
        prediction = self.model.predict([features])
        return prediction

model = CreditWorthinessModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
       
    features = []
    try:
        for key in ['loan_tenure', 'young_dependents', 'house_area', 'old_dependents',
           'occupants_count', 'monthly_expenses', 'annual_income',
           'loan_installments', 'age', 'home_ownership', 'Apparelsloan_purpose',
           'Agro Based Businessesloan_purpose', 'Animal husbandryloan_purpose',
           'Meat Businessesloan_purpose', 'Handicraftsloan_purpose',
           'Farming/ Agricultureloan_purpose', 'Education Loanloan_purpose',
           'Retail Storeloan_purpose', 'Eateriesloan_purpose',
           'Business Services - IIloan_purpose', 'type_of_house_T1',
           'type_of_house_T2', 'sex_M', 'sex_TG']:
            if request.form.get(key):  
                features.append(float(request.form[key]))
            else:
                features.append(0)
        features.append(12)


        
        prediction = model.predict_loan_amount(features)
        return render_template('results.html', prediction=prediction)
    except ValueError as e:
        # Handle ValueError (e.g., form inputs are not numeric)
        return render_template('index.html', error_message="Invalid input. Please enter numeric values for all fields.")

if __name__ == '__main__':
    app.run(debug=False)
