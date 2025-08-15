from flask import Flask, render_template, request, redirect
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load model and preprocessor from DVC-tracked folder
model = pickle.load(open('models/rf_classifier.pkl', 'rb'))
preprocessor = pickle.load(open('models/preprocessor.pkl', 'rb'))

# Load dataset (optional, for dropdowns in UI)
Telco = pd.read_csv('Churn_Prediction_Final.csv')


def predict(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
            InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
            StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
            MonthlyCharges, TotalCharges):
    
    features = np.array([[gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
                          InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                          StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                          MonthlyCharges, TotalCharges]], dtype='object')
    
    transformed_features = preprocessor.transform(features)
    result = model.predict(transformed_features).reshape(1, -1)
    
    return result[0]


@app.route('/', methods=['GET', 'POST'])
def index():
    # Prepare dropdown lists
    gender = sorted(Telco['gender'].unique())
    SeniorCitizen = sorted(Telco['SeniorCitizen'].unique())
    Partner = sorted(Telco['Partner'].unique())
    Dependents = sorted(Telco['Dependents'].unique())
    PhoneService = sorted(Telco['PhoneService'].unique())
    MultipleLines = sorted(Telco['MultipleLines'].unique())
    InternetService = sorted(Telco['InternetService'].unique())
    OnlineSecurity = sorted(Telco['OnlineSecurity'].unique())
    OnlineBackup = sorted(Telco['OnlineBackup'].unique())
    DeviceProtection = sorted(Telco['DeviceProtection'].unique())
    TechSupport = sorted(Telco['TechSupport'].unique())
    StreamingTV = sorted(Telco['StreamingTV'].unique())
    StreamingMovies = sorted(Telco['StreamingMovies'].unique())
    Contract = sorted(Telco['Contract'].unique())
    PaperlessBilling = sorted(Telco['PaperlessBilling'].unique())
    PaymentMethod = sorted(Telco['PaymentMethod'].unique())

    # Insert default prompts
    gender.insert(0, 'Confirm your Gender!')
    SeniorCitizen.insert(0, 'Are you SeniorCitizen!')
    Partner.insert(0, 'Do you have Partner?')
    Dependents.insert(0, 'Do you have Dependents?')
    PhoneService.insert(0, 'Do you have PhoneService?')
    MultipleLines.insert(0, 'Do you have MultipleLines?')
    InternetService.insert(0, 'Do you have InternetService?')
    OnlineSecurity.insert(0, 'Do you have OnlineSecurity?')
    OnlineBackup.insert(0, 'Do you have OnlineBackup?')
    DeviceProtection.insert(0, 'Do you have DeviceProtection?')
    TechSupport.insert(0, 'Do you have TechSupport?')
    StreamingTV.insert(0, 'Do you have StreamingTV?')
    StreamingMovies.insert(0, 'Do you have StreamingMovies?')
    Contract.insert(0, 'Select your Contract Mode!')
    PaperlessBilling.insert(0, 'Do you have PaperlessBilling?')
    PaymentMethod.insert(0, 'Select your PaymentMethod?')

    return render_template('index.html', gender=gender, SeniorCitizen=SeniorCitizen, Partner=Partner,
                           Dependents=Dependents, PhoneService=PhoneService, MultipleLines=MultipleLines,
                           InternetService=InternetService, OnlineSecurity=OnlineSecurity, OnlineBackup=OnlineBackup,
                           DeviceProtection=DeviceProtection, TechSupport=TechSupport, StreamingTV=StreamingTV,
                           StreamingMovies=StreamingMovies, Contract=Contract, PaperlessBilling=PaperlessBilling,
                           PaymentMethod=PaymentMethod)


@app.route('/predict_route', methods=['GET', 'POST'])
def predict_route():
    if request.method == 'POST':
        gender = request.form['gender']
        SeniorCitizen = request.form['SeniorCitizen']
        Partner = request.form['Partner']
        Dependents = request.form['Dependents']
        tenure = int(request.form['tenure'])
        PhoneService = request.form['PhoneService']
        MultipleLines = request.form['MultipleLines']
        InternetService = request.form['InternetService']
        OnlineSecurity = request.form['OnlineSecurity']
        OnlineBackup = request.form['OnlineBackup']
        DeviceProtection = request.form['DeviceProtection']
        TechSupport = request.form['TechSupport']
        StreamingTV = request.form['StreamingTV']
        StreamingMovies = request.form['StreamingMovies']
        Contract = request.form['Contract']
        PaperlessBilling = request.form['PaperlessBilling']
        PaymentMethod = request.form['PaymentMethod']
        MonthlyCharges = float(request.form['MonthlyCharges'])
        TotalCharges = float(request.form['TotalCharges'])

        prediction = predict(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
                             InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                             StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                             MonthlyCharges, TotalCharges)
        prediction_text = "The Customer will Churn" if prediction == 1 else "The Customer will not Churn"

        return render_template('index.html', prediction=prediction_text)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
