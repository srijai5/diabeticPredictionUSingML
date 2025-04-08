import joblib
import sys
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)


if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

@app.route('/')
def inm():
    return render_template('inm.html')

@app.route('/submit1', methods=['POST'])
def submit1():
    model_path = "mama.h5" 
    accuracy_path = "accuracy1.txt"
    
    
    try:
        model = joblib.load(model_path) 
    except Exception as e:
        return str(e)

    
    try:
        with open(accuracy_path, 'r') as f:
            accuracy = f.read()
    except Exception as e:
        return str(e)

    
    f = model.feature_names_in_  
    gender = str(request.form["gender"])
    age = int(request.form["age"])
    hypertension = int(request.form["hypertension"])
    heart_disease = int(request.form["heart_disease"])
    smoking_history = str(request.form["smoking_history"])
    bmi = float(request.form["bmi"])
    HbA1c_level = float(request.form["HbA1c_level"])
    blood_glucose_level = int(request.form["blood_glucose_level"])  

    
    k = {'gender': gender,
         "age": age,
         "hypertension": hypertension, 
         "heart_disease": heart_disease,
         "smoking_history": smoking_history,
         "bmi": bmi,
         "HbA1c_level": HbA1c_level,
         "blood_glucose_level": blood_glucose_level}
    
    
    k = pd.DataFrame([k])
    k = k.reindex(columns=f, fill_value=0) 

    
    try:
        result = model.predict(k)  
    except Exception as e:
        return str(e)

    
    if result[0] == 0:
        result = "The patient is not diagnosed with diabetes"
    else:
        result = "The patient is diagnosed with diabetes"
    
    return redirect(url_for('result', result=result, accuracy=accuracy))

@app.route('/result')
def result():
    result = request.args.get('result')
    accuracy = request.args.get('accuracy')
    return render_template('result.html', result=result, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
