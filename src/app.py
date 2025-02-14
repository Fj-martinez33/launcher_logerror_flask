from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from datetime import date
from pickle import load

#Cargamos el modelo

MODEL = load(open("models/ensemble.sav", "rb"))

APP = Flask(__name__)

CALENDAR = date(2016,1,1)

def VoidInt(value):
    try:
        return int(value) if value else 0
    except ValueError:
        return 0

def VoidFloat(value):
    try:
        return float(value) if value else 0
    except ValueError:
        return 0.0

@APP.route("/", methods=["GET", "POST"])

#BACKEND
def Home():
    log_error = None

    if request.method == 'POST':
        transaction_date = request.form['transaction_date']
        transaction_month = int(transaction_date.split("-")[1])
        transaction_day = int(transaction_date.split("-")[2])

        construction_date = request.form['construction_date']
        construction_year = int(construction_date.split("-")[0])

        calculated_sqmf = VoidFloat(request.form['calculated_sqmf'])
        finished_living_area = VoidFloat(request.form['finished_living_area'])
        total_sqm = VoidFloat(request.form['total_sqm'])
        lotsizesqft = VoidFloat(request.form['lotsizesqft'])

        # Calcular lot_sqft_ratio
        if lotsizesqft and calculated_sqmf != 0:
            lot_sqft_ratio = calculated_sqmf / lotsizesqft
        else:
            lot_sqft_ratio = 0

        structureTaxValue = VoidFloat(request.form['structureTaxValue'])
        taxamount = VoidFloat(request.form['taxamount'])

        # Calcular price_sqft
        if structureTaxValue and calculated_sqmf != 0:
            price_sqft = structureTaxValue / calculated_sqmf
        else:
            price_sqft = 0

        baths = VoidInt(request.form['baths'])
        beds = VoidInt(request.form['beds'])

        # Calcular bathbedratio
        if baths and beds != 0:
            bathbedratio = beds / baths
        else:
            bathbedratio = 0

        zipCode = request.form['zipCode']
        censusTractBlock = request.form['censusTractBlock']

        # Predecir log_error
        input_data = {
            'transaction_month': transaction_month,
            'transaction_day': transaction_day,
            'calculatedfinishedsquarefeet': calculated_sqmf,
            'finishedsquarefeet12': finished_living_area,
            'finishedsquarefeet15': total_sqm,
            'latitude': 0,
            'longitude': 0,
            'propertycountylandusecode': 0,
            'propertyzoningdesc': 0,
            'rawcensustractandblock': censusTractBlock,
            'regionidcity': 0,
            'regionidneighborhood': 0,
            'regionidzip': zipCode,
            'yearbuilt': construction_year,
            'structuretaxvaluedollarcnt': structureTaxValue,
            'taxamount': taxamount,
            'bath_bed_ratio': bathbedratio,
            'distantce_to_la': 0,
            'price_per_sqft': price_sqft,
            'lot_sqft_ratio': lot_sqft_ratio
        }


        input_data_clean = {key: (VoidFloat(value) if value else 0.0) for key, value in input_data.items()}
        arrayInputs = np.array([list(input_data_clean.values())]).astype(float)
        
        if arrayInputs is not None:
            try:
                predictor = MODEL.predict(arrayInputs)
                log_error = float((np.exp(predictor) - 1) * 100)
            except Exception as e:
                print(f"Error durante la predicción: {e}")
                log_error = None


    return render_template('index.html', log_error=log_error)


if __name__ == '__main__':
    APP.run(debug=True)
