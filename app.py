#from random import random
#import string
from flask import Flask, render_template, url_for, request, jsonify, json
import pickle
import numpy as np
import plotly
import plotly.express as px
import json
from sklearn.model_selection import train_test_split
import pandas as pd
import lightgbm


app = Flask(__name__)
model = pickle.load(open('LGBM_model.pkl','rb'))

def get_dataset(csv_file, add_TARGET=False):
    '''This function is to import the required dataset'''
    data = pd.read_csv(csv_file).drop(columns='Unnamed: 0')
    return data

# Importing data and training the model.
test_data_with_id = get_dataset('processed_test_data.csv')
test_data = get_dataset('processed_test_data.csv').drop(columns="SK_ID_CURR")

@app.route('/', methods=['GET','POST'])
def predict():
    applicant_id=request.json['applicant_id']
    #applicant_id=request.json['id']

    all_applicant = test_data_with_id.loc[:,'SK_ID_CURR'].tolist()
    
    if applicant_id in all_applicant:
        # Filter on the chosen applicant ID.
        client_data = test_data.loc[test_data_with_id.SK_ID_CURR==applicant_id,:].values

        # Predicting the probability for the applicant to pay back.
        prediction_proba = model.predict_proba(client_data)
        proba_paying = round(prediction_proba[0][0], 2)

        return jsonify({"Score ": proba_paying})
    else:
        return ("This applicant is not in the database. Please try another ID.")

if __name__ == "__main__" :
    app.run(debug=True)