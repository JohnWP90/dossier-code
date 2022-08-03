from random import random
import string
from flask import Flask, render_template, url_for, request, jsonify, json
import pickle
import numpy as np
import plotly
import plotly.express as px
import json
from sklearn.model_selection import train_test_split
import pandas as pd


app = Flask(__name__)
model = pickle.load(open('LGBM_model.pkl','rb'))

def get_dataset(csv_file, add_TARGET=False):
    '''This function is to import the required dataset'''
    data = pd.read_csv(csv_file).drop(columns='Unnamed: 0')
    if add_TARGET == False:
        data = data.loc[:, ['SK_ID_CURR','DAYS_BIRTH','EXT_SOURCE_1', 'EXT_SOURCE_2','EXT_SOURCE_3', 'PAYMENT_RATE']]
    else:
        data = data.loc[:, ['SK_ID_CURR','DAYS_BIRTH','EXT_SOURCE_1', 'EXT_SOURCE_2','EXT_SOURCE_3', 'PAYMENT_RATE', 'TARGET']]
    return data

# Importing data and training the model.
train_data = get_dataset('processed_train_data.csv', add_TARGET=True)
test_data = get_dataset('processed_test_data.csv')
Xtrain, Xtest, Ytrain, Ytest = train_test_split(train_data.drop(columns='TARGET'), train_data.TARGET, test_size=0.2, random_state=789)
model.fit(Xtrain.values,Ytrain)

@app.route('/', methods=['GET','POST'])
def predict():
    all_applicant = test_data.SK_ID_CURR.tolist()

    if request.json['applicant_id'] in all_applicant:
        applicant = request.json['applicant_id']
        applicant_id = request.json['applicant_id']
    else:
        return ("This applicant is not in the database. Please try another ID.")

    # Filter on the chosen applicant ID.
    client_data = test_data.loc[test_data.SK_ID_CURR==applicant_id,:].values

    # Predicting the probability for the applicant to pay back.
    prediction_proba = model.predict_proba(client_data)
    proba_paying = round(prediction_proba[0][0], 2)

    return jsonify({"Scoring ": proba_paying})

if __name__ == "__main__" :
    app.run(debug=True)
