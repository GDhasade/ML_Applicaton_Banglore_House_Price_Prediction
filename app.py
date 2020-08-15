from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
# This file should read json file and retun all col names in json file
import json
import warnings
warnings.filterwarnings("ignore")
import sklearn
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
__model = None
__data_columns = None
__locations = None


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predicts", methods=['POST'])
def predicts():
    if request.method == 'POST':
        
        total_sqft = float(request.form['Squareft'])
        location = request.form['uilocation']
        bhk = int(request.form['uiBHK'])
        bath = int(request.form['uiBathrooms'])
        
        #load artifacts
        print("Loading saved artifacts...START")
        global __data_columns
        global __locations
        with open("./artifacts/columns.json", 'r') as f:
            __data_columns = json.load(f)['data_columns']
            __locations = __data_columns[3:]  # as 4 col we have till 3 after that we have all locations
    
        global __model
        with open("./artifacts/Banglore_House_Price_Prediction_Model.pickle", 'rb') as f:
            __model = pickle.load(f)
    
        print("loading saved artifacts...DONE")
        
        #get_estimated_price
        try:
            loc_index = __data_columns.index(location.lower())
        except:
            loc_index = -1
    
        x = np.zeros(len(__data_columns))
        x[0] = total_sqft
        x[1] = bath
        x[2] = bhk
    
        if loc_index >= 0:
            x[loc_index] = 1
    
        
        response = round(__model.predict([x])[0], 2)
    
        return render_template('index.html',prediction_text="Predicted: {}k".format(response))
    
    else:
        return render_template('index.html')
    

    
def get_estimated_price(location, sqft, bhk, bath):
    
    load_saved_artifacts()
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)


def get_location_names():
    load_saved_artifacts()
    return __locations


def load_saved_artifacts():

    print("Loading saved artifacts...START")
    global __data_columns
    global __locations
    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # as 4 col we have till 3 after that we have all locations

    global __model
    with open("./artifacts/Banglore_House_Price_Prediction_Model.pickle", 'rb') as f:
        __model = pickle.load(f)

    print("loading saved artifacts...DONE")



if __name__=="__main__":
    
    app.run(debug=True)
    

