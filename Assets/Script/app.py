import pickle, bz2
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import warnings
import logging

logging.basicConfig(
    level=logging.DEBUG,
    filename="./Log File.log",
    format="%(asctime)s %(levelname)s %(module)s => %(message)s ",
    datefmt="%d-%m-%Y %H:%M:%S",
)
log = logging.getLogger()
log.setLevel(logging.DEBUG)

warnings.filterwarnings("ignore")


app = Flask(__name__)

# Import Classification and Regression model file
C_pickle = bz2.BZ2File('Classification.pkl', 'rb')
model_C = pickle.load(C_pickle)


# Route for homepage
@app.route('/')
def home():
    return '''
    <html>
        <head>
            <title>Flask Setup Success</title>
        </head>
        <body>
            <h1>Flask Setup Successful!</h1>
            <p>The system is now ready to make predictions. Proceed with your requests.</p>
        </body>
    </html>
    '''

@app.route('/predictFire', methods=['POST'])
def predictFire():
    data = request.get_json(force=True)
    print("received data from controller")
    #print(data)  # print the data to check
    try:
        predictions = []        
        for feature_dict in data['data']:
            # extract features
            features = [feature_dict['Temperature'], feature_dict['Wind_Speed'], feature_dict['FFMC'], feature_dict['DMC'], feature_dict['ISI']]
            # transform features to standard format
            final_features = np.array(features, dtype=float).reshape(1, -1)            
            prediction = model_C.predict(final_features)[0]
            predictions.append(prediction)        
        return jsonify(predictions)
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

            
@app.route('/testConnection', methods=['GET'])
def testConnection():
    message = 'Connection Successful！'
    #return f'<html><body><h1>{message}</h1></body></html>'
    # A simple route to test if the connection between Unity and Flask is successful
    return jsonify({'message': 'Connection successful!'})

# Run APP in Debug mode

if __name__ == "__main__":
    app.run(debug=True, port= 5000)
