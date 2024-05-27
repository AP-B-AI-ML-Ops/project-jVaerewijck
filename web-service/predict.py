import pickle
from flask import Flask, request, jsonify
import mlflow

RUN_ID = ""

app = Flask('duration-prediction')

logged_model = f"mlflow-artifacts:/4/{RUN_ID}/artifacts/model/MLmodel"
model = mlflow.pyfunc.load_model(logged_model)

with open('./models/lin_reg.bin','rb') as f_in:
    (dv,model) = pickle.load(f_in)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s %s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance']= ride['trip_distance']
    return features

def predict(ride):
    features = prepare_features(ride)
    prediction = model.predict(features)
    return prediction[0]

@app.route('/predict', methods=['POST'])
def predict_entrypoint():
    ride = request.get_json()
    pred = predict(ride)
    result = {
        "duration":pred
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",port=9696)