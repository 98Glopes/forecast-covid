from flask import Flask

from predict_covid.model import CovidModel



app = Flask(__name__)

@app.route('/forecast-covid/predict/<int:days>/', methods=['GET'])
def predict(days):
    model = CovidModel()
    model.load_model()

    return model.predict(days=days)

@app.route('/forecast-covid/update/', methods=['GET'])
def update():
    model = CovidModel()
    model.get_new_model(force_remote_dataset=True)

    respone = {'message': 'Updated model'}
    return respone