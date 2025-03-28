from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from prophet import Prophet

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "API di previsione con Prophet è attiva!"

@app.route('/', methods=['POST'])
def previsione():
    try:
        dati = request.get_json()
        df = pd.DataFrame(dati)
        modello = Prophet()
        modello.fit(df)
        futuro = modello.make_future_dataframe(periods=30)
        previsione = modello.predict(futuro)
        risultato = previsione[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_dict(orient='records')
        return jsonify(risultato)
    except Exception as e:
        return jsonify({'errore': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
