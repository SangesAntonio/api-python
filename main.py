from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet
import pandas as pd

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
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])

        # Raggruppa per settimana e calcola media
        df_settimanale = df.set_index('ds').resample('W').mean().reset_index()

        # Crea e addestra il modello
        modello = Prophet()
        modello.fit(df_settimanale)

        # Previsioni per 52 settimane
        futuro = modello.make_future_dataframe(periods=52, freq='W')
        previsione = modello.predict(futuro)

        risultato = previsione[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(52).to_dict(orient='records')
        return jsonify({ "success": True, "previsioni": risultato })

    except Exception as e:
        return jsonify({'errore': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
