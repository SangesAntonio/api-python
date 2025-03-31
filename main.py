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
        payload = request.get_json()

        dati = payload.get("dati")
        frequenza = payload.get("frequenza", "W")
        periodi = int(payload.get("periodi", 52))
        params = payload.get("params", {})

        df = pd.DataFrame(dati)
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])

        if frequenza != "D":
            df = df.set_index('ds').resample(frequenza).mean().reset_index()

        # Imposta parametri Prophet
        # Imposta parametri Prophet
        model_args = {
            "growth": params.get("growth", "linear"),
            "yearly_seasonality": params.get("yearly_seasonality", True),
            "weekly_seasonality": params.get("weekly_seasonality", True),
            "changepoint_prior_scale": params.get("changepoint_prior_scale", 0.1)
        }

        modello = Prophet(**model_args)
        modello.fit(df)

        futuro = modello.make_future_dataframe(periods=periodi, freq=frequenza)
        previsione = modello.predict(futuro)

        risultato = previsione[['ds', 'yhat', 'yhat_lower', 'yhat_upper','yearly','weekly']].tail(periodi).to_dict(orient='records')
        return jsonify({ "success": True, "previsioni": risultato })

    except Exception as e:
        return jsonify({'errore': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
