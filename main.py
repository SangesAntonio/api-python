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
        frequenza = payload.get("frequenza", "D")
        periods = int(payload.get("periodi", 30))

        df = pd.DataFrame(dati)
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])

        if frequenza != "D":
            df = df.set_index('ds').resample(frequenza).mean().reset_index()

        modello = Prophet()
        modello.fit(df)

        futuro = modello.make_future_dataframe(periods=periods, freq=frequenza)
        previsione = modello.predict(futuro)

        risultato = previsione[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict(orient='records')
        return jsonify({ "success": True, "previsioni": risultato })

    except Exception as e:
        return jsonify({ "success": False, "errore": str(e) }), 400

if __name__ == '__main__':
app.run(host='0.0.0.0', port=5000)
