from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet
from lifelines import KaplanMeierFitter
import pandas as pd

app = Flask(__name__)
CORS(app)

# Route di controllo
@app.route('/')
def home():
    return "API di previsione con Prophet è attiva!"

# API Prophet
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

        model_args = {
            "growth": params.get("growth", "linear"),
            "yearly_seasonality": params.get("yearly_seasonality", True),
            "weekly_seasonality": False,
            "changepoint_prior_scale": params.get("changepoint_prior_scale", 0.1)
        }

        modello = Prophet(**model_args)

        modello.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )

        modello.fit(df)

        futuro = modello.make_future_dataframe(periods=periodi, freq=frequenza)
        previsione = modello.predict(futuro)

        colonne = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        if 'yearly' in previsione.columns:
            colonne.append('yearly')
        if 'monthly' in previsione.columns:
            colonne.append('monthly')

        risultato = previsione[colonne].tail(periodi).to_dict(orient='records')
        return jsonify({ "success": True, "previsioni": risultato })

    except Exception as e:
        return jsonify({'errore': str(e)}), 400

# ✅ API Sopravvivenza
@app.route('/sopravvivenza', methods=['POST'])
def sopravvivenza():
    try:
        dati = request.get_json()
        df = pd.DataFrame(dati)

        df["data_inizio"] = pd.to_datetime(df["data_inizio"])
        df["data_fine"] = pd.to_datetime(df["data_fine"])
        df["durata"] = (df["data_fine"] - df["data_inizio"]).dt.days

        output = []
        kmf = KaplanMeierFitter()

        for terapia in [col for col in df.columns if col.startswith("terapia_")]:
            for valore in [0, 1]:
                gruppo = df[df[terapia] == valore]
                if not gruppo.empty:
                    kmf.fit(gruppo["durata"], event_observed=gruppo["evento"], label=f"{terapia}={valore}")
                    surv = kmf.survival_function_.reset_index()
                    output.append({
                        "terapia": terapia,
                        "valore": int(valore),
                        "giorni": surv["timeline"].tolist(),
                        "probabilita": surv[kmf._label].tolist()
                    })

        return jsonify({"status": "ok", "risultati": output})
    
    except Exception as e:
        return jsonify({"status": "errore", "messaggio": str(e)}), 400

# Avvio server (usato solo in locale o su Render se serve)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
