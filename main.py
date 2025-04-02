from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet
from lifelines import KaplanMeierFitter, CoxPHFitter
import pandas as pd
import json

app = Flask(__name__)
CORS(app)

cox_model = None
cox_columns = []

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

@app.route('/cox', methods=['POST'])
def analisi_cox():
    global cox_model, cox_columns
    try:
        dati = request.get_json()
        df = pd.DataFrame(dati)

        df["data_inizio"] = pd.to_datetime(df["data_inizio"])
        df["data_fine"] = pd.to_datetime(df["data_fine"])
        df["durata"] = (df["data_fine"] - df["data_inizio"]).dt.days

        df_model = df.drop(columns=["data_inizio", "data_fine"])
        categoriche = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
        df_model = pd.get_dummies(df_model, columns=categoriche, drop_first=True)
        df_model = df_model.loc[:, df_model.apply(pd.Series.nunique) > 1]

        cph = CoxPHFitter()
        cph.fit(df_model, duration_col="durata", event_col="evento")

        cox_model = cph
        cox_columns = df_model.columns.tolist()

        summary = cph.summary.reset_index()
        features = []
        for _, row in summary.iterrows():
            features.append({
                "name": row[summary.columns[0]],
                "hazard_ratio": round(row["exp(coef)"], 3),
                "p_value": round(row["p"], 4),
                "interpretation": (
                    "aumenta il rischio" if row["exp(coef)"] > 1 else
                    "riduce il rischio" if row["exp(coef)"] < 1 else
                    "nessun effetto"
                )
            })

        prompt = (
            "I dati seguenti derivano da un'analisi di sopravvivenza con il modello di Cox applicata a pazienti veterinari (cani e gatti) con differenti diagnosi e trattamenti. "
            "Ogni voce include il valore di hazard ratio (HR), il p-value e un'indicazione interpretativa.\n\n"
            "Lo scopo è determinare quali fattori aumentano o riducono il rischio di mortalità, "
            "e quali terapie potrebbero risultare più efficaci, anche in assenza di significatività statistica.\n\n"
            "Analizza i risultati e fornisci una valutazione complessiva, evidenziando possibili raccomandazioni cliniche e fattori di rischio:\n\n"
        )
        for f in features:
            prompt += f"- {f['name']}: HR={f['hazard_ratio']}, p={f['p_value']} → {f['interpretation']}\n"

        output = {
            "model": "Cox Proportional Hazards",
            "features": features,
            "prompt": prompt
        }

        return jsonify(output)

    except Exception as e:
        return jsonify({"status": "errore", "messaggio": str(e)}), 400

@app.route('/valuta_paziente', methods=['POST'])
def valuta_paziente():
    global cox_model, cox_columns
    try:
        if cox_model is None:
            return jsonify({"errore": "Modello di Cox non ancora addestrato."}), 400

        nuovo_paziente = pd.DataFrame([request.get_json()])
        nuovo_paziente = pd.get_dummies(nuovo_paziente)

        for col in cox_columns:
            if col not in nuovo_paziente.columns:
                nuovo_paziente[col] = 0

        nuovo_paziente = nuovo_paziente[cox_columns]
        rischio = cox_model.predict_partial_hazard(nuovo_paziente).values[0]

        prompt = (
            f"Questo è un nuovo paziente veterinario. I suoi dati sono:\n"
            f"{json.dumps(request.get_json(), indent=2)}\n\n"
            f"Secondo il modello di Cox già addestrato, il rischio relativo stimato per questo paziente è {round(rischio, 3)}.\n"
            "Considerando i fattori di rischio e protezione emersi dall'analisi, quali indicazioni cliniche si possono fornire?"
        )

        return jsonify({"rischio_relativo": round(rischio, 3), "prompt": prompt})

    except Exception as e:
        return jsonify({"errore": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
