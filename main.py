from flask import Flask, request, jsonify
from prophet import Prophet
import pandas as pd

@app.route("/", methods=["POST"])
def previsione():
    try:
        dati = request.get_json()
        df = pd.DataFrame(dati)
        #  Assicura che 'ds' sia datetime
        df['ds'] = pd.to_datetime(df['ds'])
        #  Raggruppa per settimana e calcola media
        df_settimanale = df.set_index('ds').resample('W').mean().reset_index()
        #  Crea e addestra il modello
        modello = Prophet()
        modello.fit(df_settimanale)
        #  52 settimane = 1 anno, frequenza settimanale
        futuro = modello.make_future_dataframe(periods=52, freq='W')
        previsione = modello.predict(futuro)

        # ✅ Ritorna tutte le 52 settimane (o meno se vuoi)
        risultato = previsione[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(52).to_dict(orient='records')
        return jsonify({ "success": True, "previsioni": risultato })
    except Exception as e:
        return jsonify({ "success": False, "errore": str(e) }), 400
