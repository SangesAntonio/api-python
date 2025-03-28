from flask import Flask, request, jsonify
from prophet import Prophet
import pandas as pd
from datetime import timedelta

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return "L'API è attiva, Antonio! 🚀"


@app.route("/", methods=["POST"])
def previsione():
    try:
        data = request.get_json()
        # Aspettati un array di oggetti con 'ds' e 'y'
        df = pd.DataFrame(data)

        # Crea e addestra il modello
        model = Prophet()
        model.fit(df)

        # Crea il dataframe futuro (30 giorni avanti)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Ritorna solo le colonne principali
        risultato = forecast[['ds', 'yhat', 'yhat_lower',
                              'yhat_upper']].tail(30).to_dict(orient='records')
        return jsonify({"success": True, "previsioni": risultato})

    except Exception as e:
        return jsonify({"success": False, "errore": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
