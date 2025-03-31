from flask import Flask, request, jsonify
from lifelines import KaplanMeierFitter
import pandas as pd
from datetime import datetime

app = Flask(__name__)

@app.route('/sopravvivenza', methods=['POST'])
def analizza_sopravvivenza():
    data = request.get_json()

    try:
        df = pd.DataFrame(data)

        # Conversione date
        df["data_inizio"] = pd.to_datetime(df["data_inizio"])
        df["data_fine"] = pd.to_datetime(df["data_fine"])
        df["durata"] = (df["data_fine"] - df["data_inizio"]).dt.days

        kmf = KaplanMeierFitter()
        output = []

        # Trova tutte le colonne che iniziano con "terapia_"
        terapie = [col for col in df.columns if col.startswith("terapia_")]

        for terapia in terapie:
            for valore in [0, 1]:
                gruppo = df[df[terapia] == valore]
                if not gruppo.empty:
                    kmf.fit(gruppo["durata"], event_observed=gruppo["evento"], label=f"{terapia}={valore}")
                    survival = kmf.survival_function_.reset_index()
                    output.append({
                        "terapia": terapia,
                        "valore": int(valore),
                        "giorni": survival["timeline"].tolist(),
                        "probabilita_sopravvivenza": survival[kmf._label].tolist()
                    })


        return jsonify({"status": "ok", "risultati": output})

    except Exception as e:
        return jsonify({"status": "errore", "messaggio": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
