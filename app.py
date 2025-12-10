from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load your trained items
model = joblib.load("models/rf_model.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")
cluster_stats = pd.read_csv("models/cluster_stats.csv")

def nearest_cluster(lat, lon):
    d = np.sqrt((cluster_stats["mean_lat"] - lat)**2 +
                (cluster_stats["mean_lon"] - lon)**2)
    idx = d.idxmin()
    return cluster_stats.loc[idx, "incidents"]

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        lat = float(request.form["lat"])
        lon = float(request.form["lon"])
        ts = pd.to_datetime(request.form["timestamp"])

        cluster_count = nearest_cluster(lat, lon)

        X = pd.DataFrame([{
            "hour": ts.hour,
            "dayofweek": ts.dayofweek,
            "month": ts.month,
            "victims": 1,
            "cluster_incident_count": cluster_count,
            "crime_type": "Unknown"
        }])

        Xp = preprocessor.transform(X)
        prob = model.predict_proba(Xp)[:, 1][0]

        if prob < 0.25:
            result = f"LOW RISK ({prob*100:.2f}%)"
        elif prob < 0.6:
            result = f"MEDIUM RISK ({prob*100:.2f}%)"
        else:
            result = f"HIGH RISK ({prob*100:.2f}%)"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
