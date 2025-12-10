from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load Random Forest model
model = joblib.load("models/rf_model.joblib")

# Load CSV datasets
trends_df = pd.read_csv("models/Trends of Incidents and Fatalities across five states in northwest Nigeria.csv")
banditry_df = pd.read_csv("models/Incident of banditry across five states in northwest Nigeria.csv")
pattern_df = pd.read_csv("models/Pattern of Violence by Bandits across five states in northwest Nigeria.csv")

# Create cluster stats from banditry_df
cluster_stats = banditry_df.groupby(['Latitude', 'Longitude']).size().reset_index(name='incidents')
cluster_stats.rename(columns={'Latitude':'mean_lat','Longitude':'mean_lon'}, inplace=True)

# Standard scaler (fit on numeric features from banditry_df as reference)
scaler = StandardScaler()
scaler.fit(banditry_df[['Latitude', 'Longitude']].fillna(0))  # simple fit on reference data

def nearest_cluster(lat, lon):
    d = np.sqrt((cluster_stats['mean_lat'] - lat)**2 + (cluster_stats['mean_lon'] - lon)**2)
    idx = d.idxmin()
    return cluster_stats.loc[idx, 'incidents']

@app.route('/', methods=['GET','POST'])
def home():
    result = None

    if request.method == 'POST':
        lat = float(request.form['lat'])
        lon = float(request.form['lon'])
        ts = pd.to_datetime(request.form['timestamp'])

        cluster_count = nearest_cluster(lat, lon)

        # Prepare features
        X = pd.DataFrame([{
            'hour': ts.hour,
            'dayofweek': ts.dayofweek,
            'month': ts.month,
            'victims': 1,
            'cluster_incident_count': cluster_count,
            'crime_type_Unknown': 1  # since we don't have preprocessor, just add default
        }])

        # Scale numeric features manually
        numeric_features = ['hour', 'dayofweek', 'month', 'victims', 'cluster_incident_count']
        X[numeric_features] = scaler.transform(X[numeric_features].fillna(0))

        # Predict probability
        prob = model.predict_proba(X)[:,1][0]

        if prob < 0.25:
            result = f'LOW RISK ({prob*100:.2f}%)'
        elif prob < 0.6:
            result = f'MEDIUM RISK ({prob*100:.2f}%)'
        else:
            result = f'HIGH RISK ({prob*100:.2f}%)'

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
