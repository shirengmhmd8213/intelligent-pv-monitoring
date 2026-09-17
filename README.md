# 🌞 Intelligent PV Monitoring System

An AI-powered digital twin platform for solar photovoltaic (PV) fault 
detection and power forecasting, with a real-time web dashboard.

---

## 🎯 Overview

This project implements a complete **Digital Twin** of a solar PV system 
in **MATLAB/Simulink**, which is used for:

- **Fault data generation** — Simulating 8 different fault scenarios
- **Annual dataset synthesis** — Generating one full year of operational 
  data for training AI models

The trained models are deployed in a **Streamlit-based web dashboard** 
for real-time monitoring, fault diagnostics, and power forecasting.

---

## 🧠 AI Models

| Task | Algorithm | Classes / Output |
|------|-----------|------------------|
| **Fault Detection** | Random Forest | 8 fault classes |
| **Power Forecasting** | XGBoost | Hourly PV power output |

### Fault Classes Detected
1. Healthy
2. Soiling 10%
3. Soiling 20%
4. Damage 20%
5. Damage 30%
6. Damage 40%
7. Open Circuit
8. Short Circuit

> 💡 **Scalable & Customizable:** The fault classification framework is 
> fully extensible — new fault classes can be added by retraining the 
> model with additional data. The class set is also customizable per 
> project, allowing adaptation to the specific requirements of any 
> solar power plant.

## 🌐 Web Dashboard

The dashboard provides:

- **Real-time telemetry** via MQTT (voltage, current, power, 
  irradiance, temperature)
- **AI-based fault detection** with majority voting
- **3-day power forecast** using Open-Meteo live weather data
- **Interactive charts** for solar irradiance and PV parameters
- **Performance snapshots** and audit logs

🔗 **Live Dashboard:** [intelligent-pv-monitoring.streamlit.app](https://intelligent-pv-monitoring-lnkzrzaqr6vurpjdfqxm7i.streamlit.app)

> ⚠️ **Note:** The dashboard is hosted on Streamlit Cloud and may not be directly accessible from Iran without a VPN.
---

## ⚙️ Tech Stack

**Simulation:** MATLAB / Simulink  
**Machine Learning:** Scikit-learn (Random Forest), XGBoost  
**Web Framework:** Streamlit  
**Communication:** MQTT (paho-mqtt)  
**Weather API:** Open-Meteo  
**Visualization:** Plotly  
**Deployment:** Streamlit Cloud

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/shirengmhmd8213/intelligent-pv-monitoring.git
cd intelligent-pv-monitoring

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
