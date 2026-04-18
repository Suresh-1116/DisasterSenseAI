 DisasterSenseAI

Real-Time Multi-Modal Disaster Risk Intelligence Platform

<img width="1914" height="969" alt="Screenshot 2026-04-18 132949" src="https://github.com/user-attachments/assets/45f05be4-93b2-46f5-ba2a-344d8bb31195" />

<img width="1905" height="917" alt="Screenshot 2026-04-18 133014" src="https://github.com/user-attachments/assets/a6116a2d-d486-4aaf-bc25-09796ca3c8ca" />

<img width="1908" height="919" alt="Screenshot 2026-04-18 133044" src="https://github.com/user-attachments/assets/f48e8c44-85e5-46b7-9a1c-be072367171a" />



An AI system that fuses live earthquake data, real-time weather signals, and NLP-based human distress detection into a unified disaster risk score — with explainable AI showing exactly WHY each decision was made.

 Live Demo →


 The Problem :
 
When disasters strike, emergency responders face three critical questions:

Where is the danger right now?

How severe is the risk in each location?

Why is that area flagged as high risk?


Existing systems answer these questions in silos — seismic agencies track earthquakes, weather services track storms, social media monitors track human signals. No open-source system fuses all three in real time.
DisasterSenseAI solves this.

 
 How It Works :

Live Seismic Data (USGS) --------──┐

  Live Weather Data (OpenWeather)      ─┼──► XGBoost ML Model ──► Risk Score + SHAP Explanation

  NLP Distress Signals --------------──┘    Trained on   ↑

(HuggingFace Transformers)         5,000 scenarios
Three Data Sources Fused in Real Time
SourceDataUpdate FrequencyUSGS Earthquake APIMagnitude, location, depthEvery 5 minutesOpenWeather APIWind speed, temperature, storm severityEvery 5 minutesHuggingFace NLPDistress signal detection from textOn demand

 
 Features :

 Live Earthquake Map — Real-time USGS data plotted on interactive world map

 Weather Fusion — OpenWeather API layered with seismic signals per location

 NLP Distress Detection — Zero-shot HuggingFace classifier detects SOS messages with confidence scores

 XGBoost Risk Model — 94% accuracy, 0.9777 AUC-ROC, trained on 5,000 disaster scenarios

 SHAP Explainability — Every prediction comes with a bar chart explaining the top risk factors

 Live Alert Panel — Auto-ranked danger zones with expandable SHAP explanations

 Custom Scenario Predictor — Input any values and get instant AI prediction with explanation



 
 Screenshots :
Main Dashboard

<img width="1914" height="969" alt="Screenshot 2026-04-18 132949" src="https://github.com/user-attachments/assets/45f05be4-93b2-46f5-ba2a-344d8bb31195" />

SHAP Explainability

<img width="1905" height="917" alt="Screenshot 2026-04-18 133014" src="https://github.com/user-attachments/assets/a6116a2d-d486-4aaf-bc25-09796ca3c8ca" />

Custom Scenario Predictor

<img width="1908" height="919" alt="Screenshot 2026-04-18 133044" src="https://github.com/user-attachments/assets/f48e8c44-85e5-46b7-9a1c-be072367171a" />


Tech Stack :
LayerTechnologyML ModelXGBoost (94% accuracy, 0.9777 AUC-ROC)ExplainabilitySHAP (TreeExplainer)NLPHuggingFace Transformers (facebook/bart-large-mnli)Seismic DataUSGS Earthquake APIWeather DataOpenWeatherMap APIFrontendStreamlitMapsFolium + streamlit-foliumDeploymentStreamlit Cloud


 Run Locally :
1. Clone the repo
bashgit clone https://github.com/Suresh-1116/DisasterSenseAI.git
cd DisasterSenseAI
2. Create virtual environment
bashpython -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
3. Install dependencies
bashpip install -r requirements.txt
4. Set up API key
Create a .env file in the root folder:
OPENWEATHER_API_KEY=your_api_key_here
Get a free key at openweathermap.org
5. Train the model
bashpython train_model.py
6. Run the app
bashpython -m streamlit run app.py

 
 Model Performance :
MetricScoreAccuracy94%AUC-ROC0.9777Precision (High Risk)95%Recall (High Risk)98%F1 Score0.96
Feature Importance (SHAP)
The model learned that magnitude and distress signal count are the strongest predictors of disaster risk, followed by wind speed and population density.

 
 Project Structure :
 
DisasterSenseAI/

│

├── app.py                  # Main Streamlit dashboard

├── distress_detector.py    # HuggingFace NLP module

├── risk_predictor.py       # ML prediction + SHAP explanation

├── train_model.py          # XGBoost model training pipeline

│

├── disaster_model.pkl      # Trained XGBoost model

├── shap_explainer.pkl      # SHAP TreeExplainer

│

├── requirements.txt        # Python dependencies

├── .env                    # API keys (not committed)

└── .gitignore


 
 What I Learned :
Building this project from scratch taught me:

Multi-modal AI — fusing structured data (seismic, weather) with unstructured data (text signals)
Explainable AI — using SHAP to make ML decisions transparent and trustworthy
Real-time data pipelines — connecting live APIs and handling failures gracefully
Production deployment — Docker concepts, environment secrets, Streamlit Cloud


 
 Future Improvements :

 Integrate real Twitter/X API for live distress signal detection
 Add satellite imagery analysis using YOLOv8 for building damage detection
 Historical simulation mode using past disaster datasets (Kerala floods 2018)
 SMS/email alert system for high-risk zone notifications
 Mobile-responsive UI


 
 Author :
V Suresh Kumar

GitHub: https://github.com/Suresh-1116
LinkedIn: https://www.linkedin.com/in/suresh-kumar-43a458255/
Email: vsureshkumar1116@gmail.com


 
 License :
MIT License — feel free to use and build on this project.

 
 
 If this project helped you, please give it a star!
