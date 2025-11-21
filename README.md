# 🚗 Vehicle Price Prediction using Machine Learning (MLP Regressor)

This project predicts the selling price of a vehicle using a fully connected neural network model implemented with **MLPRegressor** from Scikit-Learn. The model is trained on key vehicle attributes like mileage, year, fuel type, transmission, and engine specifications. A Streamlit web app is included for real-time predictions.

---

## ✨ Features

- ✔ Data preprocessing and categorical encoding  
- ✔ Neural network-based regression using **MLPRegressor**  
- ✔ Trained model saved as `.pkl`  
- ✔ Web-based prediction interface using Streamlit  
- ✔ Ready for deployment on cloud platforms (Render / HuggingFace / Streamlit Cloud)

---

## 🛠 Tech Stack

| Component | Technologies |
|----------|--------------|
| Programming Language | Python |
| Machine Learning | Scikit-Learn (MLPRegressor) |
| Preprocessing | Pandas, NumPy |
| UI / Deployment | Streamlit |
| Model Storage | Joblib / Pickle |

---

## 📂 Project Structure

├── data/
│ └── dataset.csv
├── model/
│ └── vehicle_model.pkl
├── src/
│ ├── preprocessing.py
│ ├── train_mlp.py
│ └── predict.py
├── app.py # Streamlit App
├── requirements.txt
└── README.md

