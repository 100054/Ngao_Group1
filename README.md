# 🧠 Vaccine Uptake Prediction App

## 📌 Project Overview
This project develops a machine learning model to predict the likelihood of individuals receiving **seasonal flu** and **H1N1 vaccinations**. The solution leverages demographic, behavioral, and health-related features to support data-driven public health interventions.

The workflow is implemented in a Jupyter Notebook and deployed using **Streamlit** for real-time predictions.

---

## 🎯 Objectives
- Identify key factors influencing vaccination decisions  
- Analyze demographic and behavioral patterns  
- Build and evaluate predictive models  
- Enable real-time prediction through an interactive app  

---

## 🗂️ Project Structure
```
├── seasonal_vaccine.ipynb     # Main notebook (EDA + modeling)
├── app2.py                     # Streamlit app
├── models/                    # Saved models (.pkl)
├── requirements.txt           # Dependencies
└── README.md
```

---

## 🔍 Notebook Workflow (seasonal_vaccine.ipynb)

### 1. Data Exploration & Cleaning
- Handle missing values (imputation)
- Check class imbalance
- Encode categorical variables (Label / One-hot)
- Feature selection

### 2. Feature Engineering
- Transform demographic and behavioral variables  
- Ensure consistent feature structure for modeling  

### 3. Model Building
- Train classification models:
  - Logistic Regression  
  - Gradient Boosting / Ensemble models  
- Use pipelines for preprocessing + modeling  

### 4. Model Evaluation
- Accuracy  
- ROC-AUC  
- Confusion Matrix  

### 5. Model Export
- Save trained models using `joblib`  

```python
joblib.dump(model, "models/model.pkl")
```

---

## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-username/vaccine-prediction.git
cd vaccine-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

---

## 🚀 Run the App Locally
```bash
streamlit run app2.py
```

Open:
http://localhost:8501

---

## 🌐 Deployment (Streamlit Cloud)

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-username/vaccine-prediction.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to: https://streamlit.io/cloud  
2. Sign in with GitHub  
3. Click **New App**  
4. Select your repository  
5. Set:
   - Main file: `app2.py`  
6. Click **Deploy**

### Step 3: Configure Requirements
Ensure `requirements.txt` includes:
```
streamlit
pandas
numpy
scikit-learn
joblib
```

---

## ⚠️ Common Issues

### Feature Name Mismatch
- Ensure input columns in `app2.py` match training data  
- Maintain exact column order  

### Model Loading Errors
- Re-save models if using different versions of:
  - numpy  
  - scikit-learn  

---

## 📈 Future Improvements
- Add SHAP explainability  
- Improve UI/UX  
- Integrate real-time data collection  
- Deploy API (FastAPI + React)
