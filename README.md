# 💳 Credit Scoring & Risk Prediction Model

**CodeAlpha Machine Learning Internship - Task 1**

An end-to-end Machine Learning web application designed to evaluate creditworthiness and predict loan default risk based on applicant financial profiles. Built with **Scikit-Learn**, **Pandas**, and **Streamlit**.

---

## 📌 Features

* **Data Preprocessing & Scaling**: Normalizes numeric attributes using `StandardScaler`.
* **Balanced Machine Learning Pipeline**: Implements a `RandomForestClassifier` tuned with class-weight balancing.
* **Interactive Web Interface**: Streamlit UI allows real-time feature inputs and outputs automated approval status and default probabilities.
* **Model Serialization**: Trained model pipeline saved via `joblib`.

---

## 🛠️ Project Structure

```text
├── app.py                # Streamlit Web Application interface
├── credit_model.pkl      # Trained Random Forest Model
├── scaler.pkl            # Pretrained StandardScaler object
├── credit_data.csv       # Financial Dataset
└── README.md             # Project documentation
