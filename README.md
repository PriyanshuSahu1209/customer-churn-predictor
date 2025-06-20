# 🧠 Customer Churn Predictor

This project is a machine learning-based **customer churn prediction system** built using an Artificial Neural Network (ANN). It takes key customer data from a bank and predicts whether a customer is likely to churn (leave the bank).

The project also includes an interactive **Streamlit app** for live predictions using a trained model.

---

## 🚀 Features

- Preprocessing: Label encoding, One-Hot Encoding, Feature Scaling
- ANN classifier (built using TensorFlow/Keras)
- Train/Test split with EarlyStopping
- Live prediction via Streamlit interface
- Modular & reproducible: `experiments.py` for training, `app.py` for deployment

---

## 📁 Project Structure

```
customer-churn-predictor/
├── data/
│   └── Churn_Modelling.csv
├── models/
│   ├── model.h5
│   ├── scaler.pkl
│   ├── label_encoder_gender.pkl
│   └── onehot_encoder_geo.pkl
├── notebooks/
│   └── experiments.ipynb
├── app.py
├── experiments.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

- **Source**: [Kaggle - Customer Churn Modeling](https://www.kaggle.com/datasets/shubhendra7/customer-churn-modeling)
- **Features used**:
  - Credit Score
  - Geography (One-Hot Encoded)
  - Gender (Label Encoded)
  - Age, Tenure, Balance, Number of Products
  - Has Credit Card, Is Active Member
  - Estimated Salary
- **Target**: `Exited` (1 = churned, 0 = stayed)

---

## 🧪 Model Training

To train and save the model:

```bash
python experiments.py
```

---

## 🖥️ Running the Streamlit App

```bash
streamlit run app.py
```

You’ll see an interactive form to input customer data and get churn predictions instantly.

---

## 📦 Dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## ✅ Future Improvements

- Add support for uploading CSVs for batch prediction
- Add visualizations of prediction confidence
- Convert to Flask/Django API (for deployment on cloud)

---

## 🔗 License

MIT License

---

## 👤 Author

Made by Priyanshu Sahu
