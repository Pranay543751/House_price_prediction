# 🏠 House Price Prediction

A Machine Learning based **House Price Prediction** application that predicts house prices based on selected property features. The trained machine learning model is saved using **Joblib** and can be accessed through a **FastAPI** backend.

## 🚀 Features

* Predict house prices using machine learning
* Trained using **Scikit-learn**
* Data processing with **Pandas**
* Model serialization using **Joblib**
* REST API built with **FastAPI**
* Separate scripts for model training and data exploration
* Lightweight and easy to deploy

## 🛠️ Tech Stack

* **Python**
* **FastAPI**
* **Scikit-learn**
* **Pandas**
* **Joblib**
* **Uvicorn**

## 📁 Project Structure

```text
house_price_prediction/
│
├── main.py                  # FastAPI application
├── train.py                 # Model training script
├── explore.py               # Data exploration
├── house_model.joblib       # Trained ML model
├── house_features.joblib    # Model feature information
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignored files
└── README.md                # Project documentation
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/house-price-prediction.git
cd house-price-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**

```powershell
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the FastAPI server using:

```bash
uvicorn main:app --reload
```

The application will be available at:

```text
http://127.0.0.1:8000
```

## 📚 API Documentation

FastAPI automatically provides interactive API documentation.

### Swagger UI

```text
http://127.0.0.1:8000/docs
```

### ReDoc

```text
http://127.0.0.1:8000/redoc
```

You can use Swagger UI to test the prediction API directly from your browser.

## 🧠 Machine Learning Workflow

The project follows a basic machine learning pipeline:

```text
Dataset
   ↓
Data Exploration
   ↓
Data Preprocessing
   ↓
Feature Selection
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
FastAPI API
   ↓
House Price Prediction
```

## 📌 Model Files

The trained model is stored in:

```text
house_model.joblib
```

Feature-related information is stored in:

```text
house_features.joblib
```

These files are loaded by the FastAPI application to generate predictions without retraining the model every time.

> **Note:** The trained model is managed using Git LFS because the model file is larger than GitHub's standard 100 MB file limit.

## 🔮 Future Improvements

* Add a web-based frontend
* Improve model accuracy through hyperparameter tuning
* Add multiple machine learning algorithms
* Compare model performance
* Add input validation and better error handling
* Deploy the FastAPI application to a cloud platform
* Add authentication and API security

## 👨‍💻 Author

**Pranay Zade**

B.Tech Computer Engineering Student

---

⭐ If you found this project useful, consider giving the repository a star!
