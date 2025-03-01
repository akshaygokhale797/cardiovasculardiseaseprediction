## **Cardiovascular Disease Prediction**
A web-based tool and machine learning model to predict cardiovascular disease risk based on user inputs.

### **📌 Project Overview**
This project provides an interface to predict the likelihood of cardiovascular disease using a **machine learning model**. Users can enter their age, gender, cholesterol level, glucose level, and other health-related factors, and the model will provide a prediction.

### **🔧 Features**
- **Web Interface:** Built with HTML, CSS for easy data entry.
- **Machine Learning Model:** Trained using `XGBoost` to classify cardiovascular disease risk.
- **Data Processing:** Feature engineering and transformations applied to the dataset.
- **Interactive Form:** Users can input their health details and get an instant prediction.

### **🚀 Setup Instructions**
To run this project locally, follow these steps:

#### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/cardiovascular-disease-prediction.git
cd cardiovascular-disease-prediction
```

#### **2️⃣ Set Up a Virtual Environment**
```bash
python -m venv venv
venv\Scripts\activate      
```

#### **4️⃣ Run the Web Application**
```bash
python app.py 
```

### **📊 Dataset**
- The dataset used for training the model consists of patient records with various health metrics.
- The data includes **age, gender, cholesterol levels, glucose levels, blood pressure, BMI, and more.**
- **https://www.kaggle.com/datasets/colewelkins/cardiovascular-disease**

### **📚 Technologies Used**
- **Frontend:** HTML, CSS
- **Backend:** Python, Flask
- **Machine Learning:** XGBoost, Scikit-Learn, Pandas
- **Data Processing:** Feature engineering, StandardScaler, OneHotEncoding

### **⚠️ Notes**
- **The model should not be used for medical decisions.** It is a **proof of concept** and should not replace professional medical advice.
