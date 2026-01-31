# 🚢 Titanic Survival Predictor

An end-to-end Machine Learning project that predicts passenger survival on the Titanic using Random Forest algorithm.

## 🎯 Live Demo
Coming soon... (You'll add the Render URL here after deployment)

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **ML:** scikit-learn, pandas, numpy
- **Model:** Random Forest Classifier
- **Deployment:** Render.com

## 📊 Features
- Interactive web interface
- Real-time predictions
- 76%+ accuracy on test data
- Handles missing data and feature engineering

## 🚀 How It Works
1. User inputs passenger details (age, class, fare, etc.)
2. App processes and engineers features
3. Random Forest model makes prediction
4. Results displayed with confidence score

## 📁 Project Structure
```
titanic-deployment/
├── app.py                          # Flask application
├── requirements.txt                # Dependencies
├── titanic_rf_model.joblib        # Trained model
├── titanic_feature_names.joblib   # Feature schema
└── templates/
    └── index.html                 # Frontend
```

## 💻 Local Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/titanic-ml-deployment.git
cd titanic-ml-deployment

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

Visit `http://localhost:5001` in your browser.

## 📈 Model Performance
- Algorithm: Random Forest
- Accuracy: 76%+
- Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- Feature Engineering: FamilySize, IsAlone, FarePerPerson

## 👨‍💻 Author
**Your Name**
- GitHub: [yogeshYO](https://github.com/yogeshYO)
- LinkedIn: [Yogesh Baral](https://www.linkedin.com/in/yogesh-baral-bbaaa631)

## 📝 License
MIT License
