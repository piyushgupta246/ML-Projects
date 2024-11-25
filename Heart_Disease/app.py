from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv("D:\F L A S K web\Heart_Disease\heart_disease.csv")
x = df[['age', 'cp', 'thalach']]
y = df['target']

# Split data and train the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        cp = int(request.form['cp'])
        thalach = int(request.form['thalach'])

        # Create a DataFrame for prediction
        user_data = pd.DataFrame([[age, cp, thalach]], columns=['age', 'cp', 'thalach'])
        prediction = model.predict(user_data)[0]
        
        result = "Yes, the person is suffering from heart disease." if prediction == 1 else "No, the person is not suffering from heart disease."
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
