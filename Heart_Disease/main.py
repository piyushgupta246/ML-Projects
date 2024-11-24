import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("/content/heart_disease.csv")
df.head()

sns.pairplot(df[['age']])
plt.show()
sns.pairplot(df[['target']])
plt.show()

plt.figure(figsize=(15,7))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm',vmin=-1,vmax=1)
plt.title("Feature Correlation Heatmap")
plt.show()

x = df[['age','cp','thalach']]
y = df['target']

#split Data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# model initialization and training
model = LogisticRegression()
model.fit(x_train, y_train)

#Predictions and performance metrics
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)

from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, accuracy_score
# model performance
arruracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {arruracy:.2f}")

class_pred = classification_report(y_test,y_pred)

def max_heart_rate():
  age = int(input("Enter Age:"))
  cp = int(input("Enter Chest Pain(0-3):"))
  thalach = int(input("Enter maximum heart rate achived:"))

  #Create DataFrame
  user_data = pd.DataFrame([[age,cp,thalach]],columns = ['age','cp','thalach'])

  predict = model.predict(user_data)
  if predict == 0:
    print("No,Heart Desease")
  else:
    print("Yes, Person is Suffering from Heart Disease")
max_heart_rate()