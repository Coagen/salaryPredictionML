import pandas as pd
from sklearn import linear_model
import joblib

data=pd.read_csv('data\\data.csv')

lr=linear_model.LinearRegression()
lr.fit(data[["YearsExperience"]],data["Salary"])

joblib.dump(lr,'model\\lr_model.pkl')
print("Model saved")
