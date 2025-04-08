import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
a=pd.read_csv("diabetes_prediction_dataset.csv")
x=a.drop(columns=["diabetes"])
x=pd.get_dummies(x)
y=a["diabetes"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
accuracy=accuracy_score(y_test,y_predict)
print("Logistic Regression Accuracy:")
print(accuracy)
joblib.dump(model, "mama.h5")
with open("accuracy1.txt", "w") as f:
    f.write(str(accuracy))

//train
