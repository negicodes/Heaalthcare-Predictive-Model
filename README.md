# Heaalthcare-Predictive-Model
Use healthcare data sets to build predictive model for a disease  Diagnostic, patient readmission rates or treatment outcome adding Healthcare provides in decision making

<<<<<<<    CODE    >>>>>>     <<<<<<<<<<  CODE    >>>>>>>>>      <<<<<<<    CODE    >>>>>>     <<<<<<<<<<  CODE    >>>>>>>>>    
#IGNORE <BR> TAG , BASICALLY USED FOR MAINTAING GAPS IN PREVIEW FORMAT

import pandas as pd<br>
from sklearn.model_selection import train_test_split<br>
from sklearn.preprocessing import StandardScaler<br>
from sklearn.ensemble import RandomForestClassifier<br>
from sklearn.metrics import accuracy_score, classification_report<br>


data = pd.read_csv('data.csv')<br>

X = data.drop(columns=['Target_Column'])<br>
y = data['Target_Column']<br>

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)<br>

scaler = StandardScaler()<br>
X_train_scaled = scaler.fit_transform(X_train)<br>
X_test_scaled = scaler.transform(X_test)<br>

clf = RandomForestClassifier(n_estimators=100, random_state=42)<br>
clf.fit(X_train_scaled, y_train)<br>

y_pred = clf.predict(X_test_scaled)<br>

accuracy = accuracy_score(y_test, y_pred)<br>
print("Accuracy:", accuracy)<br>

print("Classification Report:")<br>
print(classification_report(y_test, y_pred))<br>

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': clf.feature_importances_})<br>
print("Feature Importance:")<br>
print(feature_importance)
