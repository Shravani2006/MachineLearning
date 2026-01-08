import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

student_data = pd.read_csv('Student_Marks.csv')
student_data.head()
student_data.describe()
student_data.columns

X = student_data[['number_courses', 'time_study']]
y = student_data['Marks']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)

# Models

dt = DecisionTreeRegressor()
rf = RandomForestRegressor()

# Train
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predict
p2 = dt.predict(X_test)
p3 = rf.predict(X_test)
# print(p1[:5])
# print(p2[:5])
# print(p3[:5])

# Accuracy
print("Decision Tree accuracy:", r2_score(y_test, p2))
print("Random Forest accuracy:", r2_score(y_test, p3))

# New Prediction example
new = pd.DataFrame([[3, 4.274]], columns=['number_courses', 'time_study'])
print("Predicted Marks (DT):", dt.predict(new))
print("Predicted Marks (RF):", rf.predict(new))