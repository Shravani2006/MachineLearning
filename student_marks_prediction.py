import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import streamlit as st

st.title("Student Marks Prediction")
st.subheader("Predicting student marks based on number of courses and time studied")
st.set_page_config( page_title="Student Marks Prediction", page_icon="📊", layout="wide" ) 

student_data = pd.read_csv('Student_Marks.csv')
# student_data.head()
# student_data.describe()
# student_data.columns

X = student_data[['number_courses', 'time_study']]
y = student_data['Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)

dt = DecisionTreeRegressor()

dt.fit(X_train, y_train)

p2 = dt.predict(X_test)

# print("Decision Tree accuracy:", r2_score(y_test, p2))

st.slider("Number of Courses", min_value=1, max_value=10, value=3, step=1, key="num_courses")
st.slider("Time Studied (hours)", min_value=0.0, max_value=10.0, value=4.0, step=0.01, key="time_study") 
if st.button("Predict Marks"):
    num_courses = st.session_state.num_courses
    time_study = st.session_state.time_study
    input_data = pd.DataFrame([[num_courses, time_study]], columns=['number_courses', 'time_study'])
    predicted_marks = dt.predict(input_data)
    st.success(f"Predicted Marks: {predicted_marks[0]:.2f}")    



# new = pd.DataFrame([[3, 4.274]], columns=['number_courses', 'time_study'])
# print("Predicted Marks (DT):", dt.predict(new))