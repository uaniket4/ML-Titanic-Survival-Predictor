import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

#load the model
model = pickle.load(open("model.pkl", "rb"))

st.title("Titanic Survival Predictor")

#input
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Passenger Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])


#input conversion
sex = 0 if sex == "Male" else 1
embarked = {"S": 0, "C": 1, "Q": 2}[embarked]

#prediction button
if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("This passenger would have SURVIVED!")
    else:
        st.error("This passenger would NOT have survived.")