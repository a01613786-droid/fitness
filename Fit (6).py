import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


st.write(''' # Predicción de nivel de fitness de una persona ''')
st.image("fit.jpg", caption="¿Quieres saber si eres una persona fit? Ingresa los siguientes datos: ")

st.header('Datos de la persona')

def user_input_features():
  # Entrada
  age = st.number_input('age:', min_value=0, max_value=200, value = 0, step = 1)
  height_cm = st.number_input('height_cm:', min_value=0, max_value=300, value = 0, step = 1)
  weight_kg = st.number_input('weight_kg:', min_value=0, max_value=300, value = 0, step = 1)
  heart_rate = st.number_input('heart_rate',min_value=0, max_value=150, value = 0, step = 1)
  blood_pressure = st.number_input('blood_pressure', min_value=0, max_value=200, value = 0, step = 1)
  sleep_hours = st.number_input('sleep_hours', min_value=0, max_value=12, value = 0, step = 1)
  nutrition_quality = st.number_input('nutrition_quality', min_value=0, max_value=10, value = 0, step = 1)
  activity_index = st.number_input('activity_index', min_value=0, max_value=5, value = 0, step = 1)
  smokes = st.number_input('smokes', min_value=0, max_value=1, value = 0, step = 1)
  gender = st.number_input('gender', min_value=0, max_value=1, value = 0, step = 1)

  user_input_data = {'age': age,
                     'height_cm': height_cm,
                     'weight_kg': weight_kg,
                     'heart_rate': heart_rate,
                     'blood_pressure': blood_pressure,
                     'sleep_hours': sleep_hours,
                     'nutrition_quality': nutrition_quality,
                     'activity_index': activity_index,
                     'smokes': smokes,
                     'gender': gender}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

df_train =  pd.read_csv('Fitness_Classification2.csv', encoding='latin-1')
X = df_train.drop(columns='is_fit')
Y = df_train['is_fit']

classifier = DecisionTreeClassifier(max_depth=5, criterion='entropy', min_samples_leaf=10, max_features=5, random_state=1613786)
classifier.fit(X, Y)

df = df(X.columns)

pred = classifier.predict(df)
pred_final = pred[0]

st.subheader('Predicción')
if pred_final == 0:
  st.write('No es fit')
elif pred_final == 1:
  st.write('Es fit')
else:
  st.write('Sin predicción')
