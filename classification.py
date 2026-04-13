import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Function to load the Iris dataset
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

# Load the data and target names
df, target_names = load_data()

# Initialize and train the Random Forest model
model = RandomForestClassifier()
# Using iloc[:, :-1] to select all features and 'species' for the label
model.fit(df.iloc[:, :-1], df['species'])

# Sidebar configuration for user input features
st.sidebar.title("Input Features")

sepal_length = st.sidebar.slider("Sepal length", 
                                 float(df['sepal length (cm)'].min()), 
                                 float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal width", 
                                float(df['sepal width (cm)'].min()), 
                                float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal length", 
                                 float(df['petal length (cm)'].min()), 
                                 float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal width", 
                                float(df['petal width (cm)'].min()), 
                                float(df['petal width (cm)'].max()))

# Format the input for the model
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Prediction logic
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]

# Display the results on the main page
st.write("# Prediction")
st.write(f"The predicted species is: **{predicted_species}**")