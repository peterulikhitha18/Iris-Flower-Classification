import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("../models/model.pkl")

# Species names
species_names = ["Setosa", "Versicolor", "Virginica"]

# Page Configuration
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# Title
st.title("🌸 Iris Flower Classification")

st.markdown("""
This application predicts the **species of an Iris flower**
using a **Machine Learning K-Nearest Neighbors (KNN)** model.

Enter the flower measurements below and click **Predict**.
""")

st.divider()

# Sidebar
st.sidebar.header("About Project")
st.sidebar.write("""
**Algorithm:** K-Nearest Neighbors (KNN)

**Dataset:** Iris Dataset

**Features Used:**
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
""")

# User Inputs
sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1,
    step=0.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5,
    step=0.1
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4,
    step=0.1
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2,
    step=0.1
)

st.divider()

# Prediction Button
if st.button("Predict Species"):

    # Create DataFrame
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    # Prediction Probability
    probability = model.predict_proba(input_data)

    confidence = probability.max() * 100

    st.success(f"🌼 Predicted Species: **{prediction}**")

    st.info(f"Confidence: **{confidence:.2f}%**")

    st.subheader("Input Summary")

    st.write(f"Sepal Length : {sepal_length} cm")
    st.write(f"Sepal Width : {sepal_width} cm")
    st.write(f"Petal Length : {petal_length} cm")
    st.write(f"Petal Width : {petal_width} cm")

    st.subheader("Prediction Probability")

    probability_df = pd.DataFrame(
        probability,
        columns=species_names
    )

    st.bar_chart(probability_df)

st.divider()

st.caption("Developed using Python, Scikit-learn and Streamlit")

st.success(f"🌼 Predicted Species: **{prediction}**")

# Display flower image

if prediction == "Setosa":
    st.image("../images/setosa.jpg", caption="Iris Setosa")

elif prediction == "Versicolor":
    st.image("../images/versicolor.jpg", caption="Iris Versicolor")

else:
    st.image("../images/virginica.jpg", caption="Iris Virginica")