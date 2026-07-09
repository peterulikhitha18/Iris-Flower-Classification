import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
IMAGE_PATH = BASE_DIR / "images"

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
model = joblib.load(MODEL_PATH)

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Classification")

st.write("""
Predict the species of an Iris flower using a trained
K-Nearest Neighbors (KNN) Machine Learning model.
""")

st.divider()

# Sidebar
st.sidebar.header("Project Information")

st.sidebar.write("""
**Algorithm:** K-Nearest Neighbors (KNN)

**Dataset:** Iris Dataset

**Features:**
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
""")

# ---------------------------------------------------
# Inputs
# ---------------------------------------------------

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

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button("Predict Species"):

    input_df = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    # Debug output
    st.write("Raw Prediction:", prediction)

    # Convert prediction to species name
    if prediction == 0 or str(prediction).lower() == "setosa":
        species = "Setosa"
        image = IMAGE_PATH / "setosa.jpg"

    elif prediction == 1 or str(prediction).lower() == "versicolor":
        species = "Versicolor"
        image = IMAGE_PATH / "versicolor.jpg"

    elif prediction == 2 or str(prediction).lower() == "virginica":
        species = "Virginica"
        image = IMAGE_PATH / "virginica.jpg"

    else:
        species = str(prediction)
        image = None

    confidence = max(probabilities) * 100

    st.success(f"Predicted Species: {species}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Prediction Probabilities")

    probability_df = pd.DataFrame(
        [probabilities],
        columns=["Setosa", "Versicolor", "Virginica"]
    )

    st.bar_chart(probability_df)

    st.subheader("Flower Image")

    if image is not None:
        st.image(image, width=350)

    st.subheader("Input Values")

    st.write(input_df)

st.divider()

st.caption("Developed using Python, Scikit-learn and Streamlit")