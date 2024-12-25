# Import necessary libraries
import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objects as go

# Path to saved models
MODEL_SAVE_PATH = './saved_models/'  # Ensure this directory exists
MODEL_FILES = ["Random_Forest.pkl", "XGBoost.pkl", "Gradient_Boosting.pkl"]

# Initialize Age Group Encoder
BINS = [0, 20, 25, 30, 35, 40, np.inf]
LABELS = ['<20', '20-24', '25-29', '30-34', '35-39', '40+']
encoder = OneHotEncoder(sparse_output=False)
encoder.fit([[label] for label in LABELS])

# Depression questions
ANTENATAL_QUESTIONS = [
    "During your pregnancy, have you experienced persistent sadness or a consistently low mood?",
    "During your pregnancy, have you frequently felt anxious or excessively worried?",
    "Have you lost interest in activities that you previously enjoyed while pregnant?",
    "Have you experienced significant changes in your sleep patterns during pregnancy, such as insomnia or excessive sleeping?",
    "During your pregnancy, have you noticed any changes in your appetite, such as eating significantly more or less than usual?",
    "Have you experienced feelings of guilt or worthlessness during your pregnancy?",
    "During your pregnancy, have you found it difficult to concentrate or stay focused?",
    "Have you had thoughts of self-harm or suicide while pregnant?",
    "During your pregnancy, have you experienced extreme fatigue that doesn't improve with rest?",
    "Have you experienced persistent headaches, stomachaches, or other unexplained aches and pains during your pregnancy?"
]

POSTPARTUM_QUESTIONS = [
    "Since giving birth, have you experienced persistent sadness, anxiety, or a sense of emptiness?",
    "Have you noticed severe mood swings since giving birth?",
    "Have you experienced episodes of excessive crying since giving birth?",
    "Have you found it difficult to bond with your baby since giving birth?",
    "Since giving birth, how often have you withdrawn from family and friends?",
    "Have you experienced noticeable changes in your appetite, such as a significant loss of appetite or overeating?",
    "Since giving birth, have you experienced insomnia or found yourself sleeping excessively?",
    "How often have you felt overwhelming fatigue or a significant loss of energy since giving birth?",
    "Have you lost interest in activities that you once enjoyed since giving birth?",
    "Since giving birth, have you experienced intense irritability or anger?",
    "How often have you felt feelings of worthlessness, shame, guilt, or inadequacy since giving birth?",
    "Since giving birth, have you experienced difficulty thinking clearly, concentrating, or making decisions?",
    "How often have you experienced severe anxiety or panic attacks since giving birth?",
    "Have you had thoughts of harming yourself or your baby since giving birth?",
    "Have you experienced persistent, unexplained aches and pains in your back, joints, or muscles since giving birth?",
    "Was your childbirth delivered via C-section?"
]


# Functions
def classify_depression(score):
    """Classify depression severity based on score."""
    if score < 33:
        return "Low"
    elif score < 66:
        return "Moderate"
    else:
        return "Severe"

def get_most_voted_result(predictions):
    """Get the most voted result from multiple model predictions."""
    severity_map = {0: "Low", 1: "Moderate", 2: "Severe"}
    votes = np.bincount(predictions)
    most_voted = np.argmax(votes)
    return severity_map[most_voted]

def get_suggestions(severity):
    """Provide suggestions based on severity level."""
    if severity == "Low":
        return [
            "Maintain a healthy lifestyle with balanced meals and regular physical activity.",
            "Engage in mindfulness practices or meditation to manage mild stress.",
            "Stay socially connected with friends and family."
        ]
    elif severity == "Moderate":
        return [
            "Consider consulting a counselor or therapist to discuss your feelings.",
            "Join a support group to connect with others who might share your experiences.",
            "Practice self-care routines and try to get enough rest."
        ]
    else:  # Severe
        return [
            "Seek immediate professional help from a psychiatrist or psychologist.",
            "Contact a healthcare provider for potential medical interventions or therapy.",
            "Ensure a strong support system is in place to assist you during this time."
        ]

def plot_gauge(score, title):
    """Plot gauge meter for severity level."""
    severity_colors = ["green", "orange", "red"]
    severity_labels = ["Low", "Moderate", "Severe"]
    severity_idx = 0 if score < 33 else (1 if score < 66 else 2)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": severity_colors[severity_idx]},
                "steps": [
                    {"range": [0, 33], "color": "lightgreen"},
                    {"range": [33, 66], "color": "yellow"},
                    {"range": [66, 100], "color": "lightcoral"}
                ]
            },
            title={"text": f"{title} ({severity_labels[severity_idx]})"}
        )
    )
    return fig

# Sidebar About Section
with st.sidebar:
    st.subheader("About")
    st.write(
        """
        This Antenatal & Postpartum Depression Assessment Tool is designed to evaluate mood and predict postpartum depression severity.
        It provides insights into mental health and personalized suggestions based on responses to a series of questions.
        Note: The results are for informational purposes only and not a substitute for professional advice.
        """
    )

# Main App
st.title("Antenatal & Postpartum Depression Assessment Tool")

# Input Age (Reduced Width)
col1, _ = st.columns([1, 3])
with col1:
    age = st.number_input("Enter your age", min_value=12, max_value=60, step=1)

if age:
    age_group = pd.cut([age], bins=BINS, labels=LABELS)[0]
    age_group_encoded = encoder.transform([[str(age_group)]])
    
    # Sequential Questions
    pregnant = st.radio("Are you currently pregnant?", options=["Select..", "Yes", "No"])
    if pregnant == "Yes":
        phase = "Antenatal"
        questions = ANTENATAL_QUESTIONS
    elif pregnant == "No":
        recently_given_birth = st.radio("Have you given birth recently?", options=["Select..", "Yes", "No"])
        if recently_given_birth == "Yes":
            phase = "Postpartum"
            questions = POSTPARTUM_QUESTIONS
        elif recently_given_birth == "No":
            st.info("You do not appear to qualify for this assessment.")
            st.stop()
        else:
            st.stop()
    else:
        st.stop()

    # Depression Assessment Questions
    if questions:
        responses = []
        st.subheader(f"{phase} Depression Questions")
        for i, question in enumerate(questions):
            if len(responses) == i:
                answer = st.radio(f"Q{i+1}: {question}", options=["None", "Yes", "No"], key=i)
                if answer != "None":
                    responses.append(1 if answer == "Yes" else 0)

        # Submit Responses
        if len(responses) == len(questions) and st.button("Submit Responses"):
            # Calculate Current Mood
            score = sum(responses) / len(questions) * 100
            severity = classify_depression(score)
            st.plotly_chart(plot_gauge(score, "Current Mood"), use_container_width=True)
            st.success(f"Current Mood Score: {score:.2f}% - Severity Level: {severity}")
            
            # Provide suggestions
            st.subheader("Suggestions")
            suggestions = get_suggestions(severity)
            for suggestion in suggestions:
                st.write(f"- {suggestion}")

            # Prediction for Antenatal
            if phase == "Antenatal":
                input_features = responses + list(age_group_encoded.flatten())
                predictions = []
                for model_file in MODEL_FILES:
                    with open(MODEL_SAVE_PATH + model_file, 'rb') as f:
                        model = pickle.load(f)
                    prediction = model.predict([input_features])[0]
                    predictions.append(prediction)

                st.subheader("Postpartum Depression Prediction")
                for i, pred in enumerate(predictions):
                    severity = classify_depression(pred * 33)  # Normalize to 100-scale
                    model_name = MODEL_FILES[i].replace(".pkl", "")
                    st.write(f"{model_name} Prediction: **{severity}**")

                # Most voted prediction
                most_voted = get_most_voted_result(predictions)
                st.success(f"Most Voted Prediction: {most_voted}")
