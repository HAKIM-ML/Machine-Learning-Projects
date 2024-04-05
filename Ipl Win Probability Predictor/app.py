import pickle
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Fetch the model from GitHub
model_url = "https://github.com/AH-ML/IPL-Win-Probability-Predictor/raw/main/pipe.pkl"  # Updated model link
response = requests.get(model_url)
model_file = BytesIO(response.content)
pipe = pickle.load(model_file)

# Define teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 'Jaipur', 'Chennai',
          'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur',
          'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

@st.cache
def predict_probability(batting_team, bowling_team, selected_city, target, score, overs, wickets):
    runs_left = target - score
    balls_left = (20 - overs) * 6
    wickets_left = 10 - wickets
    crr = score / overs
    rrr = runs_left / balls_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                             'runs_lef': [runs_left], 'ball_left': [balls_left], 'wickets': [wickets_left],
                             'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    # Ensure that categorical variables are encoded properly
    input_df['batting_team'] = input_df['batting_team'].astype('category')
    input_df['bowling_team'] = input_df['bowling_team'].astype('category')
    input_df['city'] = input_df['city'].astype('category')

    # Transform categorical variables using the pipeline
    try:
        input_transformed = pipe.named_steps['preprocessor'].transform(input_df)
    except KeyError:
        # If the preprocessing step is not found, assume no preprocessing is required
        input_transformed = input_df

    # Predict probabilities
    result = pipe.predict_proba(input_transformed)
    
    return result[0][0], result[0][1]

# Function to create cricket-style visualization
def create_cricket_style_visualization(score, target, overs, wickets):
    labels = ['Runs Scored', 'Runs Required', 'Overs Completed', 'Wickets Lost']
    values = [score, target, overs, wickets]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1E90FF', '#FF6347', '#32CD32', '#FFD700']  # Stylish colors

    ax.barh(labels, values, color=colors)

    ax.set_xlabel('Match Metrics', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.set_title('Match Situation', fontsize=16, fontweight='bold')

    # Add text labels with values
    for i, (label, value) in enumerate(zip(labels, values)):
        ax.text(value, i, str(value), ha='left', va='center', fontsize=12, color='black', fontweight='bold')

    plt.tight_layout()

    return fig

# UI
st.title('IPL Win Predictor')

# Input fields
batting_team = st.selectbox('Select the batting team', sorted(teams))
bowling_team = st.selectbox('Select the bowling team', sorted(teams))
selected_city = st.selectbox('Select host city', sorted(cities))
target = st.number_input('Target')
score = st.number_input('Score')
overs = st.number_input('Overs completed')
wickets = st.number_input('Wickets out')

# Button to predict and display result
if st.button('Predict Probability'):
    loss, win = predict_probability(batting_team, bowling_team, selected_city, target, score, overs, wickets)
    st.header(f"{batting_team} - {round(win * 100)}%")
    st.header(f"{bowling_team} - {round(loss * 100)}%")

    # Create cricket-style visualization and display
    fig = create_cricket_style_visualization(score, target, overs, wickets)
    st.pyplot(fig)
