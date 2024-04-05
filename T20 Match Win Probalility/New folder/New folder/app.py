import pickle

import pandas as pd
import requests  # Add this import for handling HTTP requests
import streamlit as st

teams = ['South Africa', 'India', 'New Zealand', 'West Indies', 'Australia', 'Sri Lanka', 'England', 'Bangladesh', 'Pakistan']

cities = ['Sharjah Cricket Stadium', 'Saxton Oval', 'Hyderabad', 'Sydney Cricket Ground', 'Abu Dhabi', 'Harare Sports Club', 'Nagpur', 'The Rose Bowl', 'Visakhapatnam', 'Headingley', 'Premadasa Stadium', 'Dubai', 'Dharamsala', 'New Wanderers Stadium', 'Queen Park Oval', 'Kennington Oval', 'University Oval', 'Kimberley', 'Ahmedabad', 'Ranchi', 'Malahide', 'Senwes Park', 'Delhi', 'Shere Bangla', 'Boland Park', 'Bay Oval']

# Load the model from GitHub
model_url = 'https://raw.githubusercontent.com/DreamIsMl/T20-Match-Win-Probability/master/model.pkl'
response = requests.get(model_url)
pipe = pickle.loads(response.content)

st.title("🏏 Hakim's T20 Match Oracle: Win Odds Unveiled 🌟")
st.markdown(
    """
    Hi there! I'm Md. Azizul Hakim, a student at BSPI on CST. I'm passionate about Machine Learning and AI,
    actively working in the field alongside my studies. Currently, I'm diving deep into deep learning and 
    enjoying participating in Kaggle competitions to enhance my skills.
    """
)

# Links to Kaggle and GitHub Profiles
st.markdown(
    """
    **Kaggle Profile:** [Md. Azizul Hakim](https://www.kaggle.com/me) | 
    **GitHub Profile:** [DreamIsMl](https://github.com/DreamIsMl)
    """
)

# Input Section
st.header("Let's Predict T20 Match Win Probability!")
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Scores Of Batting team')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    ball_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / ball_left

    input_df = pd.DataFrame(
        {'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city], 'runs_lef': [runs_left],
         'ball_left': [ball_left], 'wickets': [wickets], 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win * 100)) + "%")
    st.header(bowling_team + "- " + str(round(loss * 100)) + "%")
