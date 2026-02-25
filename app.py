import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Models and Data
# -----------------------------

regression_model = joblib.load(r"Models\regression_model.pkl")
classification_model = joblib.load("Models\classification_model.pkl")

encoders = joblib.load("Models\encoders.pkl")

user_item_matrix = joblib.load(r"Recomendations\user_item_matrix.pkl")
item_similarity_df = joblib.load(r"Recomendations\item_similarity.pkl")
content_similarity_df = joblib.load(r"Recomendations\content_similarity.pkl")
attraction_lookup = joblib.load(r"Recomendations\attraction_lookup.pkl")

master = pd.read_csv(r"Data\master_tourism_dataset.csv")


# -----------------------------
# Title
# -----------------------------
st.title("Tourism Experience Analytics System")
st.markdown("Predict Rating, Visit Mode & Get Recommendations")

# ----------------------------------
# Sidebar Inputs
# ----------------------------------

st.sidebar.header("User Input")

continent = st.sidebar.selectbox(
    "Select Continent",
    sorted(master['Continent'].unique())
)

region = st.sidebar.selectbox(
    "Select Region",
    sorted(master['Region'].unique())
)

country = st.sidebar.selectbox(
    "Select Country",
    sorted(master['Country'].unique())
)

city = st.sidebar.selectbox(
    "Select City",
    sorted(master['CityName'].unique())
)

attraction_type = st.sidebar.selectbox(
    "Preferred Attraction Type",
    sorted(master['AttractionType'].unique())
)

visit_month = st.sidebar.slider("Visit Month", 1, 12, 7)

visit_year = st.sidebar.slider(
    "Visit Year",
    int(master['VisitYear'].min()),
    int(master['VisitYear'].max()),
    int(master['VisitYear'].mean())
)

# ----------------------------------
# Encoding Function
# ----------------------------------

def encode_value(column, value):
    if column in encoders:
        le = encoders[column]
        if value in le.classes_:
            return le.transform([value])[0]
        else:
            return 0
    return value

# ----------------------------------
# Prepare Input DataFrame
# ----------------------------------

input_dict = {
    'VisitYear': visit_year,
    'VisitMonth': visit_month,
    'VisitMode': 0,
    'Continent': encode_value('Continent', continent),
    'Region': encode_value('Region', region),
    'Country': encode_value('Country', country),
    'CityName': encode_value('CityName', city),
    'AttractionType': encode_value('AttractionType', attraction_type),
    'TotalVisits': 1,
    'UserAvgRating': master['UserAvgRating'].mean(),
    'AttractionPopularity': master['AttractionPopularity'].mean(),
    'AttractionAvgRating': master['AttractionAvgRating'].mean()
}

input_df = pd.DataFrame([input_dict])

# Ensure correct column order based on regression model
feature_order = regression_model.feature_names_in_
input_df = input_df[feature_order]

# ----------------------------------
# Prediction Section
# ----------------------------------

if st.button("Predict"):

    # Rating Prediction
    predicted_rating = regression_model.predict(input_df)[0]
    st.subheader("Predicted Rating")
    st.success(round(predicted_rating, 2))

    # Visit Mode Prediction
    clf_input = input_df.drop(columns=['VisitMode'])
    predicted_mode_encoded = classification_model.predict(clf_input)[0]

    visit_mode_decoder = encoders.get("VisitMode")

    if visit_mode_decoder:
        predicted_mode = visit_mode_decoder.inverse_transform([predicted_mode_encoded])[0]
    else:
        predicted_mode = predicted_mode_encoded

    st.subheader("Predicted Visit Mode")
    st.info(predicted_mode)

# ----------------------------------
# Recommendation Section (Item-Based CF)
# ----------------------------------

st.header("Recommended Attractions")

def item_based_recommendation(user_id, top_n=5):

    matrix = user_item_matrix.fillna(0)
    user_ratings = matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0]

    scores = {}

    for item_id, rating in rated_items.items():
        similar_items = item_similarity_df[item_id]

        for sim_item, similarity in similar_items.items():
            if user_ratings[sim_item] == 0:
                scores[sim_item] = scores.get(sim_item, 0) + similarity * rating

    rec_df = pd.DataFrame(list(scores.items()), columns=['AttractionId', 'Score'])
    rec_df = rec_df.merge(attraction_lookup, on='AttractionId', how='left')

    return rec_df.sort_values(by='Score', ascending=False).head(top_n)

# Demo: Use first user
sample_user = master['UserId'].iloc[0]

recommendations = item_based_recommendation(sample_user)

st.table(recommendations[['Attraction', 'Score']])

# ----------------------------------
# Analytics Section
# ----------------------------------

st.header("Tourism Insights")

st.subheader("Top 5 Attractions")
top_attractions = master['Attraction'].value_counts().head(5)
st.bar_chart(top_attractions)

st.subheader("Visit Mode Distribution")
visit_mode_counts = master['VisitMode'].value_counts()
st.bar_chart(visit_mode_counts)
