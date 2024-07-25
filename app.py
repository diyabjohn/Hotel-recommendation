import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Load and preprocess the dataset
hotel_data = pd.read_csv('hotel.csv')
hotel_data['Title'].fillna('', inplace=True)
hotel_data['Special'].fillna('', inplace=True)
hotel_data['Price'] = hotel_data['Price'].str.replace('₹', '').str.replace(',', '').astype(float)
hotel_data['Amount'] = hotel_data['Amount'].str.replace(',', '').astype(float)
hotel_data['Rating'].fillna(hotel_data['Rating'].mean(), inplace=True)
hotel_data['No_of_ratings'].fillna(hotel_data['No_of_ratings'].mean(), inplace=True)
hotel_data['Discount'] = hotel_data['Discount'].str.replace('%', '').astype(float)
hotel_data['Location'].fillna('Unknown', inplace=True)
hotel_data['Review'].fillna('No Review', inplace=True)
hotel_data['Offer'].fillna('No Offer', inplace=True)

# Create TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
hotel_features = tfidf_vectorizer.fit_transform(hotel_data['Title'] + " " + hotel_data['Special'])

# Compute cosine similarity matrix
cosine_sim_matrix = cosine_similarity(hotel_features, hotel_features)

# Function to get recommendations based on additional filters
def get_recommendations(amount, rating, city, cosine_sim=cosine_sim_matrix):
    filtered_data = hotel_data[
        (hotel_data['Amount'] <= amount) &
        (hotel_data['Rating'] >= rating) &
        (hotel_data['Location'].str.contains(city, case=False, na=False))
    ]

    if filtered_data.empty:
        return pd.DataFrame(columns=['Title', 'Location', 'Price', 'Rating', 'Special'])

    filtered_indices = filtered_data.index
    filtered_sim_matrix = cosine_sim[filtered_indices][:, filtered_indices]

    sim_scores = list(enumerate(filtered_sim_matrix[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    hotel_indices = [filtered_indices[i[0]] for i in sim_scores]
    return hotel_data[['Title', 'Location', 'Price', 'Rating', 'Special']].iloc[hotel_indices]

# Streamlit app
st.title('Hotel Recommendation System')

# User inputs for amount, rating, and city
amount = st.number_input('Enter maximum amount', min_value=0.0)
rating = st.number_input('Enter minimum rating', min_value=0.0, max_value=5.0, step=0.1)
city = st.text_input('Enter city name')

# Button to get recommendations
if st.button('Get Recommendations'):
    if amount and rating and city:
        recommendations = get_recommendations(amount, rating, city)
        if not recommendations.empty:
            st.write('Top Recommended Hotels:')
            st.dataframe(recommendations)
        else:
            st.write('No hotels found for the given criteria.')
    else:
        st.write('Please enter amount, rating, and city.')

