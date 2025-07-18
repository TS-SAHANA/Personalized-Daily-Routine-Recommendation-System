# Imports necessary modules from Flask for web application functionality.
from flask import Flask, request, render_template

# Imports the pandas library for data manipulation.
import pandas as pd

# Imports the TF-IDF vectorizer from scikit-learn to convert text data into numerical format.
from sklearn.feature_extraction.text import TfidfVectorizer

# Imports cosine similarity function to measure the similarity between vectors.
from sklearn.metrics.pairwise import cosine_similarity

# Imports the os module for file path operations.
import os

# Initializes a Flask web application instance.
app = Flask(__name__)

# Defines a function to load activity data from a CSV file.
def load_data():
    # Constructs the path to the 'activities.csv' file located in the same directory as the script.
    data_path = os.path.join(os.path.dirname(__file__), 'activities.csv')
    # Reads and returns the CSV file as a DataFrame.
    return pd.read_csv(data_path)

# Calls load_data() to load the activity data into the activities variable.
activities = load_data()

# Creates a TF-IDF vectorizer instance.
vectorizer = TfidfVectorizer()

# Converts the 'description' column of activities into TF-IDF matrix.
activity_matrix = vectorizer.fit_transform(activities['description'])

# Defines a function to recommend activities based on user preferences.
def recommend_activities(user_preferences, top_n=3):
    # Transforms user preferences into TF-IDF vector.
    user_vector = vectorizer.transform([user_preferences])
    # Computes cosine similarities between user preferences and activity descriptions.
    similarities = cosine_similarity(user_vector, activity_matrix).flatten()
    # Gets indices of the top N most similar activities.
    top_indices = similarities.argsort()[-top_n:][::-1]
    # Retrieves the top N recommended activities.
    recommended_activities = activities.iloc[top_indices]
    # Returns the recommended activities.
    return recommended_activities

# Defines the route for the index page.
@app.route('/')

# Defines the function to handle requests to the index page.
def index():
    # Renders and returns the 'index.html' template.
    return render_template('index.html')

# Defines the route for the recommendation page with POST method.
@app.route('/recommend', methods=['POST'])

# Defines the function to handle requests to the recommendation page.
def recommend():
    # Gets user preferences from the form data.
    user_preferences = request.form['preferences']
    # Calls recommend_activities() to get recommendations based on user preferences.
    recommendations = recommend_activities(user_preferences)
    # Renders and returns the 'results.html' template with the recommendations.
    return render_template('results.html', recommendations=recommendations)

# Checks if the script is being run directly (not imported).
if __name__ == "__main__":
    # Starts the Flask application in debug mode for development.
    app.run(debug=True)