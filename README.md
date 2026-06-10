##Title
CineGraph-Movie Recommendation System using LightGCN and Streamlit.


##Description
CineGraph is an intelligent movie recommendation platform that utilizes Graph Neural Networks (GNNs), specifically the LightGCN (Light Graph Convolutional Network) architecture, to provide personalized movie recommendations. The system is built using the MovieLens 100K dataset and offers users an interactive interface to explore movies, discover similar content, analyze viewing history, and identify trending titles.

Unlike traditional recommendation systems that rely solely on user ratings or content similarity, CineGraph models the relationships between users and movies as a graph structure. By learning user and movie embeddings through graph-based message passing, the system captures complex interaction patterns and generates more accurate and personalized recommendations.

##Key Features

##Personalized Recommendations
Generates movie recommendations tailored to individual users.
Uses LightGCN to learn latent user preferences from historical rating data.
Supports filtering by genre, rating thresholds, and recommendation count.

##Movie Search & Discovery
Allows users to search for movies from the dataset.
Finds similar movies using learned graph embeddings.
Helps users discover new content related to their interests.

##Trending & Top Rated Movies
Displays the most popular movies based on community engagement.
Highlights highly rated movies with strong audience feedback.
Provides insights into movie popularity and quality.
User Watch History Analytics
Visualizes movies rated by a selected user.
Displays genre distribution and viewing preferences.
Helps understand user behavior through interactive charts and statistics.

##Interactive Dashboard
Built with Streamlit for an intuitive and responsive user experience.
Provides real-time interaction with recommendation filters and analytics.

## Live Demo
https://cinegraph-i6vyx3ohuovunpddsy9zx2.streamlit.app

## Features
- Personalized movie recommendations
- Movie search
- Trending movies
- User history

## Tech Stack
- Python
- PyTorch
- LightGCN
- Streamlit
