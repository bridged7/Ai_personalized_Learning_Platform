# Personalized Learning Platform – Flask API

This repository contains the Flask API for the **AI-powered Personalized Learning Platform**. The platform provides personalized course recommendations based on students' previous learning experiences and progress.

## Key Features

- **Hybrid Recommendation Model**: Combines **Collaborative Filtering** and **Content-Based Filtering** for personalized course suggestions.
- **Personalized Recommendations**: Offers course recommendations tailored to students’ learning history, ensuring they get relevant and engaging content.
- **Adapting to Learning Pace**: The system adjusts to the student’s learning pace, balancing course difficulty and challenge.
- **Seamless Integration**: Integrated with a Java backend, this API serves recommendations to the frontend of the platform.

## Model Workflow

1. **Collaborative Filtering**: Recommends courses based on patterns in student interactions and preferences from similar users.
2. **Content-Based Filtering**: Suggests courses that align with a student’s prior coursework and interests, maintaining relevance to their learning path.
3. **Hybrid Approach**: Combining both models, the system provides the most relevant recommendations by considering both past behavior and content relevance.

## Tools and Technologies

- **Flask**: Used to create the API that delivers course recommendations.
- **Python**: Developed the recommendation algorithms and machine learning model.
- **Postman**: Used for testing API endpoints.
- **AI Model**: The trained recommendation model is saved in a `.pkl` file, ready for deployment.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/bridged7/Ai_personalized_learning_platform.git
