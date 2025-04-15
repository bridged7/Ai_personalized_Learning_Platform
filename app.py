from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import requests


app = Flask(__name__)

model2 = joblib.load("recommendation_model.pkl")

def fetch_interactions():
    try:
        url = "https://017f-41-90-101-26.ngrok-free.app/api/progress/all"  
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching data from backend: {e}")
        return pd.DataFrame(columns=["student_id", "course_id", "rating"])


interactions_df = fetch_interactions()


unique_students = interactions_df['student_id'].unique()
student_id_mapping = {id_: i for i, id_ in enumerate(unique_students)}
interactions_df['student_id'] = interactions_df['student_id'].map(student_id_mapping)

user_item_matrix = interactions_df.pivot(index='student_id', columns='course_id', values='rating')

data = {
    'course_id': ['c1', 'c2', 'c3', 'c4','c5'],
    'course_name': [
        'Business Communication',
        'Strategic Management',
        'Organizational Behaviour',
        'Corporate Governance and Ethics',
        'Professional Communication & Negotiation'
    ],
    'description1': [
        'this course covers the fundamentals of effective workplace communication including written verbal and non-verbal skills students will learn how to craft persuasive business messages deliver impactful presentations and negotiate successfully the course also explores professional email etiquette and strategies for conflict resolution in corporate settings',
        'this course explores business strategy formulation and implementation helping students develop analytical and decision-making skills topics include competitive analysis corporate planning leadership strategies and risk assessment students will learn how to create sustainable business models and adapt to market changes',
        'this course examines workplace dynamics focusing on motivation teamwork and leadership students will explore how organizational culture employee engagement and psychological factors influence productivity the course also covers leadership styles group behavior and methods for improving workplace performance',
        'this course delves into ethical decisionmaking in corporate settings students will learn about governance structures corporate social responsibility csr compliance regulations and ethical leadership the course covers case studies on business ethics and the impact of corporate governance on organizational success',
        'this course teaches students advanced communication skills for business settings topics include crafting professional emails persuasive presentations and active listening the course also covers negotiation tactics workplace etiquette and conflict resolution strategies to enhance professional interactions'
    ]
}

metadata1_df = pd.DataFrame(data)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(metadata1_df['description1'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=metadata1_df['course_id'], columns=metadata1_df['course_id'])

def recommend_top_course(course_id, cosine_sim=cosine_sim_df, threshold=0.1):
    if course_id not in cosine_sim.index:
        return None
    sim_scores = cosine_sim[course_id].sort_values(ascending=False)
    sim_scores = sim_scores.drop(course_id)
    filtered_scores = sim_scores[sim_scores > threshold]
    return filtered_scores.idxmax() if not filtered_scores.empty else None

def hybrid_recommend(student_id, model, user_matrix, n=3, weight_cf=0.4, weight_cb=0.6):
    student_history = interactions_df[interactions_df['student_id'] == student_id]
    recommendations = []
    
    
    if not student_history.empty:
        taken_courses = set(student_history['course_id'])
        other_courses = set(user_item_matrix.columns)
        unseen_courses = other_courses - taken_courses
        cf_recommendations = []
        
        for course_id in unseen_courses:
            est_rating = model.predict(student_id, course_id).est
            cf_recommendations.append((course_id, est_rating))
        
        if cf_recommendations:
            cf_recommendations.sort(key=lambda x: x[1], reverse=True)
            cf_recommendations = [course_id for course_id, _ in cf_recommendations[:n]]
            recommendations.extend([(course, weight_cf) for course in cf_recommendations])

    
    if not student_history.empty:
        last_course = student_history.iloc[-1]['course_id']
        content_recommendation = recommend_top_course(last_course)
        if content_recommendation:
            recommendations.append((content_recommendation, weight_cb))

    
    if not recommendations:
        return course_popularity[:n]

    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    final_recommendations = []
    added = set()
    for course, _ in recommendations:
        if course not in added:
            final_recommendations.append(course)
            added.add(course)
        if len(final_recommendations) == n:
            break

    return final_recommendations



@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.json


    if 'student_id' not in data:
        return jsonify({'error': 'Missing student_id'}), 400
    
    student_id = data["student_id"]

    course_popularity = interactions_df['course_id'].value_counts().index.tolist()

    mapped_student_id = student_id_mapping.get(data["student_id"])
    if mapped_student_id is None:
        recommended_courses = course_popularity[:3]
        return jsonify({'student_id': student_id, 'recommended_courses': recommended_courses})
    else:
        
        recommended_courses = hybrid_recommend(mapped_student_id, model2, user_item_matrix)
        return jsonify({'student_id': student_id, 'recommended_courses': recommended_courses})

    

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is live!"})
