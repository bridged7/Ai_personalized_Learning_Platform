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
        response = requests.get(
            "https://bfaa-41-90-101-26.ngrok-free.app/api/progress/all-ratings",
            headers={'ngrok-skip-browser-warning': 'true'}
        )

        data = response.json()

        processed_data = []
        for entry in data:
            processed_data.append({
                "student_id": entry.get("id"),
                "course_id": entry["courses"].get("courseId"),
                "course_name": entry["courses"].get("name"),
                "enrollment_status": entry.get("enrollment_status"),
                "rating": entry["courses"].get("rating") or 0
            })

        df = pd.DataFrame(processed_data)
        df['student_id'] = df['student_id'].astype(str)
        print(df)
        return df

    except Exception as e:
        return pd.DataFrame()

data = {
    'course_id': ['c1', 'c2', 'c3', 'c4', 'c5'],
    'course_name': [
        'Business Communication',
        'Strategic Management',
        'Organizational Behaviour',
        'Corporate Governance and Ethics',
        'Professional Communication & Negotiation'
    ],
    'description1': [
        'this course covers the fundamentals of effective workplace communication including written verbal and non-verbal skills...',
        'this course explores business strategy formulation and implementation helping students develop analytical and decision-making skills...',
        'this course examines workplace dynamics focusing on motivation teamwork and leadership...',
        'this course delves into ethical decisionmaking in corporate settings...',
        'this course teaches students advanced communication skills for business settings...'
    ]
}

metadata1_df = pd.DataFrame(data)

interactions_df = fetch_interactions()

if not interactions_df.empty:
    unique_students = interactions_df['student_id'].unique()
    student_id_mapping = {id_: id_ for id_ in unique_students}
    user_item_matrix = interactions_df.pivot(index='student_id', columns='course_id', values='rating')
else:
    student_id_mapping = {}
    user_item_matrix = pd.DataFrame()

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(metadata1_df['description1'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=metadata1_df['course_id'], columns=metadata1_df['course_id'])

def get_course_name(course_id):
    course = metadata1_df[metadata1_df['course_id'] == course_id]
    if not course.empty:
        return course.iloc[0]['course_name']
    return "Unknown Course"

def get_taken_courses(student_id):
    student_id = str(student_id)
    
    if interactions_df.empty or student_id not in interactions_df['student_id'].values:
        print(f"Student {student_id} not found in interactions data")
        return set()
    
    taken = interactions_df[interactions_df['student_id'] == student_id]['course_id'].tolist()
    print(f"Student {student_id} has taken courses: {taken}")
    return set(taken)

def recommend_content_based(course_id, taken_courses, n=3):
    if course_id not in cosine_sim_df.index:
        return []
    
    sim_scores = cosine_sim_df[course_id].sort_values(ascending=False)
    
    recommendations = []
    for rec_course in sim_scores.index:
        if rec_course != course_id and rec_course not in taken_courses:
            recommendations.append(rec_course)
            if len(recommendations) >= n:
                break
                
    return recommendations

def get_popular_courses(exclude_courses=None, n=3):
    if exclude_courses is None:
        exclude_courses = set()
        
    if interactions_df.empty:
        all_courses = metadata1_df['course_id'].tolist()
        return [c for c in all_courses if c not in exclude_courses][:n]
    
    course_counts = interactions_df['course_id'].value_counts()
    
    popular_courses = [course for course in course_counts.index if course not in exclude_courses]
    
    return popular_courses[:n]

def hybrid_recommend(student_id, n=3):
    student_id = str(student_id)
    print(f"Generating recommendations for student_id: {student_id}")
    
    taken_courses = get_taken_courses(student_id)
    
    all_courses = set(metadata1_df['course_id'])
    available_courses = all_courses - taken_courses
    
    if not available_courses:
        print("Student has taken all available courses")
        return []
    
    student_history = interactions_df[interactions_df['student_id'] == student_id]
    if student_history.empty:
        print(f"No history for student {student_id}, using popularity-based recommendations")
        return get_popular_courses(exclude_courses=taken_courses, n=n)
    
    recommendations = []
    
    try:
        cf_recommendations = []
        for course_id in available_courses:
            try:
                est_rating = model2.predict(student_id, course_id).est
                cf_recommendations.append((course_id, est_rating))
            except Exception as e:
                print(f"Error predicting rating for {student_id}, {course_id}: {e}")
                continue
        
        if cf_recommendations:
            cf_recommendations.sort(key=lambda x: x[1], reverse=True)
            for course, _ in cf_recommendations[:n]:
                if course not in taken_courses:  # Double-check
                    recommendations.append(course)
    except Exception as e:
        print(f"Error in collaborative filtering: {e}")
    
    if len(recommendations) < n and not student_history.empty:
        try:
            last_course = student_history.iloc[-1]['course_id']
            
            cb_recommendations = recommend_content_based(
                last_course, 
                taken_courses, 
                n=n-len(recommendations)
            )
            
            for course in cb_recommendations:
                if course not in recommendations and course not in taken_courses:
                    recommendations.append(course)
                if len(recommendations) >= n:
                    break
        except Exception as e:
            print(f"Error in content-based recommendation: {e}")
    
    if len(recommendations) < n:
        exclude = taken_courses.union(set(recommendations))
        popular = get_popular_courses(exclude_courses=exclude, n=n-len(recommendations))
        recommendations.extend(popular)
    
    return recommendations[:n]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'student_id' not in data:
            return jsonify({'error': 'Missing student_id'}), 400

        student_id = str(data["student_id"]) 
        
        recommended_courses = hybrid_recommend(student_id, n=3)
        
        course_details = []
        for course_id in recommended_courses:
            course_details.append({
                'course_id': course_id,
                'course_name': get_course_name(course_id)
            })
        
        return jsonify({
            'student_id': student_id, 
            'recommended_courses': recommended_courses,
            'course_details': course_details
        })
    
    except Exception as e:
        print(f"Error in prediction endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is live!", "courses_available": metadata1_df['course_id'].tolist()})

@app.route('/debug', methods=['GET'])
def debug():
    
    return jsonify({
        "students": student_id_mapping,
        "interactions_count": len(interactions_df) if not interactions_df.empty else 0,
        "taken_courses": {
            str(student): list(get_taken_courses(student)) 
            for student in student_id_mapping.keys()
        } if student_id_mapping else {}
    })

if __name__ == "__main__":
    app.run(debug=True)