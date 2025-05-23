import streamlit as st
import joblib
import numpy as np

# Load models and encoders
clf = joblib.load('model/random_forest_model.joblib')
skill_mlb = joblib.load('model/skill_mlb.joblib')
cert_mlb = joblib.load('model/cert_mlb.joblib')
industry_mlb = joblib.load('model/industry_mlb.joblib')
edu_le = joblib.load('model/edu_le.joblib')
target_le = joblib.load('model/target_le.joblib')

st.set_page_config(page_title="Career Recommendation", page_icon="🎯")
st.title("🎯 AI-Powered Career Recommendation System")
st.markdown("Provide your skills, certifications, industry interests, education level, and experience to get tailored career suggestions.")

st.divider()

# Input fields
input_skills = st.multiselect("🛠️ Select your skills", skill_mlb.classes_)
input_certs = st.multiselect("📜 Select your certifications", cert_mlb.classes_)
input_industries = st.multiselect("🏢 Select preferred industries", industry_mlb.classes_)
education = st.selectbox("🎓 Highest education level", edu_le.classes_)
experience = st.slider("🕒 Years of Experience", 0, 40, 0)

# Predict button
if st.button("🚀 Get Recommendations"):
    # Encode input features
    skills_encoded = skill_mlb.transform([input_skills]) if input_skills else np.zeros((1, len(skill_mlb.classes_)))
    certs_encoded = cert_mlb.transform([input_certs]) if input_certs else np.zeros((1, len(cert_mlb.classes_)))
    industries_encoded = industry_mlb.transform([input_industries]) if input_industries else np.zeros((1, len(industry_mlb.classes_)))
    edu_encoded = edu_le.transform([education]) if education else np.array([0])
    exp_array = np.array([[experience]])

    # Combine all features
    X_input = np.hstack([
        skills_encoded,
        certs_encoded,
        industries_encoded,
        edu_encoded.reshape(-1, 1),
        exp_array
    ])

    # Predict probabilities
    probs = clf.predict_proba(X_input)[0]
    raw_careers = [(target_le.classes_[i], probs[i]) for i in range(len(probs))]

    # Optional Boosting based on relevant keywords
    boost_map = {
        "Data Scientist": ["ML", "Python", "SQL"],
        "Digital Marketer": ["SEO", "Google Ads"],
        "Academic Researcher": ["Teaching Certificate", "Education"],
        "Embedded Engineer": ["Microcontroller", "C", "Electronics"]
        # Add more roles and keywords as needed
    }

    boosted = []
    for career, prob in raw_careers:
        keywords = boost_map.get(career, [])
        matches = len(set(input_skills + input_certs) & set(keywords))
        boost = 0.05 * matches
        boosted.append((career, prob + boost))

    # Sort and pick top 3
    top = sorted(boosted, key=lambda x: x[1], reverse=True)[:3]

    st.markdown("## 🔝 Top Career Recommendations")
    for i, (career, score) in enumerate(top, 1):
        icon = "🎯" if i == 1 else ("📚" if "Research" in career else "📈" if "Marketer" in career else "🔧")
        st.markdown(f"""
        **{icon} {career} — Confidence: {round(score * 100, 2)}%**  
        _Based on:_  
        • **Skills:** {', '.join(input_skills) if input_skills else 'None'}  
        • **Certifications:** {', '.join(input_certs) if input_certs else 'None'}  
        • **Experience:** {experience} year(s)  
        • **Industries:** {', '.join(input_industries) if input_industries else 'None'}  
        • **Education:** {education}
        """)
