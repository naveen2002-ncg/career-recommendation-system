# 🎯 AI-Powered Career Recommendation System

Out Put link🔗 : https://career-recommendation-system-pplxjcrg4xeg26stwbysij.streamlit.app/

## 🌟 Overview
This project recommends suitable careers based on user-provided **skills**, **certifications**, **industry interests**, **education level**, and **work experience**. Built with machine learning, the system delivers **personalized** and **intelligent career suggestions**.

---

## 🚀 Features
- 🛠️ Multiselect inputs for skills, certifications, and industries  
- 🎓 Education and experience profiling  
- 📈 Smart recommendations using a trained ML model  
- ⚙️ Boosting logic for more accurate predictions  
- 📊 Career confidence score with visual explanation  
- 🔒 Optional Firebase integration for login and history (future scope)  

---

## 🧠 Tech Stack

| Layer     | Technologies Used                               |
|-----------|--------------------------------------------------|
| Frontend  | Streamlit (Python-based UI framework)           |
| Backend   | Python, Scikit-learn, Pandas, NumPy             |
| ML Model  | Random Forest Classifier                        |
| Dataset   | Custom CSV (structured with career labels)      |
| Deployment| Streamlit Community Cloud                       |

---

## 📂 Folder Structure

career-recommendation-system/
┣ 📁 model/
┃ ┣ random_forest_model.joblib
┃ ┣ skill_mlb.joblib
┃ ┣ cert_mlb.joblib
┃ ┣ industry_mlb.joblib
┃ ┣ edu_le.joblib
┃ ┗ target_le.joblib
┣ 📄 app.py
┣ 📄 train_model.py
┣ 📄 careers.csv
┣ 📄 requirements.txt
┗ 📄 README.md


## 🛠️ Installation & Usage

### 1. Clone the repository
git clone https://github.com/your-username/career-recommendation-system.git
cd career-recommendation-system

2. Install dependencies
Make sure Python 3.7+ is installed. Then run:
pip install -r requirements.txt

3. Train the model
python train_model.py

4. Run the Streamlit app
streamlit run app.py

🧪 Sample Inputs
Skills: Python, ML, Data Analysis

Certifications: Power BI, SQL

Preferred Industries: IT

Education: Bachelors

Experience: 2 years


📊 Output Example
 
🎯 Data Scientist — Confidence: 105.0%
📚 Academic Researcher — Confidence: 3.0%
🔧 Graphic Designer — Confidence: 3.0%
💡 Future Enhancements
Firebase login and user history tracking

More career roles and detailed feedback

Integration with resume analysis

Better UI styling and animations

👤 Author
Naveen C Gundapalli
Computer Science Student
Jain College of Engineering and Technology, Hubballi

 
---
`https://github.com/your-username/career-recommendation-system.git` with your actual GitHub repository URL. Let me know if you'd like badges (e.g., license, deployment, built with) or a demo GIF added to the README.
