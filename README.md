# 🥤 Personalized Carbonated Drink Recommender  

A simple **Streamlit app** that recommends carbonated drinks based on user preferences.  
Built with **FAISS + Sentence Transformers** for semantic search, deployed on **Streamlit Cloud**.  

---

## 📂 Project Structure
├── streamlit_app.py # Main Streamlit app (3 pages: Login, Notebook, Feedback)
├── cleaned_carbonated_drinks.csv # Dataset
├── requirements.txt # Python dependencies
└── README.md # Documentation


---

## 🚀 Features
- **Login Page**: Dummy login (no data stored).  
- **Notebook Page**:  
  - Search drinks using natural language.  
  - Top-K FAISS recommendations.  
  - Evaluation metrics (Accuracy@K, Precision, Recall, F1).  
- **Feedback Page**: Simple text feedback (not stored).  

---

## 🔧 Setup & Run Locally
1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/drink-recommender.git
   cd drink-recommender
