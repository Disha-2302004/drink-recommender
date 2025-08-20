
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load dataset
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_carbonated_drinks.csv")

df = load_data()

# -------------------------
# Build embeddings + FAISS index
# -------------------------
@st.cache_resource
def build_index(data):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = (
        data["FlavorProfile"].fillna("") + " " +
        data["UseCase"].fillna("") + " " +
        data["HealthTags"].fillna("") + " " +
        data["Type"].fillna("")
    ).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))
    return model, index, embeddings

model, faiss_index, embeddings = build_index(df)

# -------------------------
# Helper Functions
# -------------------------
def get_top_k_recommendations(query_embedding, k=5):
    D, I = faiss_index.search(query_embedding, k)
    results = df.iloc[I[0]].copy()
    results["score"] = 1 - D[0] / 2
    return results

def get_relevant_items(user_input: dict, df: pd.DataFrame):
    rel = pd.Series([True] * len(df))

    if user_input.get("flavor"):
        flavor = user_input["flavor"].strip().lower()
        rel = rel & df["FlavorProfile"].str.lower().str.contains(flavor, na=False)

    if user_input.get("use_case"):
        case = user_input["use_case"].strip().lower()
        rel = rel & df["UseCase"].str.lower().str.contains(case, na=False)

    if user_input.get("tags"):
        tokens = [t.strip().lower() for t in user_input["tags"].replace(",", " ").split() if t.strip()]
        if tokens:
            tag_mask = pd.Series([False] * len(df))
            for t in tokens:
                tag_mask = tag_mask | df["HealthTags"].str.lower().str.contains(t, na=False)
            rel = rel & tag_mask

    if user_input.get("type"):
        drink_type = user_input["type"].strip().lower()
        rel = rel & df["Type"].str.lower().str.contains(drink_type, na=False)

    return set(df.loc[rel, "ProductName"].astype(str).tolist())

def evaluate_model(user_input, query_embedding, k=5):
    recs = get_top_k_recommendations(query_embedding, k)
    recommended = set(recs["ProductName"].astype(str).tolist())
    relevant = get_relevant_items(user_input, df)

    if not relevant:
        return {"precision": None, "recall": None, "f1": None}

    true_pos = len(recommended & relevant)
    prec = true_pos / len(recommended) if recommended else 0
    rec = true_pos / len(relevant) if relevant else 0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0
    return {"precision": prec, "recall": rec, "f1": f1}

# -------------------------
# Streamlit Pages
# -------------------------
def login_page():
    st.title("🔐 Login")
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            st.session_state["logged_in"] = True
            st.success("Logged in successfully!")
    else:
        st.success("You are already logged in.")

def notebook_page():
    st.title("📒 Personalized Drink Recommendation")

    flavor = st.selectbox("Select Flavor Profile", [""] + sorted(df["FlavorProfile"].dropna().unique().tolist()))
    use_case = st.selectbox("Select Use Case", [""] + sorted(df["UseCase"].dropna().unique().tolist()))
    drink_type = st.selectbox("Select Drink Type", [""] + sorted(df["Type"].dropna().unique().tolist()))
    tags = st.text_input("Enter Health Tags (comma separated)")

    user_input = {"flavor": flavor, "use_case": use_case, "tags": tags, "type": drink_type}

    top_k = st.slider("Number of Recommendations", 1, 10, 5)

    if st.button("Get Recommendations"):
        query = f"{flavor} {use_case} {tags} {drink_type}"
        query_embedding = model.encode([query])

        results = get_top_k_recommendations(query_embedding, top_k)
        st.subheader("🥤 Top Recommendations")
        st.table(results[["ProductName", "FlavorProfile", "UseCase", "HealthTags", "Type", "score"]])

        eval_metrics = evaluate_model(user_input, query_embedding, top_k)
        if eval_metrics["precision"] is not None:
            st.subheader("📊 Evaluation Metrics")
            st.write(f"Precision: {eval_metrics['precision']:.2f}")
            st.write(f"Recall: {eval_metrics['recall']:.2f}")
            st.write(f"F1 Score: {eval_metrics['f1']:.2f}")
        else:
            st.info("No relevant items found for evaluation.")

def feedback_page():
    st.title("📝 Feedback")
    fb = st.text_area("Share your feedback")
    rating = st.slider("Rate us", 1, 5, 3)
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

# -------------------------
# Navigation
# -------------------------
PAGES = {"Login": login_page, "Recommend_drinks": notebook_page, "Feedback": feedback_page}

choice = st.sidebar.radio("Navigation", list(PAGES.keys()))
if choice == "Login":
    PAGES[choice]()
elif st.session_state.get("logged_in", False):
    PAGES[choice]()
else:
    st.warning("Please login first.")
