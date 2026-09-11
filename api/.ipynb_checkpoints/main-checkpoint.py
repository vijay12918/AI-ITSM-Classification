from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# create the FastAPI application
app = FastAPI(
    title = "AI IT Support Ticket Classification API",
    description = "Predict ticket queue, priority, type and retrieves similar tickets.",
    version = "1.0.0"
)

# Load our saved models
queue_tfidf = joblib.load("models/queue_tfidf.joblib")
queue_svm = joblib.load("models/queue_svm.joblib")

priority_tfidf = joblib.load("models/priority_tfidf.joblib")
priority_svm = joblib.load("models/priority_svm.joblib")

type_tfidf = joblib.load("models/type_tfidf.joblib")
type_svm = joblib.load("models/type_svm.joblib")

# Load the similarity search components
ticket_embeddings = np.load(
    "models/ticket_embeddings.npy"
)

ticket_data = pd.read_pickle(
    "models/ticket_data.pkl"
)

with open("models/embedding_model_name.txt", "r") as f:
    embedding_model_name = f.read().strip()

embedding_model = SentenceTransformer(
    embedding_model_name
)

# Define the input format
class TicketRequest(BaseModel):
    subject: str 
    body : str 
    top_n = 5

# Add the margin function
def calculate_margin(scores):

    scores = np.asarray(scores)

    if scores.ndim>1:
        scores = scores[0]
    
    sorted_scores = np.sort(scores)[::-1]

    if len(sorted_scores)<2:
        return 0.0

    return float(
        sorted_scores[0] - sorted_scores[1]
    )


# Add Similar-Ticket search
def find_similar_tickets(text, top_n=5):
    
    new_embedding = embedding_model.encode([text])
    new_embedding = normalize(new_embedding)

    similarities = cosine_similarity(
        new_embedding,
        ticket_embeddings
    )[0]

    top_indices = np.argsort(
        similarities
    )[::-1][:top_n]

    result = []

    for index in top_indices:
        row = ticket_data.iloc[index]

        results.append({
            "text":row["text"],
            "queue":row["queue"],
            "priority":row["type"],
            "similarity":float(similarities[index])
        })

    return results


# Add the /predict endpoint

@app.post("/predict")
def predict_ticket(ticket: TicketRequest):

    text = (
        f"{ticket.subject} {ticket.body}"
    ).strip()

    # Queue
    queue_vector = queue_tfidf.transform([text])
    queue_prediction = queue_svm.predict(queue_vector)[0]
    queue_scores = queue_svm.decision_function(queue_vector)
    queue_margin = calculate_margin(queue_scores)

    # Priority
    priority_vector = priority_tfidf.transform([text])
    priority_prediction = priority_svm.predict(priority_vector)[0]
    priority_scores = priority_svm.decision_function(priority_vector)
    priority_margin = calculate_margin(priority_scores)

    # Type
    type_vector = type_tfidf.transform([text])
    type_prediction = type_svm.predict(type_vector)[0]
    type_scores = type_svm.decision_function(type_vector)
    type_margin = calculate_margin(type_scores)

    # Manual review
    queue_review = queue_margin < 0.10
    priority_review = priority_margin < 0.20
    type_review = type_margin < 0.10

    # Similar tickets
    similar_tickets = find_similar_tickets(
        text,
        ticket.top_n
    )

    return {
        "ticket": {
            "subject": ticket.subject,
            "body": ticket.body
        },

        "prediction": {
            "queue": queue_prediction,
            "priority": priority_prediction,
            "type": type_prediction
        },

        "margins": {
            "queue": queue_margin,
            "priority": priority_margin,
            "type": type_margin
        },

        "manual_review": {
            "queue": queue_review,
            "priority": priority_review,
            "type": type_review
        },

        "similar_tickets": similar_tickets
    }















# simple health check
@app.get("/")
def root():

    return {
        "status": "running",
        "message": "AI IT Support Ticket Classification API"
    }

