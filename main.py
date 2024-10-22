import json
import os
import re
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount static files for serving CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Load FAQ data from the JSON file
with open('faqs.json', 'r') as f:
    faq_data = json.load(f)

# Initialize Sentence Transformer model for semantic search
model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_faqs(faq_data):
    faq_list = []
    for category, items in faq_data.items():
        for faq in items:
            question = faq['question']
            answer = faq['answer']
            faq_list.append((question, answer, category))
    return faq_list

# Prepare FAQ data
faq_list = preprocess_faqs(faq_data)
faq_questions = [faq[0] for faq in faq_list]
faq_embeddings = model.encode(faq_questions)

# Data model for the POST request
class QueryRequest(BaseModel):
    query: str
    category: str
    top_n: int

# Function to find the most relevant FAQs based on the query
def find_relevant_faq(query, category, top_n=1):
    if category == "all":
        faqs_to_search = faq_list
        faq_embeddings_to_search = faq_embeddings
    else:
        filtered_faqs = [(q, a, c) for q, a, c in faq_list if c == category]
        faqs_to_search = filtered_faqs
        filtered_questions = [q for q, a, c in filtered_faqs]
        faq_embeddings_to_search = model.encode(filtered_questions)

    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, faq_embeddings_to_search)[0]

    # Get the top n most similar FAQs
    top_indices = similarities.argsort()[-top_n:][::-1]
    relevant_faqs = [{"question": faqs_to_search[i][0], "answer": faqs_to_search[i][1], "category": faqs_to_search[i][2]} for i in top_indices]

    return relevant_faqs




# Route to serve the main FAQ page
@app.get("/", response_class=HTMLResponse)
async def get_faq_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to handle FAQ query submission
@app.post("/query")
async def generate_faqs(request: QueryRequest):
    query = request.query
    category = request.category
    top_n = request.top_n
    relevant_faqs = find_relevant_faq(content, category, top_n)
    
    return {"results": relevant_faqs}
