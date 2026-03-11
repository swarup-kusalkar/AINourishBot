import chromadb
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

# Persistent ChromaDB client stored in project directory
_client = chromadb.PersistentClient(path="./chroma_db")


def _get_collection(name: str):
    return _client.get_or_create_collection(name=name)


# --------------- User Feedback ---------------

def store_feedback(recipe_title: str, rating: int, comment: str, dietary_restrictions: str = "") -> str:
    """Store user feedback about a recipe into the vector DB."""
    collection = _get_collection("user_feedback")
    doc_text = (
        f"Recipe: {recipe_title}. Rating: {rating}/5. "
        f"Comment: {comment}. Diet: {dietary_restrictions}."
    )
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[doc_text],
        metadatas=[{
            "recipe_title": recipe_title,
            "rating": rating,
            "comment": comment,
            "dietary_restrictions": dietary_restrictions,
            "timestamp": datetime.now().isoformat(),
        }],
    )
    return "Feedback stored successfully."


def retrieve_feedback(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant past feedback from vector DB via semantic search."""
    collection = _get_collection("user_feedback")
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query], n_results=min(n_results, collection.count()))
    items = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        items.append({"document": doc, **meta})
    return items


# --------------- Health News ---------------

def store_news(headline: str, summary: str, health_tip: str, source: str) -> str:
    """Store a processed health news item into the vector DB."""
    collection = _get_collection("health_news")
    doc_text = f"Headline: {headline}. Summary: {summary}. Health tip: {health_tip}."
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[doc_text],
        metadatas=[{
            "headline": headline,
            "summary": summary,
            "health_tip": health_tip,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }],
    )
    return "News stored successfully."


def retrieve_news(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant health news from vector DB via semantic search."""
    collection = _get_collection("health_news")
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query], n_results=min(n_results, collection.count()))
    items = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        items.append({"document": doc, **meta})
    return items


# --------------- Recipe History ---------------

def store_recipe_history(recipe_title: str, ingredients: List[str], health_score: int,
                         prep_time: str, dietary_restrictions: str = "") -> str:
    """Store a generated recipe into history for future learning."""
    collection = _get_collection("recipe_history")
    doc_text = (
        f"Recipe: {recipe_title}. Ingredients: {', '.join(ingredients)}. "
        f"Health score: {health_score}/10. Prep time: {prep_time}. Diet: {dietary_restrictions}."
    )
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[doc_text],
        metadatas=[{
            "recipe_title": recipe_title,
            "ingredients": json.dumps(ingredients),
            "health_score": health_score,
            "prep_time": prep_time,
            "dietary_restrictions": dietary_restrictions,
            "timestamp": datetime.now().isoformat(),
        }],
    )
    return "Recipe history stored successfully."


def retrieve_recipe_history(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant past recipes from vector DB."""
    collection = _get_collection("recipe_history")
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query], n_results=min(n_results, collection.count()))
    items = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        items.append({"document": doc, **meta})
    return items
