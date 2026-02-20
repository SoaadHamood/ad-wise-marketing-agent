import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from app.llm import ask_llm
from dotenv import load_dotenv

load_dotenv()
model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_strategy_and_ads(user_text: str, analysis_report: str, top_k: int = 5) -> dict:
    """
    Worker 2: Generates original strategy based on diagnosis and Pinecone inspiration.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("ad-wise-index")

    query_vector = model.encode(user_text).tolist()
    search_results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    retrieved_ads = [m["metadata"]["ad_text"] for m in search_results["matches"]]

    system_prompt = (
        "You are the Creative Strategist for Ad-Wise. Use the provided diagnosis to build a strategy.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Formulate 2 strategies to improve performance. DO NOT repeat the diagnosis numbers.\n"
        "2. Write 3 original ad headlines based on the invisible inspiration from successful ads.\n"
        "3. DO NOT mention the retrieved ads or the historical data by name."
    )

    user_prompt = f"Diagnosis: {analysis_report}\n\nInspiration Ads: {chr(10).join(retrieved_ads)}"
    return {"strategy_and_ads": ask_llm(system_prompt, user_prompt), "retrieved_ads": retrieved_ads}