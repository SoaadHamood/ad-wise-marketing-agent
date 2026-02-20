import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm

# Load API keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


def main():
    if not PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY is missing from .env")
        return

    print("⏳ Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "ad-wise-index"

    # 1. Create the index if it doesn't exist
    # Dimension 384 matches our free 'all-MiniLM-L6-v2' model
    if index_name not in pc.list_indexes().names():
        print(f"🏗️ Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(index_name)

    # 2. Load the CSV data
    # (MVP: We limit to 1000 rows so the upload finishes in seconds, not hours)
    print("⏳ Loading Amazon Ads CSV...")
    df = pd.read_csv("data/amazon_ads.csv").dropna(subset=["ad"]).head(1000)

    # 3. Load the free, local embedding AI
    print("🤖 Loading Embedding Model (this is free and runs locally)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 4. Embed and Upload in batches
    print("🚀 Embedding and uploading to Pinecone...")
    batch_size = 100

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i + batch_size]

        # AI converts text into numbers (vectors)
        embeddings = model.encode(batch["ad"].tolist())

        # Prepare data for Pinecone: (id, vector, metadata)
        vectors = []
        for j, (idx, row) in enumerate(batch.iterrows()):
            vectors.append((
                str(idx),
                embeddings[j].tolist(),
                {"ad_text": row["ad"]}  # We store the text so we can read it later!
            ))

        index.upsert(vectors=vectors)

    print("\n✅ Successfully embedded and uploaded Amazon ads to Pinecone!")


if __name__ == "__main__":
    main()