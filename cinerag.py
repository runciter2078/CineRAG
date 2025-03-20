#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
############################# CineRAG #############################
--------------------------- Redis-based RAG Pipeline for Movie Plots ---------------------------

Description:
This script implements a Retrieval-Augmented Generation (RAG) pipeline for movie plot data.
It uses Redis Search to store and retrieve vector embeddings of movie plots and integrates a 
transformer-based language model (from Hugging Face) to generate answers based on the retrieved texts.

Before running this script, make sure to:
  - Install Redis Stack and run the redis-stack-server (see README for details).
  - Install the required Python packages:
      pip install redis torch numpy pandas tqdm transformers sentence_transformers
  - Place your Hugging Face token in a file named "huggingface_key.txt" in the same directory.
  - Place the CSV file "wiki_movie_plots_deduped.csv" in the same directory or adjust the path accordingly.
  - (Optional) Pre-download and save the embeddings in "embeddings.npy" if already computed.

Usage:
  Run the script directly. It will load the data, compute or load embeddings,
  create (or recreate) a Redis index, insert the data, and finally perform retrieval
  and answer generation queries.
  
The function `ask(question)` is provided for convenience.
"""

import os
import redis
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from redis.commands.search.field import NumericField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# ---------------------
# Configuration
# ---------------------

# Set HF_HOME to a local folder (adjust if needed)
os.environ["HF_HOME"] = "./huggingface"

# Read Hugging Face token from a local file (ensure the file exists)
with open("huggingface_key.txt", "r", encoding="utf-8") as f:
    hf_token = f.read().strip()

model_id = "google/gemma-2-2b-it"
torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the language model and tokenizer for generation
model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, torch_dtype=torch.float16)

# Load the SentenceTransformer model for embeddings
embeddings_model = SentenceTransformer("thenlper/gte-large")

# ---------------------
# Redis Client Setup
# ---------------------
client = redis.Redis(host="localhost", port=6379, decode_responses=True)
try:
    client.ping()
    print("Connected to Redis successfully.")
except redis.exceptions.ConnectionError:
    print("Error: Redis server is not running. Please start redis-stack-server and try again.")
    exit(1)

# ---------------------
# Data Loading and Embeddings Computation
# ---------------------

def load_data(csv_path="wiki_movie_plots_deduped.csv", embeddings_path="embeddings.npy"):
    """
    Loads the movie plots CSV, computes embeddings if necessary, and cleans the DataFrame.
    """
    df = pd.read_csv(csv_path, encoding='utf-8')
    print("Data loaded. Showing head:")
    print(df.head())
    print("\nData info:")
    df.info()
    print("\nNumber of duplicated entries (Title, Release Year):", df.duplicated(subset=["Title", "Release Year"]).sum())

    # Define lambda to compute embedding
    get_embedding = lambda text: embeddings_model.encode(text).astype(np.float32).tolist() if text else []

    # Clean the DataFrame
    clean_df = df[["Title", "Plot", "Cast", "Release Year"]].copy()
    clean_df.columns = clean_df.columns.str.lower().str.replace(' ', '_')
    clean_df["title"] = clean_df["title"].str.strip()
    clean_df = clean_df.dropna(subset=["plot"])
    clean_df["cast"] = clean_df["cast"].fillna("Unknown")

    # Compute embeddings if not already computed
    if not os.path.exists(embeddings_path):
        print("Computing embeddings for each movie...")
        clean_df["embedding"] = clean_df.apply(lambda x: get_embedding(f'{x["title"]}: {x["plot"]}'), axis=1)
        embeddings = np.array(clean_df["embedding"].tolist())
        np.save(embeddings_path, embeddings)
        print("Embeddings computed and saved.")
    else:
        print("Loading precomputed embeddings...")
        embeddings = np.load(embeddings_path, allow_pickle=True)
        clean_df["embedding"] = [x.tolist() for x in embeddings]

    clean_df = clean_df.drop_duplicates(subset=["title", "release_year"], ignore_index=True)
    return clean_df

# ---------------------
# Create Redis Index and Insert Data
# ---------------------

def create_index_and_insert_data(df):
    """
    Defines the Redis index schema and inserts the data into Redis.
    """
    # Define the index schema for the JSON documents
    vector_dim = len(df["embedding"].iloc[0])
    data_schema = (
        TextField("$.title", no_stem=True, as_name="title"),
        TextField("$.plot", no_stem=True, as_name="plot"),
        TextField("$.cast", as_name="cast"),
        NumericField("$.release_year", as_name="release_year"),
        VectorField("$.embedding", as_name="embedding",
                    algorithm="FLAT",
                    attributes={
                        "TYPE": "FLOAT32",
                        "DIM": vector_dim,
                        "DISTANCE_METRIC": "COSINE",
                    })
    )

    definition = IndexDefinition(prefix=["movies:"], index_type=IndexType.JSON)

    # (Re)create the index
    try:
        res = client.ft("idx:movies_vss").create_index(fields=data_schema, definition=definition)
        print("Index created successfully.")
    except redis.exceptions.ResponseError:
        print("Index already exists. Dropping and recreating index...")
        client.ft("idx:movies_vss").dropindex()
        res = client.ft("idx:movies_vss").create_index(fields=data_schema, definition=definition)
        print("Index recreated successfully.")

    # Insert data into Redis using pipeline
    pipeline = client.pipeline()
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Inserting data"):
        key = f"movies:{i+1:03}"
        movie = row.to_dict()
        pipeline.json().set(key, "$", movie)
    res = pipeline.execute()
    print(f"Data insertion complete. All inserted? : {all(res)}")

    # Print index info
    info = client.ft("idx:movies_vss").info()
    print("Number of documents in index:", info["num_docs"])
    print("Indexing failures:", info["hash_indexing_failures"])

# ---------------------
# Retrieval and Question Answering
# ---------------------

# Define the query used for KNN retrieval (using 4 nearest neighbors)
query = (Query("(*)=>[KNN 4 @embedding $query_vector AS vector_score]")
         .sort_by("vector_score")
         .return_fields("title", "plot", "release_year")
         .dialect(2))

def retrieve(text):
    """
    Performs a KNN search in Redis for the given text using the embeddings model.
    """
    query_vector = np.array(embeddings_model.encode(text)).astype(np.float32).tobytes()
    result = client.ft("idx:movies_vss").search(query, {"query_vector": query_vector}).docs
    return result

def ask(question):
    """
    Takes a question, retrieves relevant movies from Redis, builds a prompt,
    and generates an answer using the language model.
    """
    # Retrieve documents based on the question
    retrieved_movies = retrieve(question)
    movies_str = "\n".join([f"{movie['title']} (year: {movie['release_year']}), plot: {movie['plot']}"
                            for movie in retrieved_movies])

    # Define the prompt with instructions
    prompt = """
Give me the information that fits the following instruction:

{question}

Using the following texts:

############

{movies_str}

###########

To do this:

    - **DONT** give information that is not in the texts
    - **DONT** give information that is not asked
    - **DONT** formulate any question

Answer:
"""
    formatted_prompt = prompt.format_map({"question": question, "movies_str": movies_str})
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    output = model.generate(**input_ids, max_new_tokens=120)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# ---------------------
# Main Execution
# ---------------------
def main():
    # Load data and compute embeddings if needed
    df = load_data()

    # (Re)create the Redis index and insert data
    create_index_and_insert_data(df)

    # --- Retrieval example ---
    print("\n--- Retrieval Example: Querying 'Aliens' ---")
    retrieved = retrieve("Aliens")
    for doc in retrieved:
        print("Title:", doc["title"])
        print("Release Year:", doc["release_year"])
        print("Plot:", doc["plot"])
        print("-" * 40)

    # --- QA Example ---
    question_text = "Which are the main characters on the film Scarface?"
    print("\n--- Question Answering Example ---")
    print("Question:", question_text)
    answer = ask(question_text)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
