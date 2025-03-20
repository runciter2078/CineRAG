# CineRAG

**A Redis-backed Retrieval-Augmented Generation (RAG) pipeline for movie plot data, integrating Redis Search for vector retrieval with a transformer-based text generation model.**

---

## Overview

CineRAG is a demonstration project that implements a RAG pipeline using Redis as a vector database. The pipeline:
- Loads movie plot data from a CSV file.
- Computes embeddings for each movie using a SentenceTransformer model.
- Stores the movie data and embeddings in Redis using Redis Search.
- Performs KNN retrieval based on a query.
- Uses a Hugging Face causal language model to generate an answer based on the retrieved texts.

This project is particularly suited for exploring how to combine vector search and generative models for retrieval-augmented question answering.

---

## Repository Structure

```
CineRAG/
├── cinerag.py              # Main Python script (converted from Jupyter notebook)
├── README.md               # This file
├── requirements.txt        # (Optional) List of Python dependencies
├── wiki_movie_plots_deduped.csv  # CSV file with movie plot data (or instructions to obtain it)
└── huggingface_key.txt     # File containing your Hugging Face API token (do not share publicly)
```

---

## Requirements

- **Python:** 3.8 or above (Python 3.10+ recommended)
- **Redis Stack:** Install and run [Redis Stack](https://redis.io/docs/stack/) (make sure `redis-stack-server` is running on localhost, port 6379)
- **Python Libraries:**  
  Install via pip:
  ```bash
  pip install redis torch numpy pandas tqdm transformers sentence_transformers
  ```

---

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/runciter2078/CineRAG.git
   cd CineRAG
   ```

2. **Configure Hugging Face Token:**
   - Create a file named `huggingface_key.txt` in the repository folder and paste your Hugging Face API token in it.

3. **Obtain the Data:**
   - Download the CSV file `wiki_movie_plots_deduped.csv` (if not provided) and place it in the repository folder.
   - Alternatively, adjust the path in `cinerag.py` accordingly.

4. **Install Dependencies:**
   - (Optional) Create a virtual environment and install the dependencies listed in `requirements.txt`:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     pip install -r requirements.txt
     ```

5. **Run Redis Stack Server:**
   - Ensure that Redis Stack is installed and running. You can download and run it from [Redis Stack Downloads](https://redis.io/docs/stack/get-started/install/).

---

## Usage

Run the main script:
```bash
python cinerag.py
```

The script will:
- Load and preprocess the movie data.
- Compute (or load) embeddings.
- (Re)create the Redis index and insert the data.
- Perform a sample retrieval (e.g., querying "Aliens").
- Execute a question-answering example for the question:
  > "Which are the main characters on the film Scarface?"

You can use the provided `ask(question)` function within the script as a starting point to integrate your own queries.

---

## Additional Notes

- **Indexing:**  
  The script drops and recreates the Redis index each time it runs. In a production scenario, you might want to modify this behavior.

- **Performance:**  
  Computing embeddings for a large dataset can take some time. Consider precomputing and saving the embeddings (the script saves them in `embeddings.npy`).

- **Extensibility:**  
  CineRAG is modular. You can replace the embedding model or the text generation model by adjusting the configuration in the script.

---

## License

MIT License (https://opensource.org/license/mit)

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests with improvements or bug fixes.

---

Enjoy exploring CineRAG and showcasing the power of Redis-backed RAG pipelines!
```
