
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load the embedding model
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# Convert the CSV dataset into searchable text documents
def create_documents(df):
    documents = []

    # Document 1: Dataset overview
    overview = (
        "Dataset overview. "
        f"The dataset contains {len(df)} rows "
        f"and {len(df.columns)} columns. "
        f"The column names are: {', '.join(df.columns)}."
    )

    documents.append(overview)

    # Document 2: Information about each column
    for column in df.columns:
        description = (
            f"Column name: {column}. "
            f"Data type: {df[column].dtype}. "
            f"Number of missing values: {df[column].isna().sum()}. "
        )

        if pd.api.types.is_numeric_dtype(df[column]):
            description += (
                f"Minimum value: {df[column].min()}. "
                f"Maximum value: {df[column].max()}. "
                f"Average value: {df[column].mean():.2f}. "
            )
        else:
            examples = (
                df[column]
                .dropna()
                .astype(str)
                .unique()[:5]
                .tolist()
            )

            description += (
                f"Example values: {', '.join(examples)}. "
            )

        documents.append(description)

    # Documents containing small batches of actual rows
    # Limit the number of rows to keep the demo manageable
    limited_df = df.head(1000)

    chunk_size = 20

    for start in range(0, len(limited_df), chunk_size):
        chunk = limited_df.iloc[start:start + chunk_size]

        text = (
            f"Dataset rows {start + 1} to "
            f"{start + len(chunk)}:\n"
            + chunk.to_string(index=False)
        )

        documents.append(text)

    return documents


# Convert documents into embeddings
def build_search_index(documents, model):
    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return embeddings


# Find documents most relevant to a question
def retrieve_documents(
    question,
    documents,
    embeddings,
    model,
    top_k=4
):
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        question_embedding,
        embeddings
    )[0]

    # Get the indexes of the most relevant documents
    ranked_indexes = similarities.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indexes:
        results.append({
            "text": documents[index],
            "score": float(similarities[index])
        })

    return results