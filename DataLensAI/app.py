
import os
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google import genai

from rag_engine import (
    load_embedding_model,
    create_documents,
    build_search_index,
    retrieve_documents
)
from data_analyzer import analyze_question, build_insights_summary, clean_dataframe

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DataLens AI | RAG Analytics",
    page_icon="📊",
    layout="wide"
)

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #21183f 0%, transparent 35%),
        #080b14;
    color: #f0f3ff;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

[data-testid="stSidebar"] {
    background: #0c101d;
    border-right: 1px solid #252c40;
}

.hero {
    font-size: clamp(32px, 4vw, 48px);
    font-weight: 800;
    letter-spacing: -1.5px;
    line-height: 1.2;
}

.gradient {
    background: linear-gradient(90deg, #a997ff, #63e4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #9ba6c2;
    font-size: 14px;
    line-height: 1.8;
}

.panel {
    background: #111727;
    border: 1px solid #293149;
    border-radius: 16px;
    padding: 20px;
}

[data-testid="stChatMessage"] {
    background: #111727;
    border: 1px solid #293149;
    border-radius: 15px;
    padding: 16px;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid #454064;
    background: #19182d;
    color: #f0f3ff;
}

.stButton > button:hover {
    border-color: #9b87ff;
    color: white;
}

[data-testid="stChatInput"] {
    border-color: #454064;
    border-radius: 14px;
}

h1, h2, h3 {
    color: #f0f3ff;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD AI MODELS
# --------------------------------------------------

@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


if not API_KEY:
    st.error(
        "Gemini API key not found. "
        "Check your .env file."
    )
    st.stop()

client = genai.Client(api_key=API_KEY)
def apply_new_dataframe(new_df):
    """Replace the dataset and rebuild the search index."""
    model = get_embedding_model()
    docs = create_documents(new_df)

    st.session_state.df = new_df
    st.session_state.documents = docs
    st.session_state.embeddings = build_search_index(docs, model)
    st.session_state.doc_hash = None
    st.session_state.insights = None

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "dataset_hash" not in st.session_state:
    st.session_state.dataset_hash = None

if "df" not in st.session_state:
    st.session_state.df = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
    
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None    


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 📊 DataLens AI")
    st.caption("YOUR PERSONAL DATA INTELLIGENCE")

    st.divider()

    st.markdown("### 📁 Upload your dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )
    doc_file = st.file_uploader(
        "Optional: dataset documentation",
        type=["txt", "md"]
    )

    if st.button("＋ New conversation",
                    use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    top_k = st.slider(
        "Sources to use",
        min_value=1, max_value=8, value=4,
        help="How many pieces of information the AI looks at "
            "before answering. Fewer is more focused, "
            "more gives broader context."
    )

    st.divider()

    if st.session_state.df is not None:
        st.markdown("### 💡 Try asking")

        side_df = st.session_state.df
        num_cols = side_df.select_dtypes(
            include="number"
        ).columns.tolist()
        cat_cols = [
            c for c in side_df.columns
            if not pd.api.types.is_numeric_dtype(side_df[c])
            and 2 <= side_df[c].nunique() <= 30
        ]

        suggestions = ["Summarize the dataset"]

        if num_cols:
            suggestions.append(f"What is the average {num_cols[0]}?")
            suggestions.append(f"What is the highest {num_cols[0]}?")

        if num_cols and cat_cols:
            suggestions.append(
                f"Which {cat_cols[0]} has the highest "
                f"average {num_cols[0]}?"
            )

        if len(num_cols) >= 2:
            suggestions.append(
                f"Show the relationship between "
                f"{num_cols[0]} and {num_cols[1]}"
            )

        for i, s in enumerate(suggestions):
            if st.button(s, key=f"suggest_{i}",
                        use_container_width=True):
                st.session_state.pending_question = s

        st.divider()

    if st.session_state.messages:
        lines = []
        for m in st.session_state.messages:
            lines.append(f"{m['role'].upper()}:")
            if m.get("calculation"):
                lines.append(
                    f"Calculated result: {m['calculation']}"
                )
            lines.append(m["content"])
            lines.append("")

        st.download_button(
            "⬇ Download chat",
            data="\n".join(lines),
            file_name="datalens_chat.txt",
            mime="text/plain",
            use_container_width=True
        )

    with st.expander("ℹ️ How it works"):
        st.markdown(
            "1. Your CSV is converted into text documents.\n"
            "2. Documents become embeddings (all-MiniLM-L6-v2).\n"
            "3. Your question retrieves the most similar documents.\n"
            "4. Pandas calculates the exact result.\n"
            "5. Gemini explains the result in simple words."
        )

    st.divider()

    st.divider()

    if st.session_state.df is not None:
        st.success("Dataset loaded")
        st.caption(
            f"{len(st.session_state.df)} rows · "
            f"{len(st.session_state.df.columns)} columns"
        )
    else:
        st.info("Upload a CSV to begin")


# --------------------------------------------------
# PROCESS UPLOADED CSV
# --------------------------------------------------

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(file_bytes).hexdigest()

    # Only rebuild the search index when the file changes
    if current_hash != st.session_state.dataset_hash:

        try:
            df = pd.read_csv(
                uploaded_file
            )

            if df.empty:
                st.error("The uploaded CSV has no data.")
                st.stop()

            if len(df.columns) == 0:
                st.error("No columns were found.")
                st.stop()

            with st.spinner(
                "Preparing your dataset and creating embeddings..."
            ):
                model = get_embedding_model()

                documents = create_documents(df)

                embeddings = build_search_index(
                    documents,
                    model
                )

                st.session_state.df = df
                st.session_state.original_df = df.copy()
                st.session_state.documents = documents
                st.session_state.embeddings = embeddings
                st.session_state.dataset_hash = current_hash
                st.session_state.doc_hash = None
                st.session_state.insights = None
                st.session_state.messages = []

                st.rerun()

        except Exception as e:
            st.error("Could not process this CSV file.")
            st.caption(str(e))

# --------------------------------------------------
# PROCESS DOCUMENTATION FILE
# --------------------------------------------------

if doc_file is not None and st.session_state.df is not None:
    doc_text = doc_file.getvalue().decode(
        "utf-8", errors="ignore"
    )
    doc_hash = hashlib.sha256(doc_text.encode()).hexdigest()

    if st.session_state.get("doc_hash") != doc_hash:
        chunks = [
            "Documentation: " + p.strip()
            for p in doc_text.split("\n\n")
            if p.strip()
        ]

        if chunks:
            with st.spinner("Indexing documentation..."):
                model = get_embedding_model()
                new_embeddings = build_search_index(chunks, model)

                st.session_state.documents = (
                    st.session_state.documents + chunks
                )
                st.session_state.embeddings = np.vstack([
                    st.session_state.embeddings,
                    new_embeddings
                ])
                st.session_state.doc_hash = doc_hash

            st.toast(f"Added {len(chunks)} documentation chunks")
# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    Ask your data.<br>
    <span class="gradient">Discover the insight.</span>
</div>

<p class="subtitle">
    Upload a dataset, ask questions in natural language,
    and let AI retrieve relevant information to help answer them.
</p>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

if st.session_state.df is not None:
    df = st.session_state.df

    st.markdown("### 📁 Your dataset")

    m1, m2, m3 = st.columns(3)

    m1.metric("Total rows", f"{len(df):,}")
    m2.metric("Total columns", len(df.columns))
    m3.metric(
        "Missing values",
        f"{int(df.isna().sum().sum()):,}"
    )

    with st.expander("Preview uploaded data"):
        st.dataframe(
            df.head(100),
            use_container_width=True
        )

    with st.expander("View column information"):
        info = pd.DataFrame({
            "Column": df.columns,
            "Data type": [
                str(df[col].dtype) for col in df.columns
            ],
            "Missing values": [
                int(df[col].isna().sum())
                for col in df.columns
            ]
        })
    if st.button("✨ Generate AI insights report"):
        with st.spinner("Analyzing your dataset..."):
            stats = build_insights_summary(df)

            insight_prompt = f"""
You are DataLens AI. Below are statistics computed with
Pandas on a dataset. Write a short insights report with
3 to 5 bullet points in simple language, then one
suggestion for further analysis.
Use only the numbers given. Do not invent statistics.
Correlation does not prove causation.

STATISTICS:
{stats}
"""
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=insight_prompt
            )
            st.session_state.insights = response.text

        if st.session_state.get("insights"):
            st.markdown("### 🧠 AI insights")
            st.markdown(st.session_state.insights)

    with st.expander("📈 Charts explorer"):
        numeric_cols = df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_cols:
            chosen = st.selectbox(
                "Histogram of column", numeric_cols
            )
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(df[chosen].dropna(), bins=10,
                    color="#a997ff", edgecolor="#080b14")
            ax.set_xlabel(chosen)
            ax.set_ylabel("Count")
            st.pyplot(fig)

        if len(numeric_cols) >= 2:
            st.markdown("**Correlation matrix**")
            st.dataframe(
                df[numeric_cols].corr().round(2)
                .style.background_gradient(
                    cmap="RdBu", vmin=-1, vmax=1
                ),
                use_container_width=True
            )

        missing = df.isna().sum()
        missing = missing[missing > 0]

        if len(missing) > 0:
            st.markdown("**Missing values by column**")
            st.bar_chart(missing)
        else:
            st.success("No missing values found.")
            
    with st.expander("🧹 Data cleaning"):
        if st.session_state.get("original_df") is None:
            st.session_state.original_df = st.session_state.df.copy()

        remove_dupes = st.checkbox(
            "Remove duplicate rows", value=True
        )
        strip_text = st.checkbox(
            "Trim extra spaces in text columns", value=True
        )
        missing_option = st.selectbox(
            "Missing values",
            [
                "Leave as they are",
                "Fill numeric with median, text with 'Unknown'",
                "Drop rows with missing values"
            ]
        )

        c1, c2 = st.columns(2)
        apply_clean = c1.button(
            "Apply cleaning", use_container_width=True
        )
        reset_clean = c2.button(
            "Restore original", use_container_width=True
        )

        if apply_clean:
            cleaned, report = clean_dataframe(
                st.session_state.original_df,
                remove_dupes, strip_text, missing_option
            )

            if cleaned.empty:
                st.error(
                    "Cleaning removed every row. "
                    "Try a different missing-value option."
                )
            else:
                with st.spinner("Rebuilding search index..."):
                    apply_new_dataframe(cleaned)
                st.session_state.clean_report = report
                st.rerun()

        if reset_clean:
            with st.spinner("Restoring original data..."):
                apply_new_dataframe(
                    st.session_state.original_df.copy()
                )
            st.session_state.clean_report = [
                "Original dataset restored"
            ]
            st.rerun()

        if st.session_state.get("clean_report"):
            for line in st.session_state.clean_report:
                st.success(line)    

        st.dataframe(
            info,
            use_container_width=True,
            hide_index=True
        )
        
        with st.expander("View summary statistics"):
            st.dataframe(
            df.describe(),
            use_container_width=True
        )    

else:
    st.markdown("""
    <div class="panel">
        <h3>👋 Welcome to DataLens AI</h3>
        <p style="color:#9ba6c2;">
            Upload a CSV file using the sidebar to begin.
            Your data will be converted into searchable
            text embeddings for retrieval.
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_chart(chart_type, chart_df, chart_title=None):
    if chart_df is None or chart_type is None:
        return

    st.markdown(f"#### 📊 {chart_title or 'Visualization'}")

    if chart_type == "bar":
        st.bar_chart(
            chart_df,
            x=chart_df.columns[0],
            y="Average"
        )

    elif chart_type == "scatter":
        st.scatter_chart(
            chart_df,
            x=chart_df.columns[0],
            y=chart_df.columns[1]
        )
# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

st.markdown("### 💬 Chat with your data")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("calculation"):
            st.markdown("#### 📌 Calculated result")
            st.write(message["calculation"])

        st.markdown(message["content"])
        
        render_chart(
            message.get("chart_type"),
            message.get("chart_df"),
            message.get("chart_title")
        )

        if message.get("sources"):
            with st.expander("🔎 Retrieved context"):
                for i, source in enumerate(
                    message["sources"], start=1
                ):
                    st.markdown(f"**Source {i}**")
                    st.caption(
                        f"Similarity: {source['score']:.3f}"
                    )
                    st.code(source["text"])

# --------------------------------------------------
# QUESTION AND RAG PIPELINE
# --------------------------------------------------


# --------------------------------------------------
# QUESTION AND RAG + DATA ANALYSIS PIPELINE
# --------------------------------------------------
question = st.chat_input(
    "Ask a question about your dataset..."
)

if question is None and st.session_state.pending_question:
    question = st.session_state.pending_question

st.session_state.pending_question = None

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        if st.session_state.df is None:
            answer = (
                "Please upload a CSV file using the sidebar "
                "before asking questions about your data."
            )

            st.info(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        else:
            try:
                df = st.session_state.df

                # STEP 1: Retrieve relevant context using RAG
                with st.spinner(
                    "🔎 Retrieving relevant information..."
                ):
                    model = get_embedding_model()

                    results = retrieve_documents(
                        question=question,
                        documents=st.session_state.documents,
                        embeddings=st.session_state.embeddings,
                        model=model,
                        top_k=top_k
                    )

                context = "\n\n".join([
                    result["text"] for result in results
                ])

                # STEP 2: Perform actual calculations
                with st.spinner(
                    "📊 Analyzing your dataset..."
                ):
                    analysis = analyze_question(
                        question, df
                    )

                # STEP 3: Generate a grounded explanation
                with st.spinner(
                    "✨ Preparing your answer..."
                ):

                    if analysis is not None:
                        calculation = analysis["answer"]

                        prompt = f"""
You are DataLens AI, a data analysis assistant.

The following result was calculated directly using
Python and Pandas on the user's dataset.

VERIFIED CALCULATION:
{calculation}

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

Explain the calculated result in simple language.
Keep the numerical result unchanged.
Do not invent additional statistics.
If relevant, explain what the result means.
"""

                    else:
                        prompt = f"""
You are DataLens AI, a data analysis assistant.

Answer the user's question using the retrieved
context from the uploaded dataset.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

Rules:
- Do not invent columns or values.
- Do not guess numerical calculations.
- If the context is insufficient, say so.
- The retrieved rows may be only a sample.
- Explain the answer in simple language.
"""

                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=prompt
                    )

                    explanation = response.text or (
                        "No additional explanation was generated."
                    )

                # STEP 4: Display the result
                if analysis is not None:
                    st.success("Pandas calculation completed")

                    st.markdown("#### 📌 Calculated result")
                    st.write(analysis["answer"])

                    st.markdown("#### ✨ AI explanation")
                    st.markdown(explanation)

                    render_chart(
                        analysis["chart_type"],
                        analysis["chart_df"],
                        analysis["chart_title"]
                    )

                else:
                    st.markdown(explanation)

                    st.info(
                        "This question was answered using "
                        "retrieved context. It did not match "
                        "one of the supported calculation types."
                    )

                # STEP 5: Show retrieved sources
                with st.expander("🔎 Retrieved context"):
                    for i, source in enumerate(
                        results, start=1
                    ):
                        st.markdown(f"**Source {i}**")

                        st.caption(
                            f"Similarity: {source['score']:.3f}"
                        )

                        st.code(source["text"])

                # Save conversation
                saved_message = {
                    "role": "assistant",
                    "content": explanation,
                    "sources": results
                }

                if analysis is not None:
                    saved_message["calculation"] = analysis["answer"]
                    saved_message["chart_type"] = analysis["chart_type"]
                    saved_message["chart_df"] = analysis["chart_df"]
                    saved_message["chart_title"] = analysis["chart_title"]

                st.session_state.messages.append(
                    saved_message
                )

            except Exception as e:
                st.error(
                    "Something went wrong during analysis."
                )
                st.caption(str(e))

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "DATALENS AI · RAG-POWERED DATA ANALYSIS ASSISTANT"
)