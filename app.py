import os
import streamlit as st
import numpy as np

from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env file")
    st.stop()

client = genai.Client(api_key=API_KEY)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Campus Assistant",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666666;
    margin-bottom: 25px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🎓 AI Campus Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your intelligent assistant for campus information'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(text):

    chunk_size = 1000
    overlap = 200

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def get_embeddings(texts):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )

    return np.array([
        embedding.values
        for embedding in response.embeddings
    ])


# ============================================================
# COSINE SIMILARITY
# ============================================================

def similarity(a, b):

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0

    return np.dot(a, b) / denominator


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve(
    question,
    chunks,
    embeddings,
    top_k=3
):

    question_embedding = get_embeddings(
        [question]
    )[0]

    scores = []

    for index, embedding in enumerate(embeddings):

        score = similarity(
            question_embedding,
            embedding
        )

        scores.append(
            (score, index)
        )

    scores.sort(
        key=lambda x: x[0],
        reverse=True
    )

    results = []

    for score, index in scores[:top_k]:

        results.append({
            "text": chunks[index],
            "score": score,
            "index": index
        })

    return results


# ============================================================
# GENERIC GEMINI GENERATION
# ============================================================

def generate_with_prompt(
    title,
    context
):

    prompt = f"""
You are an AI Campus Assistant.

Use ONLY the information provided in the context.

Task:
{title}

Context:
{context}

Do not invent information.

If the required information is not available
in the context, clearly say that it was not
found in the provided campus documents.

Give a clear and useful response for a college student.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


# ============================================================
# GENERATE RAG ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_chunks
):

    context = "\n\n".join(
        item["text"]
        for item in retrieved_chunks
    )

    prompt = f"""
You are an AI Campus Assistant.

Answer the student's question using ONLY
the information provided in the context.

Do not invent information.

If the answer cannot be found in the context,
say:

"I could not find this information in the
provided campus documents."

Keep the answer clear, concise and
student-friendly.

CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


# ============================================================
# GET CONTEXT FOR QUICK ACTIONS
# ============================================================

def get_document_context():

    if "chunks" not in st.session_state:

        return None

    chunks = st.session_state["chunks"]

    # Use first 10 chunks for quick prototype tools
    context = "\n\n".join(
        chunks[:10]
    )

    return context


# ============================================================
# PDF UPLOAD
# ============================================================

st.subheader("📄 Upload Campus Document")

uploaded_file = st.file_uploader(
    "Upload your campus PDF",
    type=["pdf"]
)


if uploaded_file:

    file_id = uploaded_file.name

    # Process only if a new file is uploaded
    if st.session_state.get("file_name") != file_id:

        with st.spinner(
            "📚 Processing campus document..."
        ):

            try:

                # ----------------------------------------
                # STEP 1: PDF → TEXT
                # ----------------------------------------

                text = extract_text(
                    uploaded_file
                )

                if not text.strip():

                    st.error(
                        "❌ Could not extract text "
                        "from this PDF."
                    )

                    st.stop()

                # ----------------------------------------
                # STEP 2: TEXT → CHUNKS
                # ----------------------------------------

                chunks = create_chunks(
                    text
                )

                if not chunks:

                    st.error(
                        "❌ No text chunks were created."
                    )

                    st.stop()

                # ----------------------------------------
                # STEP 3: CHUNKS → EMBEDDINGS
                # ----------------------------------------

                embeddings = get_embeddings(
                    chunks
                )

                # ----------------------------------------
                # STORE IN SESSION
                # ----------------------------------------

                st.session_state["chunks"] = chunks

                st.session_state[
                    "embeddings"
                ] = embeddings

                st.session_state[
                    "file_name"
                ] = file_id

                st.session_state[
                    "action_result"
                ] = None

                st.session_state[
                    "active_action"
                ] = None

                st.success(
                    f"✅ Document processed successfully! "
                    f"{len(chunks)} chunks created."
                )

            except Exception as e:

                st.error(
                    f"❌ Error processing document: {e}"
                )


st.divider()


# ============================================================
# MAIN ASK ASSISTANT
# ============================================================

st.subheader("💬 Ask Your Campus Question")

question = st.text_input(
    "Enter your question",
    placeholder=(
        "Example: What is the attendance requirement?"
    )
)


if st.button(
    "🤖 Ask Assistant",
    use_container_width=True
):

    if not question:

        st.warning(
            "⚠️ Please enter a question."
        )

    elif "chunks" not in st.session_state:

        st.warning(
            "⚠️ Please upload a campus PDF first."
        )

    else:

        try:

            # RETRIEVAL

            with st.spinner(
                "🔍 Searching campus knowledge..."
            ):

                retrieved_chunks = retrieve(
                    question,
                    st.session_state["chunks"],
                    st.session_state["embeddings"],
                    top_k=3
                )

            # GENERATION

            with st.spinner(
                "🤖 Generating answer..."
            ):

                answer = generate_answer(
                    question,
                    retrieved_chunks
                )

            # ANSWER

            st.success(
                "✅ Answer generated successfully!"
            )

            st.subheader("🤖 Answer")

            st.markdown(answer)

            # SOURCES

            st.subheader(
                "📚 Retrieved Sources"
            )

            for i, item in enumerate(
                retrieved_chunks
            ):

                with st.expander(
                    f"Source {i + 1} "
                    f"• Similarity: "
                    f"{item['score']:.3f}"
                ):

                    st.write(
                        item["text"]
                    )

        except Exception as e:

            st.error(
                f"❌ Error generating answer: {e}"
            )


st.divider()


# ============================================================
# QUICK ACTIONS
# ============================================================

st.subheader("⚡ Quick Actions")

col1, col2, col3, col4 = st.columns(4)


# ============================================================
# ASK DOCUMENTS
# ============================================================

with col1:

    if st.button(
        "📚 Ask Documents",
        use_container_width=True
    ):

        if "chunks" not in st.session_state:

            st.warning(
                "⚠️ Please upload a campus PDF first."
            )

        else:

            st.session_state[
                "active_action"
            ] = "ask"

            st.session_state[
                "action_result"
            ] = None


# ============================================================
# GENERATE NOTES
# ============================================================

with col2:

    if st.button(
        "📝 Generate Notes",
        use_container_width=True
    ):

        if "chunks" not in st.session_state:

            st.warning(
                "⚠️ Please upload a campus PDF first."
            )

        else:

            with st.spinner(
                "📝 Generating notes..."
            ):

                context = get_document_context()

                notes = generate_with_prompt(
                    """
                    Generate well-structured study notes
                    from the provided academic content.

                    Use:
                    - Clear headings
                    - Bullet points
                    - Important definitions
                    - Important concepts

                    Keep the notes concise and useful
                    for a college student.
                    """,
                    context
                )

            st.session_state[
                "action_result"
            ] = notes

            st.session_state[
                "action_title"
            ] = "📝 Generated Notes"

            st.session_state[
                "active_action"
            ] = None


# ============================================================
# GENERATE QUIZ
# ============================================================

with col3:

    if st.button(
        "❓ Generate Quiz",
        use_container_width=True
    ):

        if "chunks" not in st.session_state:

            st.warning(
                "⚠️ Please upload a campus PDF first."
            )

        else:

            with st.spinner(
                "❓ Generating quiz..."
            ):

                context = get_document_context()

                quiz = generate_with_prompt(
                    """
                    Create a 5-question multiple-choice
                    quiz from the provided content.

                    For every question provide:

                    1. Question
                    A)
                    B)
                    C)
                    D)

                    Correct Answer:

                    Make the questions relevant to
                    the provided content.
                    """,
                    context
                )

            st.session_state[
                "action_result"
            ] = quiz

            st.session_state[
                "action_title"
            ] = "❓ Generated Quiz"

            st.session_state[
                "active_action"
            ] = None


# ============================================================
# IMPORTANT TOPICS
# ============================================================

with col4:

    if st.button(
        "⭐ Important Topics",
        use_container_width=True
    ):

        if "chunks" not in st.session_state:

            st.warning(
                "⚠️ Please upload a campus PDF first."
            )

        else:

            with st.spinner(
                "⭐ Finding important topics..."
            ):

                context = get_document_context()

                topics = generate_with_prompt(
                    """
                    Identify the most important topics
                    from the provided academic content.

                    Give:
                    1. Topic name
                    2. Short explanation
                    3. Why it is important

                    Provide a numbered list.
                    """,
                    context
                )

            st.session_state[
                "action_result"
            ] = topics

            st.session_state[
                "action_title"
            ] = "⭐ Important Topics"

            st.session_state[
                "active_action"
            ] = None


# ============================================================
# ASK DOCUMENTS PANEL
# ============================================================

if st.session_state.get(
    "active_action"
) == "ask":

    st.divider()

    st.subheader(
        "📚 Ask Documents"
    )

    document_question = st.text_input(
        "Ask something from your uploaded document",
        placeholder=(
            "Example: What are the examination rules?"
        ),
        key="document_question"
    )

    if st.button(
        "🔍 Search Documents"
    ):

        if not document_question:

            st.warning(
                "⚠️ Please enter a question."
            )

        elif "chunks" not in st.session_state:

            st.warning(
                "⚠️ Please upload a PDF first."
            )

        else:

            try:

                with st.spinner(
                    "🔍 Retrieving relevant information..."
                ):

                    retrieved = retrieve(
                        document_question,
                        st.session_state["chunks"],
                        st.session_state["embeddings"],
                        top_k=3
                    )

                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    answer = generate_answer(
                        document_question,
                        retrieved
                    )

                st.subheader(
                    "🤖 Answer"
                )

                st.markdown(
                    answer
                )

                st.subheader(
                    "📚 Sources"
                )

                for i, item in enumerate(
                    retrieved
                ):

                    with st.expander(
                        f"Source {i + 1} "
                        f"• Similarity: "
                        f"{item['score']:.3f}"
                    ):

                        st.write(
                            item["text"]
                        )

            except Exception as e:

                st.error(
                    f"❌ Error: {e}"
                )


# ============================================================
# QUICK ACTION RESULT
# ============================================================

if st.session_state.get(
    "action_result"
):

    st.divider()

    st.subheader(
        st.session_state.get(
            "action_title",
            "Result"
        )
    )

    st.markdown(
        st.session_state[
            "action_result"
        ]
    )


st.divider()


# ============================================================
# CAMPUS INFORMATION
# ============================================================

st.subheader(
    "🏫 Campus Information"
)

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown("""
    <div class="card">

    ### 🎓 Academics

    Courses, syllabus and academic information.

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="card">

    ### 👨‍🏫 Faculty

    Faculty and department information.

    </div>
    """, unsafe_allow_html=True)


with col3:

    st.markdown("""
    <div class="card">

    ### 📢 Notices

    Important campus announcements.

    </div>
    """, unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown("""
    <div class="card">

    ### 💼 Placements

    Placement and career information.

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="card">

    ### 🎉 Events

    Campus events and activities.

    </div>
    """, unsafe_allow_html=True)


with col3:

    st.markdown("""
    <div class="card">

    ### 🏫 Departments

    Department and course information.

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 AI Campus Knowledge Assistant • "
    "Powered by RAG + Gemini"
)