import streamlit as st

from auth import authenticate
from retriever_manager import ALGORITHM_LABELS, search_all, search_single
from utils.data_loader import load_qa_dataset

st.set_page_config(
    page_title="Amazon QA Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
    .sub-header { color: #666; margin-bottom: 1.5rem; }
    .result-card {
        background: #f8f9fa;
        border-left: 4px solid #ff9900;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .score-badge {
        background: #232f3e;
        color: white;
        padding: 0.15rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None


def login_page() -> None:
    st.markdown('<p class="main-header">Amazon QA Chatbot</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Sign in to search product Q&A with multiple retrieval algorithms</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                user = authenticate(username, password)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with st.expander("Demo credentials"):
            st.markdown(
                """
                | Username | Password |
                |----------|----------|
                | admin    | admin123 |
                | demo     | demo123  |
                | analyst  | analyst123 |
                """
            )


def render_result(result: dict) -> None:
    score_text = f"Score: {result['score']}"
    if "score_label" in result:
        score_text += f" ({result['score_label']})"
    if "bm25_component" in result:
        score_text += f" | BM25: {result['bm25_component']} | Cosine: {result['cosine_component']}"

    st.markdown(
        f"""
        <div class="result-card">
            <strong>#{result['rank']} — {result['product']}</strong><br/>
            <span class="score-badge">{score_text}</span>
            <p><strong>Q:</strong> {result['question']}</p>
            <p><strong>A:</strong> {result['answer']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chatbot_page() -> None:
    user = st.session_state.user

    with st.sidebar:
        st.markdown(f"### Welcome, {user['name']}")
        st.caption(f"Logged in as `{user['username']}`")

        mode = st.radio(
            "Search mode",
            ["Single algorithm", "Compare all algorithms"],
            index=1,
        )

        algorithm = st.selectbox(
            "Algorithm",
            options=list(ALGORITHM_LABELS.keys()),
            format_func=lambda k: ALGORITHM_LABELS[k],
            disabled=mode == "Compare all algorithms",
        )

        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()

        st.divider()
        st.markdown("**Algorithms**")
        for key, label in ALGORITHM_LABELS.items():
            st.markdown(f"- **{label}**")

        st.caption(f"Indexed {len(load_qa_dataset())} Q&A pairs")

    st.markdown('<p class="main-header">Amazon QA Search</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask a product question and compare retrieval results</p>',
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Your question",
        placeholder="e.g. Does Echo Dot work with Spotify?",
    )

    if st.button("Search", type="primary") and query.strip():
        with st.spinner("Searching..."):
            if mode == "Compare all algorithms":
                all_results = search_all(query.strip(), top_k=top_k)
                tabs = st.tabs([ALGORITHM_LABELS[name] for name in all_results])

                for tab, (name, results) in zip(tabs, all_results.items(), strict=True):
                    with tab:
                        if not results:
                            st.info("No results found.")
                        else:
                            for result in results:
                                render_result(result)
            else:
                results = search_single(algorithm, query.strip(), top_k=top_k)
                st.subheader(ALGORITHM_LABELS[algorithm])
                if not results:
                    st.info("No results found.")
                else:
                    for result in results:
                        render_result(result)

    elif not query.strip():
        st.info("Enter a question above to search the Amazon Q&A knowledge base.")

        st.subheader("Example questions")
        examples = [
            "Does Echo Dot work with Spotify?",
            "Is Kindle Paperwhite waterproof?",
            "Do I need a hub for Blink Mini?",
            "Can I schedule the smart plug?",
        ]
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"ex_{i}"):
                    st.session_state.example_query = example
                    st.rerun()

        if "example_query" in st.session_state:
            st.text_input("Your question", value=st.session_state.example_query, key="prefilled")


def main() -> None:
    init_session_state()
    if st.session_state.authenticated:
        chatbot_page()
    else:
        login_page()


if __name__ == "__main__":
    main()
