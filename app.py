import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Reyaz's AI Chatbot", page_icon="🤖")
st.title("Reyaz's AI Chatbot")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer clearly and simply. "
    "Reply in the same language the user writes in. "
    "If the user writes in Hinglish (Hindi in English letters), reply in Hinglish. "
    "Never reply in Arabic or Urdu unless the user writes in it."
)
DEFAULT_MODEL = "llama-3.3-70b-versatile"

client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


@st.cache_data(ttl=3600)
def get_models():
    """Groq se chalte hue chat models ki list lao."""
    try:
        ids = [m.id for m in client.models.list().data]
        skip = ("whisper", "tts", "guard", "playai", "orpheus", "allam")
        chat_models = [i for i in ids if not any(s in i.lower() for s in skip)]
        return sorted(chat_models) or [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    models = get_models()
    index = models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0
    model = st.selectbox("Model", models, index=index)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User input ---
prompt = st.chat_input("Say something")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                         + st.session_state.messages,
                temperature=temperature,
                stream=True,
            )
            response = st.write_stream(stream)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.messages.pop()  # failed message hata do
