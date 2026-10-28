# 🤖 Reyaz's AI Chatbot

### An Interactive AI Assistant Built with Python, Streamlit & Groq API

A web-based AI chatbot that enables natural conversations through a clean and interactive interface. Powered by the Groq API, this project demonstrates practical skills in Python development, Large Language Model (LLM) integration, prompt engineering, session-based conversation history, and cloud deployment.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/AI-Groq%20API-orange" alt="Groq API"/>
  <img src="https://img.shields.io/badge/Deployment-Streamlit%20Cloud-success" alt="Deployment"/>
</p>

---

## 🌐 Live Demo

🚀 **Try the chatbot here:**

### 👉 [Launch Reyaz's AI Chatbot](https://ai-chatbot-r6dhcyr4ez8scugkek4q3m.streamlit.app/)

📂 **Source Code:** [View on GitHub](https://github.com/Reyaz99578/Ai-chatbot)

---

## ✨ Features

- 💬 **Interactive Chat Interface** — Chat with an AI assistant through a simple web UI.
- ⚡ **AI-Powered Responses** — Generate responses using the Groq API.
- 🌍 **Language-Aware Responses** — Designed to reply in the user's language, including Hinglish.
- 🧠 **Session-Based Chat History** — Maintains conversation context during the active session.
- 🔄 **Streaming Responses** — Display AI responses progressively as they arrive.
- 🎛️ **Model Selection** — Choose from available models returned by the API.
- 🌡️ **Temperature Control** — Adjust the variability of generated responses.
- 🧹 **Clear Chat** — Reset the current conversation.
- 🔐 **Secrets-Based API Configuration** — Configure the API key using Streamlit Secrets.
- ☁️ **Cloud Deployment** — Access the application online through Streamlit Community Cloud.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Web interface and chat components |
| Groq API | AI model inference |
| Groq Python SDK | Communicating with the API |
| Git & GitHub | Version control and source code management |
| Streamlit Community Cloud | Hosting and deployment |

---

## 🏗️ Application Workflow

```text
User Input
    ↓
Streamlit Chat Interface
    ↓
Session-Based Conversation History
    ↓
System Prompt + Selected AI Model
    ↓
Groq API
    ↓
Streamed AI Response
    ↓
Chat Interface
```

1. The user enters a message in the chat interface.
2. The application stores the message in Streamlit session state.
3. The system prompt and conversation history are sent to the selected Groq model.
4. The API generates a response.
5. Streamlit displays the response as it streams in.
6. The assistant's response is saved to the conversation history.

---

## 📂 Project Structure

```text
Ai-chatbot/
├── app.py              # Main chatbot application
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Reyaz99578/Ai-chatbot.git
cd Ai-chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create `.streamlit/secrets.toml` in the project directory:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Replace the placeholder with your own Groq API key. Never upload your actual API key to GitHub.

### 5. Start the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL where you can interact with the chatbot.

---

## 🔐 Security Practices

- Store API credentials in Streamlit Secrets.
- Never hardcode private API keys into source code.
- Do not commit `.streamlit/secrets.toml` to a public repository.
- Keep virtual environments and sensitive configuration files out of version control.

---

## 🧠 Key Technical Learnings

This project provides hands-on experience with:

- Integrating a third-party LLM API into a Python application.
- Building interactive chat experiences using Streamlit.
- Managing application state with `st.session_state`.
- Using system prompts to guide assistant behavior.
- Streaming model output for a responsive user experience.
- Managing Python dependencies through `requirements.txt`.
- Configuring API credentials for cloud deployment.
- Maintaining and documenting source code with GitHub.

---

## 🚀 Future Enhancements

- Export chat conversations to text or PDF.
- Add downloadable chat history.
- Introduce persistent conversation storage.
- Add document upload and document-based question answering.
- Improve error handling and API usage feedback.
- Add automated tests and continuous integration.

---

## 👨‍💻 Author

**Reyaz**

- GitHub: [@Reyaz99578](https://github.com/Reyaz99578)
- Project Repository: [Ai-chatbot](https://github.com/Reyaz99578/Ai-chatbot)
- Live Application: [Try the Chatbot](https://ai-chatbot-r6dhcyr4ez8scugkek4q3m.streamlit.app/)

---

⭐ If you find this project interesting, consider giving the repository a star!

*Built to demonstrate practical Python development, AI API integration, and cloud deployment skills.*
