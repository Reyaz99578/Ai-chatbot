# 🤖 Reyaz's AI Chatbot

### An Interactive AI Assistant Powered by Python, Streamlit & Groq

An AI-powered chatbot designed to provide fast, interactive, and user-friendly conversations through a clean web interface. Built with Python and Streamlit, this project integrates the Groq API to generate AI responses and support natural conversations in multiple languages.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/AI-Groq%20API-orange" alt="Groq API"/>
  <img src="https://img.shields.io/badge/Status-Deployed-success" alt="Deployment"/>
</p>

---

## 🚀 Live Demo

**Try the chatbot:** [Open Live App](PASTE_YOUR_STREAMLIT_APP_URL_HERE)

**Source Code:** [View GitHub Repository](https://github.com/Reyaz99578/Ai-chatbot)

---

## ✨ Features

- **AI-Powered Conversations:** Generates responses using the Groq API.
- **Interactive Chat Interface:** Chat with the AI through a simple, responsive Streamlit interface.
- **Multilingual Responses:** Designed to respond in the language used by the user, including Hinglish.
- **Model Selection:** Select from the chat models returned by the Groq API.
- **Adjustable Temperature:** Control response variability with a temperature setting.
- **Chat History:** Maintains conversation history during the current session.
- **Clear Chat:** Reset the conversation with one click.
- **Cloud Deployment:** Deployed using Streamlit Community Cloud.
- **Secure API Configuration:** Uses Streamlit Secrets to configure the API key.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Interactive web interface |
| Groq API | AI model access and response generation |
| Groq Python SDK | API integration |
| Git & GitHub | Version control and source code hosting |
| Streamlit Community Cloud | Application deployment |

---

## 🏗️ How It Works

1. The user enters a message through the Streamlit chat interface.
2. The application stores the message in the current session's chat history.
3. A system prompt guides the AI's response style and language.
4. The application sends the conversation and selected model to the Groq API.
5. The response is streamed to the interface.
6. The conversation continues until the user clears the chat or the session ends.

---

## 📂 Project Structure

```text
Ai-chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Reyaz99578/Ai-chatbot.git
cd Ai-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Groq API key

Create a file named `.streamlit/secrets.toml` in the project directory:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Replace the placeholder with your own API key. Never commit API keys or secrets to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔐 Security

- API credentials are configured through Streamlit Secrets.
- Secret values should never be hardcoded into source code.
- The `.streamlit/secrets.toml` file should not be committed to a public repository.
- A `.gitignore` file should exclude secrets and local virtual environments.

---

## 🧠 Key Learning Outcomes

This project demonstrates practical experience with:

- Integrating a third-party AI API into a Python application.
- Building an interactive application using Streamlit.
- Managing dependencies using `requirements.txt`.
- Handling conversation state with `st.session_state`.
- Streaming AI responses for a more interactive experience.
- Configuring API credentials securely.
- Deploying a Python application using Streamlit Community Cloud.
- Organizing and documenting a project using GitHub.

---

## 🔮 Future Improvements

- Persistent chat history across sessions.
- Export conversations as text or PDF.
- Support for uploading documents and asking questions about them.
- Improved error handling and API usage feedback.
- Unit tests and automated deployment checks.
- More advanced conversation controls.

---

## 👨‍💻 Author

**Reyaz**

GitHub: [@Reyaz99578](https://github.com/Reyaz99578)

---

⭐ If you find this project interesting, consider starring the repository!

*Built as a hands-on project to explore Python application development and AI API integration.*
