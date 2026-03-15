# Xenia AI - Hotel Guest Concierge

This is a RAG-based AI concierge built for the hackathon. It solves the problem of overloaded hotel receptions by allowing guests to ask questions about hotel policies, spa, or breakfast in their native language.
The app processes standard hotel directories (PDFs), extracts the exact rules, and provides accurate answers with source references (zero hallucinations).

## How to run locally

1. Clone this repository to your machine.
2. Install the required dependencies:
   pip install -r requirements.txt
3. Create a .env file in the root directory and add your OpenAI API key
4. Start the application: streamlit run main.py

## How to test the app

For your convenience, I have included a real hotel directory in this repository (amalia_hotel_guide.pdf).
Run the app and upload amalia_hotel_guide.pdf via the sidebar (Hotel Admin Panel).
Wait a few seconds for the vector database to process the document.
Try asking guest questions in the chat. For example:

"What time is breakfast?"
"Do you have luggage storage?"
Try asking the same questions in Greek, Spanish, or Russian to test the multilingual capabilities!

## Tech Stack

UI: Streamlit
AI / RAG: LangChain, ChromaDB (Local Vector Store)
Models: OpenAI GPT-4.1 & text-embedding-3-small
Processing: PyPDF
