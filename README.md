# LLM-Hotel-AI-Chatbot
Local AI hotel assistant built with LangChain, LangGraph & Ollama (Llama 3.2). Answers hotel FAQs using embedding-based RAG with ChromaDB, checks room availability and prices, and handles bookings and cancellations from the terminal.

1. REQUIREMENTS

Download the Ollama models required by the assistant:
    ollama pull llama3.2
    ollama pull nomic-embed-text

Install the Python dependencies:
    py -m pip install -r requirements.txt



2. RUN THE PROJECT

    py main.py
    Type exit to stop the program.



3. SAMPLE QUESTIONS (TYPE THESE INTO THE TERMINAL)
--------------------------------------------------

HOTEL INFORMATION - embedding-based RAG tool
1. Does your hotel have a swimming pool?
2. What time does the swimming pool close?
3. Is free Wi-Fi available?
4. Are pets allowed in the hotel?
5. What time is breakfast served?
6. Can guests smoke inside hotel rooms?

ROOMS AND PRICES - check_rooms tool

7. Show me all hotel room types and prices.
8. Which rooms are available on 2026-11-05?
9. Show available family rooms for 2026-11-05.

BOOKINGS - book_room tool

10. Book room 104 for Kasun, 2 guests, on 2026-11-05.
11. Book room 104 for Amal, 2 guests, on 2026-11-05.
12. Book room 104 for Amal, 2 guests, on 2026-11-06.
13. Book room 101 for Nimal, 3 guests, on 2026-11-05.

CANCELLATION - cancel_booking tool

14. Cancel Kasun's booking for room 104 on 2026-11-05.
15. Book room 104 for Amal, 2 guests, on 2026-11-05.

IMPORTANT: To test duplicate booking and cancellation, ask questions
10 through 15 in order during the SAME program run.
