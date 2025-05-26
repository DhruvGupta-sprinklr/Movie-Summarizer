# Langchain Movie Details & Theme Generator

This Node.js script fetches movie details (IMDb rating, cast, plot) using the OMDB API and generates a movie theme using Google Gemini, saving the output to a text file.

## Prerequisites

*   Node.js & npm
*   OMDB API Key
*   Google API Key (with Gemini enabled)

## Setup

1.  **Install Dependencies:**
    npm install dotenv axios @langchain/google-genai @langchain/core langchain

2.  **Create `.env` file:**
    In the project root, create `.env` with your API keys:

    OMDB_API_KEY="YOUR_OMDB_API_KEY"
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    

## How to Run

Execute the script from your terminal, providing the movie title as a command-line argument (use quotes for titles with spaces):

```bash
node movieAgent.js "Movie Title Here"
For ex: node movieAgent.js "Inception"