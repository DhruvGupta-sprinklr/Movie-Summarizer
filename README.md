# Movie Details Generator (LangChain & Gemini)

This script takes a movie title, uses Google's Gemini LLM via LangChain to:
1.  Refine the movie title.
2.  Simulate fetching movie details (like plot, actors, year, rating).
3.  Generate a short theme for the movie.
4.  Create a formatted text file in the `movie_details` directory containing this information.

## Prerequisites

1.  **Node.js** (v18+).
2.  **Google API Key**:
    *   Create a `.env` file in the project root.
    *   Add your key: `GOOGLE_API_KEY="YOUR_GOOGLE_AI_STUDIO_API_KEY"`
    *   (`OMDB_API_KEY="PLACEHOLDER"` - used for LLM context)

## How to Run

1.  **Install Dependencies**:
    ```bash
    npm install
    ```

2.  **Run the Script**:
    Provide a movie title as a command-line argument. If no title is given, it defaults to "Inception 2010".

    **Example:**
    ```bash
    node main.js "Pulp Fiction"
    ```

    **Output:**
    *   Console logs will show the processing steps.
    *   A file like `movie_details/pulp_fiction.txt` (or similar) will be created with the movie's details.