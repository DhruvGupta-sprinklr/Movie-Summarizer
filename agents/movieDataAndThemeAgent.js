import 'dotenv/config';
import {
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
} from '@langchain/core/prompts';
import { RunnableLambda } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { extractJsonFromString } from '../utils.js';

const OMDB_API_KEY = process.env.OMDB_API_KEY;
const OMDB_BASE_URL = 'http://www.omdbapi.com/';

const movieDataAndThemePromptTemplate = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(
`You are a multi-talented movie information assistant.
Your task is to process a movie title through two stages and return a SINGLE JSON object.

STAGE 1: OMDB API Simulation
- Act as the OMDB API for the given movie title.
- The OMDB API base URL for context is: ${OMDB_BASE_URL}
- The API key for context is: ${OMDB_API_KEY} (Do NOT include this API key in your JSON response).
- If the movie is found based on your knowledge, populate the following fields in the JSON:
    - "Title": string (official movie title)
    - "Year": string (e.g., "2010")
    - "imdbRating": string (e.g., "8.8")
    - "Actors": string (comma-separated list of main actors, ideally 3-5 names)
    - "Genre": string (comma-separated list of genres)
    - "Plot": string (full plot summary)
    - "Response": "True" (as a string)
- If the movie is NOT found or you simulate an error:
    - Set "Response": "False" (as a string)
    - Set "Error": string (e.g., "Movie not found!" or "Error simulating data.")
    - For "Title", use the input movie title if possible, otherwise "N/A".
    - Set "Year", "imdbRating", "Actors", "Genre", "Plot" to "N/A".

STAGE 2: Movie Theme Generation
- Based *only* on the "Title", "Genre", and "Plot" you generated in STAGE 1:
- Derive a concise 'MovieTheme'. This theme should be a short, descriptive phrase (10-15 words max) capturing the central idea, tone, or dominant message.
- If "Plot" from STAGE 1 is "N/A", or "Response" from STAGE 1 is "False", or the plot is too brief, set "MovieTheme" to "Theme could not be determined due to lack of plot details."

OUTPUT REQUIREMENT:
- You MUST output a single, well-formed JSON object.
- This JSON object MUST contain all the fields from STAGE 1 ("Title", "Year", "imdbRating", "Actors", "Genre", "Plot", "Response", and "Error" if applicable) AND the "MovieTheme" field from STAGE 2.
- Do not include any other text, explanations, or markdown formatting (like \`\`\`json ... \`\`\`) around the JSON.
- Ensure "Actors" and "Genre" are comma-separated strings. For empty lists, use "N/A".
`
    ),
    HumanMessagePromptTemplate.fromTemplate(
        "Process the movie title: {refined_movie_title}"
    ),
]);

export function createMovieDataAndThemeAgent(llm) {
    const movieDataAndThemeLlmTool = movieDataAndThemePromptTemplate
        .pipe(llm)
        .pipe(new StringOutputParser());

    const movieDataAndThemeAgentRunnable = RunnableLambda.from(async (inputObject) => {
        console.log("\n[Agent: Movie Data Simulation & Theme Generation (Combined)]");
        const { refinedTitle, titleIsUncertain, raw_title } = inputObject;

        let movieDataWithTheme = {
            Title: refinedTitle || raw_title || "N/A",
            Year: "N/A",
            imdbRating: "N/A",
            Actors: "N/A",
            Genre: "N/A",
            Plot: "N/A",
            Response: "False",
            MovieTheme: "Theme could not be determined due to processing error.",
            Error: "Initial error before LLM call or due to uncertain title."
        };

        if (titleIsUncertain) {
            const message = `Cannot reliably fetch data or generate theme for uncertain title: "${refinedTitle || raw_title}"`;
            console.warn(`  ${message}`);
            movieDataWithTheme.Error = message;
        } else {
            try {
                const rawLlmJsonResponse = await movieDataAndThemeLlmTool.invoke({ refined_movie_title: refinedTitle });
                console.log(`  LLM Raw JSON Response (Data & Theme): ${rawLlmJsonResponse}`);

                const cleanedJsonString = extractJsonFromString(rawLlmJsonResponse);
                const parsedLlmOutput = JSON.parse(cleanedJsonString);

                movieDataWithTheme = {
                    ...movieDataWithTheme,
                    ...parsedLlmOutput,
                    Title: parsedLlmOutput.Title || refinedTitle,
                    Response: parsedLlmOutput.Response === "True" ? "True" : "False",
                    MovieTheme: parsedLlmOutput.MovieTheme || "Theme could not be determined (LLM response missing theme).",
                };

                if (movieDataWithTheme.Response === "False" && !movieDataWithTheme.Error) {
                    movieDataWithTheme.Error = "LLM indicated movie not found or error simulating data.";
                }

                if (movieDataWithTheme.Response === "False") {
                    console.warn(`  LLM indicated problem for title "${refinedTitle}": ${movieDataWithTheme.Error}`);
                } else {
                    console.log(`  LLM successfully processed data and theme for: "${movieDataWithTheme.Title}"`);
                }

            } catch (error) {
                console.error(`  Critical error processing data/theme for "${refinedTitle}" with LLM: ${error.message}. Raw LLM output might have been: ${extractJsonFromString(error.llmOutput || "")}`);
                movieDataWithTheme.Error = `LLM processing error: ${error.message}`;
                movieDataWithTheme.Response = "False";
                movieDataWithTheme.Plot = "N/A";
                movieDataWithTheme.MovieTheme = "Theme could not be determined due to processing error.";
            }
        }
        
        const movieDataForNextStep = {
            title: movieDataWithTheme.Title,
            year: movieDataWithTheme.Year || "N/A",
            imdbRating: movieDataWithTheme.imdbRating || "N/A",
            mainCast: movieDataWithTheme.Actors || "N/A",
            genre: movieDataWithTheme.Genre || "N/A",
            plotSummary: movieDataWithTheme.Plot || "N/A",
            Response: movieDataWithTheme.Response,
            error: movieDataWithTheme.Error
        };

        return { ...inputObject, movieDataFromOMDB: movieDataForNextStep, generatedTheme: movieDataWithTheme.MovieTheme };

    }).withConfig({ runName: "MovieDataAndThemeAgentStep" });

    return movieDataAndThemeAgentRunnable;
}