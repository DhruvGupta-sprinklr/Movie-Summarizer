import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import axios from 'axios';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import {
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
} from '@langchain/core/prompts';
import { LLMChain } from 'langchain/chains';
import { Tool } from '@langchain/core/tools';

const OMDB_API_KEY = process.env.OMDB_API_KEY;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const OMDB_BASE_URL = 'http://www.omdbapi.com/';
const OUTPUT_DIR = 'movie_details';

if (!OMDB_API_KEY) {
    console.error('Error: OMDB_API_KEY is not set in your .env file.');
    process.exit(1);
}
if (!GOOGLE_API_KEY) {
    console.error('Error: GOOGLE_API_KEY is not set in your .env file.');
    process.exit(1);
}

const llm = new ChatGoogleGenerativeAI({
    apiKey: GOOGLE_API_KEY,
    model: 'gemini-1.5-flash-latest',
    temperature: 0.2, 
});

function sanitizeFilename(name) {
    if (!name || typeof name !== 'string') return 'untitled_movie';
    return name.replace(/[^a-z0-9_\-\s\.]/gi, '').replace(/\s+/g, '_').toLowerCase();
}

// --- AGENT 1 (Chain): Title Refinement ---
const titleRefinementPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(
        "You are an expert movie title refiner. Given a user's raw movie title, your goal is to return the most likely official movie title. " +
        "Correct common misspellings, understand context (like 'bollywood', 'hollywood', release year clues if any), and disambiguate. " +
        "For example:\n" +
        "- 'uri bollywood' should become 'Uri: The Surgical Strike'\n" +
        "- 'inceptio' should become 'Inception'\n" +
        "- 'the surgical strike uri' should become 'Uri: The Surgical Strike'\n" +
        "- 'terminator salvation' should become 'Terminator Salvation'\n" +
        "If the input is already a clear and official-looking title, return it as is. " +
        "If you are highly uncertain or the input is too vague (e.g., just 'action movie'), return the original input and add a note like 'UNCERTAIN: Could not reliably refine title.' " +
        "Only return the refined title (or original with note if uncertain)."
    ),
    HumanMessagePromptTemplate.fromTemplate("Raw movie title: {raw_title}\nRefined Title:"),
]);

const titleRefinementChain = new LLMChain({
    llm: llm,
    prompt: titleRefinementPrompt,
    outputKey: "refined_title_text",
});

// --- AGENT 2 (Tool): Movie Data Fetcher ---
class OMDBMovieInfoTool extends Tool {
    name = "OMDBMovieInfoFetcher";
    description = "Fetches detailed movie information (IMDb rating, cast, genre, plot) from the OMDB API given a refined movie title. Input should be the refined movie title string.";

    async _call(refinedMovieTitle) {
        console.log(`  OMDBTool: Searching for refined title "${refinedMovieTitle}"...`);
        if (!refinedMovieTitle || refinedMovieTitle.startsWith("UNCERTAIN:")) {
            throw new Error(`Cannot search OMDB with uncertain title: ${refinedMovieTitle}`);
        }
        try {
            const apiUrl = `${OMDB_BASE_URL}?apikey=${OMDB_API_KEY}&t=${encodeURIComponent(refinedMovieTitle)}&plot=full`;
            const response = await axios.get(apiUrl);
            const data = response.data;

            if (data.Response === 'False') {
                throw new Error(`OMDB API Error for title "${refinedMovieTitle}": ${data.Error}`);
            }

            const imdbRating = data.imdbRating || 'N/A';
            const actors = data.Actors ? data.Actors.split(',').map(actor => actor.trim()) : [];
            const mainCast = actors.slice(0, 5).join(', ') || 'N/A';
            const genre = data.Genre || 'N/A';
            const plotSummary = data.Plot || 'N/A';
            const year = data.Year || 'N/A';
            const title = data.Title || refinedMovieTitle; 

            return JSON.stringify({ 
                title,
                year,
                imdbRating,
                mainCast,
                genre,
                plotSummary
            });
        } catch (error) {
            console.error(`  Error in OMDBMovieInfoTool with title "${refinedMovieTitle}":`, error.message);
            throw error; 
        }
    }
}

// --- AGENT 3 (Chain): Content Refinement (Theme) ---
const themeGenerationPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(
        "You are an expert movie analyst. Your task is to derive a concise 'Movie Theme' from the provided movie title, genre, and plot summary. " +
        "The theme should be a short, descriptive phrase (10-15 words max) capturing the central idea, tone, or dominant message, going beyond just listing genres. " +
        "For example, if genre is Action/Sci-Fi and plot involves dystopian future, a good theme could be 'Rebellion against oppressive technology in a dystopian future.' or 'The human spirit's fight for freedom against overwhelming odds.'"
    ),
    HumanMessagePromptTemplate.fromTemplate(
        "Movie Title: {title}\nGenre(s): {genre}\nPlot Summary: {plot}\n\nDerived Movie Theme:"
    ),
]);

const themeGenerationChain = new LLMChain({
    llm: llm,
    prompt: themeGenerationPrompt,
    outputKey: "movie_theme_text",
});

// --- AGENT 4 (Tool): File Writer ---
class FileWriterTool extends Tool {
  name = "MovieInfoFileWriter";
  description = "Writes movie info (title, year, rating, cast, theme, plot) to a text file.";

  async _call(movieData) {
    
    let dataObj = movieData;
    if (typeof movieData === "string") {
      try {
        dataObj = JSON.parse(movieData);
      } catch (e) {
        throw new Error(`MovieInfoFileWriter: Failed to parse JSON input: ${e.message}`);
      }
    }
    
    if (!dataObj.title) {
      throw new Error("MovieInfoFileWriter: Missing required field `title` on dataObj");
    }

    
    console.log(`  FileWriterTool: Preparing to write file for "${dataObj.title}"...`);
    const fileContent = `Movie Title: ${dataObj.title} (${dataObj.year || 'N/A'})
--------------------------------------

IMDb Rating:
  ${dataObj.imdbRating || 'N/A'}

Main Cast:
  ${dataObj.mainCast || 'N/A'}

Movie Theme:
  ${dataObj.movieTheme || 'N/A'}

Plot Summary:
  ${dataObj.plotSummary || 'N/A'}
`;
    await fs.mkdir(OUTPUT_DIR, { recursive: true });
    const sanitizedMovieName = sanitizeFilename(dataObj.title);
    const filePath = path.join(OUTPUT_DIR, `${sanitizedMovieName}.txt`);
    await fs.writeFile(filePath, fileContent);

    const successMessage = `Successfully wrote movie details to: ${filePath}`;
    console.log(`  ${successMessage}`);
    return successMessage;
  }
}



// --- Main Orchestration Logic ---
async function processMoviePipeline(rawUserMovieTitle) {
    if (!rawUserMovieTitle) {
        console.error('Error: Please provide a movie name.');
        console.log('Usage: node your_script_name.js "Your Movie Title"');
        return;
    }
    console.log(`\n Starting movie processing pipeline for raw title: "${rawUserMovieTitle}"`);

    let refinedTitle;
    let movieDataFromOMDB;
    let generatedTheme;

    try {
        // === Step 1: Refine Title ===
        console.log("\n[AGENT 1/CHAIN: Title Refinement]");
        const titleResult = await titleRefinementChain.call({ raw_title: rawUserMovieTitle });
        refinedTitle = titleResult.refined_title_text.trim(); 
        console.log(`  Input Raw Title: "${rawUserMovieTitle}"`);
        console.log(`  LLM Refined Title: "${refinedTitle}"`);

        if (refinedTitle.startsWith("UNCERTAIN:") || !refinedTitle) {
            console.warn(`   Title refinement was uncertain or failed. Original: "${rawUserMovieTitle}", Refined: "${refinedTitle}"`);
        
        }

        // === Step 2: Fetch Movie Data ===
        console.log("\n[AGENT 2/TOOL: Fetching Movie Data from OMDB]");
        const omdbTool = new OMDBMovieInfoTool();
        const omdbDataString = await omdbTool.call(refinedTitle); // Pass the refined title
        movieDataFromOMDB = JSON.parse(omdbDataString); // Parse the JSON string from the tool
        console.log(`  Successfully fetched data for: "${movieDataFromOMDB.title} (${movieDataFromOMDB.year})"`);
        console.log(`  IMDb Rating: ${movieDataFromOMDB.imdbRating}`);

        // === Step 3: Generate Theme ===
        console.log("\n[AGENT 3/CHAIN: Generating Movie Theme]");
        const themeResult = await themeGenerationChain.call({
            title: movieDataFromOMDB.title,
            genre: movieDataFromOMDB.genre,
            plot: movieDataFromOMDB.plotSummary,
        });
        generatedTheme = themeResult.movie_theme_text.trim();
        console.log(`  Generated Theme: "${generatedTheme}"`);

        // Combine all data for the file writer
        const finalMovieData = {
            ...movieDataFromOMDB,
            movieTheme: generatedTheme,
        };

        // === Step 4: Write to File ===
        console.log("\n[AGENT 4/TOOL: Writing Details to File]");
        const fileWriterTool = new FileWriterTool();
        const fileWriteResult = await fileWriterTool._call(finalMovieData); 
        console.log(`  File Writer Status: ${fileWriteResult}`);

        console.log("\n Pipeline completed successfully!");

    } catch (error) {
        console.error("\n Error during movie processing pipeline:");
        console.error(`  Message: ${error.message}`);
        if (refinedTitle) console.error(`  Refined Title at time of error: "${refinedTitle}"`);
        if (movieDataFromOMDB && movieDataFromOMDB.title) console.error(`  Movie being processed: "${movieDataFromOMDB.title}"`);
        
    }
}

(async () => {
    const movieNameInput = process.argv.slice(2).join(" ");
    await processMoviePipeline(movieNameInput);
})();