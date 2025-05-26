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
import { RunnableSequence, RunnableLambda, RunnablePassthrough } from "@langchain/core/runnables";

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
    temperature: 0.1,
});

function sanitizeFilename(name) {
    if (!name || typeof name !== 'string') return 'untitled_movie';
    return name.replace(/[^a-z0-9_\-\s\.]/gi, '').replace(/\s+/g, '_').toLowerCase();
}

const titleRefinementPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(
        "You are an expert movie title refiner. Given a user's raw movie title, your goal is to return the most likely official movie title. " +
        "Correct common misspellings, understand context (like 'bollywood', 'hollywood', release year clues if any), and disambiguate. " +
        "For example:\n" +
        "- 'uri bollywood' should become 'Uri: The Surgical Strike'\n" +
        "- 'inceptio' should become 'Inception'\n" +
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

class OMDBMovieInfoTool extends Tool {
    name = "OMDBMovieInfoFetcher";
    description = "Fetches detailed movie information from OMDB. Input: refined movie title. Output: JSON string of movie data or error string.";

    async _call(refinedMovieTitle) {
        console.log(`  [OMDBTool]: Searching for refined title "${refinedMovieTitle}"...`);
        if (!refinedMovieTitle || refinedMovieTitle.startsWith("UNCERTAIN:")) {
            const message = `OMDBTool: Cannot reliably search OMDB with title: ${refinedMovieTitle}`;
            console.warn(`  ${message}`);
            return JSON.stringify({
                error: message,
                title: refinedMovieTitle.replace("UNCERTAIN: ", ""),
                plotSummary: "N/A", genre:"N/A", year:"N/A", imdbRating:"N/A", mainCast:"N/A"
            });
        }
        try {
            const apiUrl = `${OMDB_BASE_URL}?apikey=${OMDB_API_KEY}&t=${encodeURIComponent(refinedMovieTitle)}&plot=full`;
            const response = await axios.get(apiUrl);
            const data = response.data;

            if (data.Response === 'False') {
                const errorMsg = `OMDB API Error for title "${refinedMovieTitle}": ${data.Error}`;
                 console.warn(`  ${errorMsg}`);
                return JSON.stringify({
                    error: errorMsg,
                    title: refinedMovieTitle,
                    plotSummary: "N/A", genre:"N/A", year:"N/A", imdbRating:"N/A", mainCast:"N/A"
                });
            }
            const result = {
                title: data.Title || refinedMovieTitle,
                year: data.Year || 'N/A',
                imdbRating: data.imdbRating || 'N/A',
                mainCast: (data.Actors ? data.Actors.split(',').map(actor => actor.trim()).slice(0,5).join(', ') : 'N/A') || 'N/A',
                genre: data.Genre || 'N/A',
                plotSummary: data.Plot || 'N/A'
            };
            console.log(`  [OMDBTool]: Successfully fetched data for: "${result.title} (${result.year})"`);
            return JSON.stringify(result);
        } catch (error) {
            console.error(`  [OMDBTool]: Error with title "${refinedMovieTitle}":`, error.message);
            return JSON.stringify({
                error: `OMDBTool internal error for title "${refinedMovieTitle}": ${error.message}`,
                title: refinedMovieTitle,
                plotSummary: "N/A", genre:"N/A", year:"N/A", imdbRating:"N/A", mainCast:"N/A"
            });
        }
    }
}
const omdbTool = new OMDBMovieInfoTool();

const themeGenerationPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(
        "You are an expert movie analyst. Your task is to derive a concise 'Movie Theme' from the provided movie title, genre, and plot summary. " +
        "The theme should be a short, descriptive phrase (10-15 words max) capturing the central idea, tone, or dominant message. " +
        "If the plot is N/A or too brief, state 'Theme could not be determined due to lack of plot details.'"
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

class FileWriterTool extends Tool {
  name = "MovieInfoFileWriter";
  description = "Writes movie info to a text file. Input: JSON string of movie data including theme.";

  async _call(movieDataJsonString) {
    console.log(`  [FileWriterTool]: Preparing to write file...`);
    let dataObj;
    try {
        dataObj = JSON.parse(movieDataJsonString);
    } catch (e) {
        const errorMsg = `MovieInfoFileWriter: Failed to parse JSON input: ${e.message}. Input: ${movieDataJsonString}`;
        console.error(`  ${errorMsg}`);
        return errorMsg;
    }

    if (!dataObj.title) {
      const errorMsg = `MovieInfoFileWriter: Missing 'title' in data: ${JSON.stringify(dataObj)}`;
      console.error(`  ${errorMsg}`);
      return errorMsg;
    }
    if (dataObj.error && !dataObj.plotSummary) {
        const message = `FileWriterTool: Skipped writing for "${dataObj.title}" due to previous errors: ${dataObj.error}`;
        console.warn(`  ${message}`);
        return message;
    }

    const fileContent = `Movie Title: ${dataObj.title} (${dataObj.year || 'N/A'})
--------------------------------------
IMDb Rating:
  ${dataObj.imdbRating || 'N/A'}

Main Cast:
  ${dataObj.mainCast || 'N/A'}

Genre(s):
  ${dataObj.genre || 'N/A'}

Movie Theme:
  ${dataObj.movieTheme || 'Theme could not be determined.'}

Plot Summary:
  ${dataObj.plotSummary || 'N/A'}
`;
    try {
        await fs.mkdir(OUTPUT_DIR, { recursive: true });
        const sanitizedMovieName = sanitizeFilename(dataObj.title);
        const filePath = path.join(OUTPUT_DIR, `${sanitizedMovieName}.txt`);
        await fs.writeFile(filePath, fileContent);
        const successMessage = `Successfully wrote movie details for "${dataObj.title}" to: ${filePath}`;
        console.log(`  [FileWriterTool]: ${successMessage}`);
        return successMessage;
    } catch (error) {
        const errorMsg = `MovieInfoFileWriter: Error writing file for "${dataObj.title}": ${error.message}`;
        console.error(`  [FileWriterTool]: ${errorMsg}`);
        return errorMsg;
    }
  }
}
const fileWriterTool = new FileWriterTool();

async function processMovieWithLCELSequence(rawUserMovieTitle) {
    if (!rawUserMovieTitle) {
        console.error('Error: Please provide a movie name.');
        console.log('Usage: node your_script_name.js "Your Movie Title"');
        return;
    }
    console.log(`\n⚙️ Starting LCEL Sequential processing for: "${rawUserMovieTitle}"`);

    const titleRefinerRunnable = RunnableLambda.from(async (inputObject) => {
        console.log("\n[LCEL Step 1: Title Refinement]");
        const result = await titleRefinementChain.invoke({ raw_title: inputObject.raw_title });
        const refinedTitle = result.refined_title_text.trim();
        console.log(`  Input Raw Title: "${inputObject.raw_title}"`);
        console.log(`  LLM Refined Title: "${refinedTitle}"`);
        return { ...inputObject, refinedTitle: refinedTitle };
    }).withConfig({ runName: "TitleRefinementStep" });

    const omdbFetcherRunnable = RunnableLambda.from(async (inputObject) => {
        console.log("\n[LCEL Step 2: Fetching Movie Data from OMDB]");
        const omdbDataString = await omdbTool._call(inputObject.refinedTitle);
        let movieDataFromOMDB;
        try {
            movieDataFromOMDB = JSON.parse(omdbDataString);
        } catch (e) {
            console.error("  OMDB Data JSON parse error:", e, "Received:", omdbDataString);
            movieDataFromOMDB = {
                error: `Failed to parse OMDB response: ${omdbDataString}`,
                title: inputObject.refinedTitle,
                plotSummary: "N/A", genre:"N/A", year:"N/A", imdbRating:"N/A", mainCast:"N/A"
            };
        }
        if (movieDataFromOMDB.error) {
            console.warn(`  OMDB Fetch problem for "${inputObject.refinedTitle}": ${movieDataFromOMDB.error}`);
        } else {
            console.log(`  Successfully fetched/parsed data for: "${movieDataFromOMDB.title} (${movieDataFromOMDB.year})"`);
        }
        return { ...inputObject, movieDataFromOMDB };
    }).withConfig({ runName: "OMDBFetcherStep" });

    const themeGeneratorRunnable = RunnableLambda.from(async (inputObject) => {
        console.log("\n[LCEL Step 3: Generating Movie Theme]");
        const { movieDataFromOMDB } = inputObject;

        if (movieDataFromOMDB.error || !movieDataFromOMDB.plotSummary || movieDataFromOMDB.plotSummary === "N/A") {
            console.warn(`  Skipping theme generation for "${movieDataFromOMDB.title}" due to missing/error in movie data.`);
            return { ...inputObject, generatedTheme: "Theme could not be determined due to prior issues." };
        }

        const themeResult = await themeGenerationChain.invoke({
            title: movieDataFromOMDB.title,
            genre: movieDataFromOMDB.genre,
            plot: movieDataFromOMDB.plotSummary,
        });
        const theme = themeResult.movie_theme_text.trim();
        console.log(`  Generated Theme for "${movieDataFromOMDB.title}": "${theme}"`);
        return { ...inputObject, generatedTheme: theme };
    }).withConfig({ runName: "ThemeGenerationStep" });

    const fileWriterRunnable = RunnableLambda.from(async (inputObject) => {
        console.log("\n[LCEL Step 4: Writing Details to File]");
        const { movieDataFromOMDB, generatedTheme } = inputObject;

        const finalMovieDataForFile = {
            title: movieDataFromOMDB.title || inputObject.refinedTitle,
            year: movieDataFromOMDB.year || 'N/A',
            imdbRating: movieDataFromOMDB.imdbRating || 'N/A',
            mainCast: movieDataFromOMDB.mainCast || 'N/A',
            genre: movieDataFromOMDB.genre || 'N/A',
            plotSummary: movieDataFromOMDB.plotSummary || 'N/A',
            movieTheme: generatedTheme || 'Theme could not be determined.',
            omdbError: movieDataFromOMDB.error || null
        };
         if (finalMovieDataForFile.omdbError && !finalMovieDataForFile.plotSummary === 'N/A') {
            console.warn(`  File for "${finalMovieDataForFile.title}" will reflect data fetching issues.`);
        }

        const fileWriteResult = await fileWriterTool._call(JSON.stringify(finalMovieDataForFile));
        console.log(`  File Writer Status: ${fileWriteResult}`);
        return { finalMessage: fileWriteResult, writtenData: finalMovieDataForFile };
    }).withConfig({ runName: "FileWriterStep" });

    const fullSequence = RunnableSequence.from([
        titleRefinerRunnable,
        omdbFetcherRunnable,
        themeGeneratorRunnable,
        fileWriterRunnable,
    ]);

    try {
        const result = await fullSequence.invoke({ raw_title: rawUserMovieTitle });
        console.log("\n LCEL Sequence completed.");
        console.log("Final Output from Sequence:", result.finalMessage);
    } catch (error) {
        console.error("\n Critical error during LCEL sequence execution:", error);
    }
}

(async () => {
    const movieNameInput = process.argv.slice(2).join(" ") || "inceptio";
    await processMovieWithLCELSequence(movieNameInput);
})();