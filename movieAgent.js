import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import {
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
} from '@langchain/core/prompts';
import { LLMChain } from 'langchain/chains';
import { RunnableSequence, RunnableLambda } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

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
    return name
        .toLowerCase()
        .replace(/\s+/g, '_')
        .replace(/[^a-z0-9_\-\.]/gi, '')
        .replace(/\.+/g, '.')
        .slice(0, 100);
}

function extractJsonFromString(str) {
    if (typeof str !== 'string') {
        console.warn("extractJsonFromString: input was not a string, returning as is.", str);
        return str;
    }
   
    const match = str.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    if (match && match[1]) {
        return match[1].trim(); 
    }
    return str.trim(); 
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


// --- 2. LLM-driven OMDB Data Simulation ---
const omdbSimulationPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(
`You are an API simulator. Your task is to act as the OMDB API.
Given a movie title, you must generate a JSON response that mimics a real OMDB API call.
The OMDB API base URL is: ${OMDB_BASE_URL}
The API key for context is: ${OMDB_API_KEY} (Do not include this in the response, just use for context)
The conceptual query structure is: ?apikey=YOUR_KEY&t=MOVIE_TITLE&plot=full

If the movie is found based on your knowledge, the JSON response should include:
- "Title": string (official movie title)
- "Year": string (e.g., "2010")
- "imdbRating": string (e.g., "8.8")
- "Actors": string (comma-separated list of main actors, ideally 3-5 names)
- "Genre": string (comma-separated list of genres)
- "Plot": string (full plot summary)
- "Response": "True"

If the movie is NOT found in your knowledge or if you simulate an error:
- "Response": "False"
- "Error": string (e.g., "Movie not found!" or "Error simulating data.")

IMPORTANT:
1.  Only output the JSON string. Do not include any other text, explanations, or markdown formatting (like \`\`\`json ... \`\`\`).
2.  Ensure the JSON is well-formed.
3.  For "Actors" and "Genre", provide comma-separated strings.
4.  If providing data, be as accurate as possible based on your training. If unsure, it's better to state "Movie not found!".
`
    ),
    HumanMessagePromptTemplate.fromTemplate(
        "Simulate OMDB API JSON response for the movie title: {refined_movie_title}"
    ),
]);

const omdbSimulationLlmChain = omdbSimulationPrompt.pipe(llm).pipe(new StringOutputParser());



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



const fileContentGenerationPrompt = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(
`You are an expert file creation assistant.
Your task is to take movie details and generate a JSON object containing:
1.  A "filename": A sanitized filename string (lowercase, spaces to underscores, remove most non-alphanumeric characters except underscore/hyphen, max 60 chars, ending with .txt). Example: 'the_matrix_reloaded.txt'.
2.  A "file_content": The fully formatted text content for the file.

The file content MUST follow this exact structure:
Movie Title: {{title}} ({{year}})
--------------------------------------
IMDb Rating:
  {{imdbRating}}

Main Cast:
  {{mainCast}}

Genre(s):
  {{genre}}

Movie Theme:
  {{movieTheme}}

Plot Summary:
  {{plotSummary}}

Instructions:
- Replace placeholders like {{title}} with the actual data.
- If a piece of data is 'N/A', 'Not Available', or empty, represent it as 'N/A' in the output content.
- The Movie Theme should be 'Theme could not be determined.' if the provided theme is empty, 'N/A', or indicates inability to determine.
- You MUST output a single, well-formed JSON object with the two keys: "filename" and "file_content".
- Do not include any other text, explanations, or markdown formatting (like \`\`\`json ... \`\`\`) around the JSON.
- Ensure the file_content string correctly uses newline characters (\\n) for line breaks where appropriate in the template.
`
    ),
    HumanMessagePromptTemplate.fromTemplate(
`Generate the JSON for filename and file content using these details:
Title: {title}
Year: {year}
IMDb Rating: {imdbRating}
Main Cast: {mainCast}
Genre: {genre}
Movie Theme: {movieTheme}
Plot Summary: {plotSummary}`
    ),
]);

const fileContentGenerationLlmChain = fileContentGenerationPrompt.pipe(llm).pipe(new StringOutputParser());


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
        const refinedTitle = result.refined_title_text ? result.refined_title_text.trim() : inputObject.raw_title;
        console.log(`  Input Raw Title: "${inputObject.raw_title}"`);
        console.log(`  LLM Refined Title: "${refinedTitle}"`);
        return { ...inputObject, refinedTitle };
    }).withConfig({ runName: "TitleRefinementStep" });

    const omdbSimulatorRunnable = RunnableLambda.from(async (inputObject) => {
        console.log("\n[LCEL Step 2: Simulating Movie Data Fetch via LLM]");
        const { refinedTitle } = inputObject;

        if (!refinedTitle || refinedTitle.startsWith("UNCERTAIN:")) {
            const message = `OMDB LLM Simulator: Cannot reliably simulate for title: ${refinedTitle}`;
            console.warn(`  ${message}`);
            return {
                ...inputObject,
                movieDataFromOMDB: {
                    error: message,
                    title: refinedTitle.replace("UNCERTAIN: ", ""),
                    plotSummary: "N/A", genre:"N/A", year:"N/A", imdbRating:"N/A", mainCast:"N/A", Response: "False"
                }
            };
        }

        try {
            const rawSimulatedJsonString = await omdbSimulationLlmChain.invoke({ refined_movie_title: refinedTitle });
            console.log(`  LLM Raw Simulated OMDB Response String: ${rawSimulatedJsonString}`);
            
            const cleanedJsonString = extractJsonFromString(rawSimulatedJsonString); // MODIFIED
            let data;
            try {
                data = JSON.parse(cleanedJsonString); // MODIFIED
            } catch (parseError) {
                console.error(`  Error parsing LLM's cleaned simulated JSON for "${refinedTitle}": ${parseError.message}. Cleaned: ${cleanedJsonString}. Raw: ${rawSimulatedJsonString}`);
                throw new Error(`Failed to parse LLM's simulated OMDB JSON. Cleaned content: ${cleanedJsonString}`);
            }

            if (data.Response === 'False') {
                const errorMsg = data.Error || "LLM indicated movie not found or error.";
                console.warn(`  LLM Simulated OMDB Error for title "${refinedTitle}": ${errorMsg}`);
                return {
                    ...inputObject,
                    movieDataFromOMDB: {
                        error: `LLM Sim: ${errorMsg}`,
                        title: data.Title || refinedTitle,
                        plotSummary: "N/A", genre:"N/A", year:"N/A", imdbRating:"N/A", mainCast:"N/A", Response: "False"
                    }
                };
            }
            const result = {
                title: data.Title || refinedTitle,
                year: data.Year || 'N/A',
                imdbRating: data.imdbRating || 'N/A',
                mainCast: (data.Actors ? String(data.Actors).split(',').map(actor => actor.trim()).slice(0,5).join(', ') : 'N/A') || 'N/A',
                genre: data.Genre || 'N/A',
                plotSummary: data.Plot || 'N/A',
                Response: "True"
            };
            console.log(`  [LLM OMDB Sim]: Successfully processed simulated data for: "${result.title} (${result.year || 'N/A'})"`);
            return { ...inputObject, movieDataFromOMDB: result };

        } catch (error) {
            console.error(`  [LLM OMDB Sim]: Critical error simulating OMDB for "${refinedTitle}":`, error);
            return {
                ...inputObject,
                movieDataFromOMDB: {
                    error: `LLM OMDB Sim critical error for title "${refinedTitle}": ${error.message}`,
                    title: refinedTitle,
                    plotSummary: "N/A", genre:"N/A", year:"N/A", imdbRating:"N/A", mainCast:"N/A", Response: "False"
                }
            };
        }
    }).withConfig({ runName: "OMDBLLMSimulatorStep" });

    const themeGeneratorRunnable = RunnableLambda.from(async (inputObject) => {
        console.log("\n[LCEL Step 3: Generating Movie Theme]");
        const { movieDataFromOMDB } = inputObject;

        if (movieDataFromOMDB.error || !movieDataFromOMDB.plotSummary || movieDataFromOMDB.plotSummary === "N/A" || movieDataFromOMDB.Response === "False") {
            const reason = movieDataFromOMDB.error || "missing plot/data from OMDB sim";
            console.warn(`  Skipping theme generation for "${movieDataFromOMDB.title || inputObject.refinedTitle}" due to: ${reason}.`);
            return { ...inputObject, generatedTheme: "Theme could not be determined due to prior issues or lack of plot details." };
        }

        try {
            const themeResult = await themeGenerationChain.invoke({
                title: movieDataFromOMDB.title,
                genre: movieDataFromOMDB.genre,
                plot: movieDataFromOMDB.plotSummary,
            });
            const theme = themeResult.movie_theme_text ? themeResult.movie_theme_text.trim() : "Theme could not be determined.";
            console.log(`  Generated Theme for "${movieDataFromOMDB.title}": "${theme}"`);
            return { ...inputObject, generatedTheme: theme };
        } catch (error) {
            console.error(`  Error during theme generation for "${movieDataFromOMDB.title}": ${error.message}`);
            return { ...inputObject, generatedTheme: "Theme generation failed due to an error." };
        }
    }).withConfig({ runName: "ThemeGenerationStep" });

    const fileWriterLLMRunnable = RunnableLambda.from(async (inputObject) => {
        console.log("\n[LCEL Step 4: LLM Generating File Content/Name & Writing File]");
        const { movieDataFromOMDB, generatedTheme, refinedTitle } = inputObject;

        const detailsForFileLLM = {
            title: movieDataFromOMDB.title || refinedTitle || "Unknown Movie",
            year: movieDataFromOMDB.year || 'N/A',
            imdbRating: movieDataFromOMDB.imdbRating || 'N/A',
            mainCast: movieDataFromOMDB.mainCast || 'N/A',
            genre: movieDataFromOMDB.genre || 'N/A',
            plotSummary: movieDataFromOMDB.plotSummary || 'N/A',
            movieTheme: generatedTheme || 'Theme could not be determined.',
        };

        if (movieDataFromOMDB.error && detailsForFileLLM.plotSummary === 'N/A') {
            console.warn(`  File for "${detailsForFileLLM.title}" will reflect data fetching issues from OMDB sim: ${movieDataFromOMDB.error}`);
        }

        try {
            const rawLlmJsonOutput = await fileContentGenerationLlmChain.invoke(detailsForFileLLM);
            console.log(`  LLM Raw File Details JSON String: ${rawLlmJsonOutput}`);
            
            const cleanedLlmJsonOutput = extractJsonFromString(rawLlmJsonOutput); // MODIFIED
            let fileDetails;
            try {
                fileDetails = JSON.parse(cleanedLlmJsonOutput); // MODIFIED
                if (!fileDetails.filename || typeof fileDetails.filename !== 'string' || !fileDetails.file_content || typeof fileDetails.file_content !== 'string') {
                    throw new Error("LLM response missing 'filename' or 'file_content', or they are not strings.");
                }
            } catch (parseError) {
                console.error(`  Error parsing LLM's cleaned file details JSON for "${detailsForFileLLM.title}": ${parseError.message}. Cleaned: ${cleanedLlmJsonOutput}. Raw: ${rawLlmJsonOutput}`);
                console.warn(`  Falling back to manual file naming for "${detailsForFileLLM.title}".`);
                const fallbackFilename = sanitizeFilename(detailsForFileLLM.title) + ".txt";
                const fallbackFileContent = `Movie Title: ${detailsForFileLLM.title} (${detailsForFileLLM.year})\n` +
                                           `--------------------------------------\n` +
                                           `Error: LLM failed to generate structured file content. Raw output was: ${rawLlmJsonOutput}\n` +
                                           `OMDB Sim Error: ${movieDataFromOMDB.error || 'N/A'}\n` +
                                           `Plot: ${detailsForFileLLM.plotSummary}\n` +
                                           `Theme: ${detailsForFileLLM.movieTheme}`;
                fileDetails = { filename: fallbackFilename, file_content: fallbackFileContent };
            }

            const finalFilename = sanitizeFilename(fileDetails.filename);
            const filePath = path.join(OUTPUT_DIR, finalFilename);

            await fs.mkdir(OUTPUT_DIR, { recursive: true });
            await fs.writeFile(filePath, fileDetails.file_content);

            const successMessage = `Successfully wrote movie details for "${detailsForFileLLM.title}" to: ${filePath} (content/name by LLM, fallback: ${fileDetails.filename === sanitizeFilename(detailsForFileLLM.title) + ".txt" && fileDetails.file_content.includes("LLM failed to generate")})`;
            console.log(`  [FileWriterLLM]: ${successMessage}`);
            return { finalMessage: successMessage, writtenData: { ...detailsForFileLLM, filename: finalFilename } };

        } catch (error) {
            const errorMsg = `[FileWriterLLM]: Error during LLM file generation or writing for "${detailsForFileLLM.title}": ${error.message}`;
            console.error(`  ${errorMsg}`);
            return {
                finalMessage: errorMsg,
                writtenData: { ...detailsForFileLLM, filename: sanitizeFilename(detailsForFileLLM.title) + "_error.txt" }
            };
        }
    }).withConfig({ runName: "LLMFileWriterStep" });


    const fullSequence = RunnableSequence.from([
        titleRefinerRunnable,
        omdbSimulatorRunnable,
        themeGeneratorRunnable,
        fileWriterLLMRunnable,
    ]);

    try {
        const result = await fullSequence.invoke({ raw_title: rawUserMovieTitle });
        console.log("\n LCEL Sequence completed.");
        if (result && result.finalMessage) {
            console.log(" Final Output from Sequence:", result.finalMessage);
        } else {
            console.log(" Sequence finished, but no final message was explicitly returned by the last step. Check logs for file path.");
        }
    } catch (error) {
        console.error("\n Critical error during LCEL sequence execution:", error);
        if (error.message && error.input) {
            console.error("  Error details:", { message: error.message, input: error.input, step: error.runName });
        }
    }
}

(async () => {
    const movieNameInput = process.argv.slice(2).join(" ") || "Inception";
    await processMovieWithLCELSequence(movieNameInput);
})();