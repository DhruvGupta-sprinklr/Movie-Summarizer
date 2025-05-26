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
    temperature: 0.3, 
});


function sanitizeFilename(name) {
    return name.replace(/[^a-z0-9_\-\s\.]/gi, '').replace(/\s+/g, '_');
}


class OMDBMovieInfoTool extends Tool {
    name = "OMDBMovieInfoTool";
    description = "Fetches detailed movie information (IMDb rating, cast, genre, plot) from the OMDB API given a movie title. Input should be the movie title.";

    async _call(movieTitle) {
        try {
            console.log(`OMDBTool: Searching for "${movieTitle}"...`);
            const apiUrl = `${OMDB_BASE_URL}?apikey=${OMDB_API_KEY}&t=${encodeURIComponent(movieTitle)}&plot=full`;
            const response = await axios.get(apiUrl);
            const data = response.data;

            if (data.Response === 'False') {
                return `Error from OMDB: ${data.Error}`;
            }

            
            const imdbRating = data.imdbRating || 'N/A';
            const actors = data.Actors ? data.Actors.split(',').map(actor => actor.trim()) : [];
            const mainCast = actors.slice(0, 5).join(', ') || 'N/A'; 
            const genre = data.Genre || 'N/A';
            const plotSummary = data.Plot || 'N/A';
            const year = data.Year || 'N/A';
            const title = data.Title || movieTitle; 

           
            return JSON.stringify({
                title,
                year,
                imdbRating,
                mainCast,
                genre, 
                plotSummary
            });
        } catch (error) {
            console.error("Error in OMDBMovieInfoTool:", error);
            return `Failed to fetch movie data from OMDB. Error: ${error.message}`;
        }
    }
}

async function getAndStoreMovieDetails(movieName) {
    if (!movieName) {
        console.error('Error: Please provide a movie name.');
        console.log('Usage: node movieAgent.js "Your Movie Title"');
        return;
    }

    console.log(`Processing movie: "${movieName}"`);

    
    const omdbTool = new OMDBMovieInfoTool();
    let omdbDataString;
    try {
        omdbDataString = await omdbTool.call(movieName); 
    } catch (e) {
        console.error("Failed to execute OMDB Tool:", e);
        return;
    }

    let movieData;
    try {
        movieData = JSON.parse(omdbDataString);
        if (movieData.Error) { 
            console.error(`Error fetching details for "${movieName}": ${movieData.Error}`);
            return;
        }
    } catch (e) {
        console.error(`Error parsing OMDB data: ${omdbDataString}`, e);
        return;
    }


    console.log("\nFetched from OMDB:");
    console.log(`  Title: ${movieData.title} (${movieData.year})`);
    console.log(`  IMDb Rating: ${movieData.imdbRating}`);
    console.log(`  Main Cast (raw): ${movieData.mainCast}`);
    console.log(`  Genre (raw): ${movieData.genre}`);
    console.log(`  Plot Summary (raw): ${movieData.plotSummary.substring(0, 100)}...`);


    
    console.log("\n Asking Gemini to refine the 'Movie Theme'...");

    const themePrompt = ChatPromptTemplate.fromMessages([
        SystemMessagePromptTemplate.fromTemplate(
            "You are an expert movie analyst. Your task is to derive a concise 'Movie Theme' from the provided genre and plot summary. " +
            "The theme should be a short, descriptive phrase capturing the central idea, tone, or message, going beyond just listing genres. " +
            "For example, if genre is Action/Sci-Fi and plot involves dystopian future, theme could be 'Survival and rebellion in a technologically advanced dystopia'."
        ),
        HumanMessagePromptTemplate.fromTemplate(
            "Movie Title: {title}\nGenre(s): {genre}\nPlot Summary: {plot}\n\nDerive the Movie Theme:"
        ),
    ]);

    const themeChain = new LLMChain({
        llm: llm,
        prompt: themePrompt,
        verbose: false, 
    });

    let movieTheme = 'N/A';
    try {
        const themeResult = await themeChain.call({
            title: movieData.title,
            genre: movieData.genre,
            plot: movieData.plotSummary,
        });
        movieTheme = themeResult.text.trim();
    } catch (error) {
        console.error("Error getting theme from LLM:", error);
        movieTheme = `Could not derive theme (using genre: ${movieData.genre})`;
    }

    console.log(`  Derived Theme: ${movieTheme}`);

    
    const fileContent = `Movie Title: ${movieData.title} (${movieData.year})
--------------------------------------

 IMDb Rating:
   ${movieData.imdbRating}

 Main Cast:
   ${movieData.mainCast}

 Movie Theme:
   ${movieTheme}

 Plot Summary:
   ${movieData.plotSummary}
`;


    try {
        await fs.mkdir(OUTPUT_DIR, { recursive: true }); 
        const sanitizedMovieName = sanitizeFilename(movieData.title);
        const filePath = path.join(OUTPUT_DIR, `${sanitizedMovieName}.txt`);
        await fs.writeFile(filePath, fileContent);
        console.log(`\n Successfully wrote movie details to: ${filePath}`);
    } catch (error) {
        console.error(`Error writing file: ${error}`);
    }
}



(async () => {
    const movieNameInput = process.argv.slice(2).join(" ");
    await getAndStoreMovieDetails(movieNameInput);
})();