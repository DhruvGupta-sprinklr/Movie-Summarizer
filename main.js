import 'dotenv/config';
import { RunnableSequence, RunnableBranch, RunnableLambda } from "@langchain/core/runnables";
import { llm } from './llm_config.js';
import { createTitleRefinementAgent } from './agents/titleRefinementAgent.js';
import { createMovieDataAndThemeAgent } from './agents/movieDataAndThemeAgent.js';
import { createFileWriterAgent } from './agents/fileWriterAgent.js';

const OMDB_API_KEY_CHECK = process.env.OMDB_API_KEY;
const OUTPUT_DIR = 'movie_details';

if (!OMDB_API_KEY_CHECK) {
    console.warn('Warning: OMDB_API_KEY is not set in your .env file. The OMDB simulation prompt might be less effective.');
}

async function processMoviePipeline(rawUserMovieTitle) {
    if (!rawUserMovieTitle || rawUserMovieTitle.trim() === "") {
        console.error('Error: Please provide a movie name.');
        console.log('Usage: node main.js "Your Movie Title"');
        return;
    }
    console.log(`\n⚙️ Starting Movie Processing Pipeline for: "${rawUserMovieTitle}"`);

    const titleRefinementAgent = createTitleRefinementAgent(llm);
    const movieDataAndThemeAgent = createMovieDataAndThemeAgent(llm);
    const fileWriterAgent = createFileWriterAgent(llm, OUTPUT_DIR);

    const handleMovieNotFoundOrUncertain = RunnableLambda.from(async (input) => {
        const titleForMessage = input.movieDataFromOMDB?.title || input.refinedTitle || input.raw_title || "the provided movie";
        let reason = "The movie data/theme agent indicated the movie was not found or an error occurred.";

        if (input.titleIsUncertain) {
            reason = `The initial movie title ("${input.raw_title}") was too uncertain for reliable processing. Refined attempt: "${input.refinedTitle}".`;
        } else if (input.movieDataFromOMDB && input.movieDataFromOMDB.Error) {
            reason = `Reason from data/theme agent for "${titleForMessage}": ${input.movieDataFromOMDB.Error}`;
        } else if (input.movieDataFromOMDB?.Response === "False"){
             reason = `The data/theme agent could not find details for "${titleForMessage}" (Response: False).`;
        }

        console.warn(`\n Could not find or fully process details for: "${titleForMessage}".`);
        console.warn(`   Reason: ${reason}`);
        console.log("   No output file will be created for this movie.");

        return {
            ...input,
            finalMessage: `Processing halted for "${titleForMessage}": ${reason}`,
            writtenFilePath: null
        };
    }).withConfig({ runName: "HandleMovieNotFoundOrUncertainStep" });

    
    const isMovieNotFoundCondition = RunnableLambda.from(
        (input) => input.movieDataFromOMDB?.Response === "False"
    ).withConfig({ runName: "IsMovieNotFoundCondition" });


    const fullMovieProcessingSequence = RunnableSequence.from([
        titleRefinementAgent,
        movieDataAndThemeAgent,
        new RunnableBranch({
            branches: [
                [
                    isMovieNotFoundCondition, 
                    handleMovieNotFoundOrUncertain
                ]
            ],
            default: fileWriterAgent
        })
    ]);

    try {
        const initialInput = { raw_title: rawUserMovieTitle };
        const result = await fullMovieProcessingSequence.invoke(initialInput);

        console.log("\nMovie Processing Pipeline finished.");
        if (result && result.finalMessage) {
            console.log(" Final Status:", result.finalMessage);
            if (result.writtenFilePath) {
                console.log(" Output File:", result.writtenFilePath);
            }
        } else {
            console.log(" Sequence finished. Check logs for specific outcomes. This state should ideally not be reached if branches cover all paths.");
        }
    } catch (error) {
        console.error("\n Critical error during pipeline execution:", error);
        if (error.message && typeof error.input !== 'undefined' && error.runName) {
            console.error("  Error details:", { message: error.message, input: JSON.stringify(error.input, null, 2), step: error.runName });
        } else if (error.message && typeof error.input !== 'undefined') {
             console.error("  Error details:", { message: error.message, input: JSON.stringify(error.input, null, 2) });
        } else {
            console.error("  Error details:", error);
        }
    }
}

(async () => {
    const movieNameInput = process.argv.slice(2).join(" ").trim() || "Inception 2010";
    await processMoviePipeline(movieNameInput);
})();