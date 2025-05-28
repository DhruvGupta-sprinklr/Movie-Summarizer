import fs from 'fs/promises';
import path from 'path';
import {
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
} from '@langchain/core/prompts';
import { RunnableLambda } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { sanitizeFilename, extractJsonFromString } from '../utils.js';

const fileContentGenerationPromptTemplate = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(
`You are an expert file creation assistant.
Your task is to take movie details and generate a SINGLE JSON object containing:
1.  A "filename": A fully sanitized filename string (lowercase, spaces to underscores, remove most non-alphanumeric characters except underscore/hyphen, max 60 chars, ending with .txt). Example: 'the_matrix_reloaded.txt'.
2.  A "file_content": The fully formatted text content for the file.

The file content MUST follow this exact structure, with data placeholders filled:
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

Instructions for JSON generation:
- Replace placeholders like {{title}} with the actual data provided.
- If a piece of data is 'N/A', 'Not Available', empty, or indicates an error, represent it as 'N/A' in the output file_content string (unless the placeholder is for an error message itself).
- For "Movie Theme": if the provided theme is empty, 'N/A', or indicates inability to determine, use "Theme could not be determined." in the file_content.
- You MUST output a single, well-formed JSON object with exactly two keys: "filename" (string) and "file_content" (string).
- Do not include any other text, explanations, or markdown formatting (like \`\`\`json ... \`\`\`) around the JSON.
- The "file_content" string MUST correctly use newline characters (\\n) for line breaks as shown in the template.
- The "filename" you generate MUST be pre-sanitized according to the rules above.
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
Plot Summary: {plotSummary}
OMDB/Theme Error (if any, for context): {omdbError}` 
    ),
]);


async function nativeFileSystemWriteTool(filePath, content, outputDir) {
    await fs.mkdir(outputDir, { recursive: true });
    await fs.writeFile(filePath, content);
    console.log(`    [NativeFSWriteTool] Successfully wrote to: ${filePath}`);
}

export function createFileWriterAgent(llm, outputDir) {
   
    const llmJsonGenerationTool = fileContentGenerationPromptTemplate
        .pipe(llm)
        .pipe(new StringOutputParser());

    const fileWriterAgentRunnable = RunnableLambda.from(async (inputObject) => {
        console.log("\n[Agent: File Content Generation & Writing]");
        const { movieDataFromOMDB, generatedTheme, refinedTitle } = inputObject;

        const detailsForFileLlm = {
            title: movieDataFromOMDB.title || refinedTitle || "Unknown Movie",
            year: movieDataFromOMDB.year || 'N/A',
            imdbRating: movieDataFromOMDB.imdbRating || 'N/A',
            mainCast: movieDataFromOMDB.mainCast || 'N/A',
            genre: movieDataFromOMDB.genre || 'N/A',
            plotSummary: movieDataFromOMDB.plotSummary || 'N/A',
            movieTheme: generatedTheme || 'Theme could not be determined.',
            omdbError: movieDataFromOMDB.error || 'N/A' 
        };

        if (movieDataFromOMDB.Response === "False" || movieDataFromOMDB.error) {
            console.warn(`  File for "${detailsForFileLlm.title}" will reflect data fetching/theme issues: ${movieDataFromOMDB.error || 'OMDB Sim indicated failure.'}`);
        }

        let finalFilename;
        let finalFileContent;
        let wasLlmSuccessful = false;

        try {                           
            const rawLlmJsonOutput = await llmJsonGenerationTool.invoke(detailsForFileLlm);
            console.log(`  LLM Raw File Details JSON String: ${rawLlmJsonOutput}`);
            
            const cleanedLlmJsonOutput = extractJsonFromString(rawLlmJsonOutput);
            let fileDetailsJsonFromLlm;
            try {
                fileDetailsJsonFromLlm = JSON.parse(cleanedLlmJsonOutput);
                if (!fileDetailsJsonFromLlm.filename || typeof fileDetailsJsonFromLlm.filename !== 'string' ||
                    !fileDetailsJsonFromLlm.file_content || typeof fileDetailsJsonFromLlm.file_content !== 'string') {
                    throw new Error("LLM JSON response missing 'filename' or 'file_content', or they are not strings.");
                }
                
                finalFilename = sanitizeFilename(fileDetailsJsonFromLlm.filename);
                finalFileContent = fileDetailsJsonFromLlm.file_content;
                wasLlmSuccessful = true;
            } catch (parseError) {
                console.error(`  Error parsing LLM's cleaned file details JSON for "${detailsForFileLlm.title}": ${parseError.message}. Cleaned: ${cleanedLlmJsonOutput}.`);
                
            }
        } catch (llmError) {
            console.error(`  Error invoking LLM for file content generation for "${detailsForFileLlm.title}": ${llmError.message}`);
          
        }

        if (!wasLlmSuccessful) {
            console.warn(`  Falling back to manually constructed filename and content for "${detailsForFileLlm.title}".`);
            finalFilename = sanitizeFilename(detailsForFileLlm.title) + ".txt";
            
            finalFileContent = `Movie Title: ${detailsForFileLlm.title} (${detailsForFileLlm.year})\n` +
                               `--------------------------------------\n` +
                               `IMDb Rating:\n  ${detailsForFileLlm.imdbRating}\n\n` +
                               `Main Cast:\n  ${detailsForFileLlm.mainCast}\n\n` +
                               `Genre(s):\n  ${detailsForFileLlm.genre}\n\n` +
                               `Movie Theme:\n  ${(detailsForFileLlm.movieTheme && !detailsForFileLlm.movieTheme.toLowerCase().includes("could not be determined")) ? detailsForFileLlm.movieTheme : "Theme could not be determined."}\n\n` +
                               `Plot Summary:\n  ${detailsForFileLlm.plotSummary}\n\n` +
                               `--- Fallback Content Note ---\n` +
                               `This content was generated due to an issue with the LLM producing the structured file details.\n` +
                               `OMDB/Theme Agent Error (if any): ${detailsForFileLlm.omdbError}\n`;
        }

        const filePath = path.join(outputDir, finalFilename);
        try {
            await nativeFileSystemWriteTool(filePath, finalFileContent, outputDir); 
            const successMessage = `Successfully wrote movie details for "${detailsForFileLlm.title}" to: ${filePath} (LLM generated content/name: ${wasLlmSuccessful})`;
            console.log(`  ${successMessage}`);
            return { ...inputObject, finalMessage: successMessage, writtenFilePath: filePath };
        } catch (writeError) {
             const errorMsg = `Error writing file "${filePath}" for "${detailsForFileLlm.title}": ${writeError.message}`;
            console.error(`  ${errorMsg}`);
            return { ...inputObject, finalMessage: errorMsg, writtenFilePath: filePath }; 
        }

    }).withConfig({ runName: "FileWriterAgentStep" });

    return fileWriterAgentRunnable;
}