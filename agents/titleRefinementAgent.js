import {
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
} from '@langchain/core/prompts';
import { LLMChain } from 'langchain/chains';
import { RunnableLambda } from "@langchain/core/runnables";

const titleRefinementPromptTemplate = ChatPromptTemplate.fromMessages([
    SystemMessagePromptTemplate.fromTemplate(
        "You are an expert movie title refiner. Given a user's raw movie title, your goal is to return the most likely official movie title. " +
        "Correct common misspellings, understand context (like 'bollywood', 'hollywood', release year clues if any), and disambiguate. " +
        "the name can also be reversed like pulp fiction van be written as fiction pulp"+
        "For example:\n" +
        "- 'uri bollywood' should become 'Uri: The Surgical Strike'\n" +
        "- 'inceptio' should become 'Inception'\n" +
        "If the input is already a clear and official-looking title, return it as is. " +
        "If you are highly uncertain or the input is too vague (e.g., just 'action movie'), return ONLY the original input text, followed by ' (UNCERTAIN)'. " + // Modified for simpler parsing
        "Your output should be JUST the refined title string (or original title + ' (UNCERTAIN)'). No other text."
    ),
    HumanMessagePromptTemplate.fromTemplate("Raw movie title: {raw_title}\nRefined Title:"),
]);

export function createTitleRefinementAgent(llm) {
    const titleRefinementLlmTool = new LLMChain({ 
        llm: llm,
        prompt: titleRefinementPromptTemplate,
        outputKey: "refined_title_text",
    });

    const titleRefinementAgentRunnable = RunnableLambda.from(async (inputObject) => {
        console.log("\n[Agent: Title Refinement]");
        const result = await titleRefinementLlmTool.invoke({ raw_title: inputObject.raw_title });
        let refinedTitle = result.refined_title_text ? result.refined_title_text.trim() : inputObject.raw_title;

        let isUncertain = false;
        if (refinedTitle.endsWith(" (UNCERTAIN)")) {
            refinedTitle = refinedTitle.replace(" (UNCERTAIN)", "").trim();
            isUncertain = true;
            console.log(`  LLM Refined Title (Uncertain): "${refinedTitle}"`);
        } else {
            console.log(`  LLM Refined Title: "${refinedTitle}"`);
        }

        console.log(`  Input Raw Title: "${inputObject.raw_title}"`);
        return { ...inputObject, refinedTitle, titleIsUncertain: isUncertain };
    }).withConfig({ runName: "TitleRefinementAgentStep" });

    return titleRefinementAgentRunnable;
}