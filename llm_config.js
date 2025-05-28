import 'dotenv/config';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

if (!GOOGLE_API_KEY) {
    console.error('Error: GOOGLE_API_KEY is not set in your .env file.');
    process.exit(1);
}

export const llm = new ChatGoogleGenerativeAI({
    apiKey: GOOGLE_API_KEY,
    model: 'gemini-1.5-flash-latest', 
    temperature: 0.1,
});