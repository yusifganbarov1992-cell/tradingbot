import { GoogleGenerativeAI } from "@google/generative-ai";

// Initialize the Gemini API client
// The API key must be obtained exclusively from the environment variable process.env.API_KEY
const genAI = new GoogleGenerativeAI(import.meta.env.VITE_GEMINI_API_KEY || "");

export const analyzeMarketSentiment = async (symbol: string, recentNews: string): Promise<string> => {
  // Ensure we don't proceed if key is missing, although initialization handles it.
  if (!import.meta.env.VITE_GEMINI_API_KEY) {
    console.error("API Key missing in import.meta.env.VITE_GEMINI_API_KEY");
    return "API Key configuration error. Please ensure VITE_GEMINI_API_KEY is set.";
  }

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });
    const prompt = `
      You are an expert financial crypto trading analyst. 
      Analyze the sentiment for ${symbol} based on the following news headlines/context:
      "${recentNews}"

      Provide a concise 3-sentence summary:
      1. Overall Sentiment (Bullish/Bearish/Neutral)
      2. Key Risk Factor
      3. Recommended Action (Accumulate/Distribute/Wait)
    `;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    return response.text() || "No analysis generated.";
  } catch (error) {
    console.error("Gemini API Error:", error);
    return "Error generating sentiment analysis. Please check API configuration.";
  }
};