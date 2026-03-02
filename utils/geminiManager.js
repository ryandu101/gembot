const { GoogleGenAI } = require('@google/genai');

class GeminiManager {
    constructor() {
        // Parse the comma-separated keys from the .env file
        this.keys = (process.env.GEMINI_KEYS || '').split(',').map(k => k.trim()).filter(k => k);
        this.currentIndex = 0;
        if (this.keys.length === 0) {
            throw new Error("No Gemini API keys found in .env");
        }
    }

    getClient() {
        return new GoogleGenAI({ apiKey: this.keys[this.currentIndex] });
    }

    rotateKey() {
        this.currentIndex = (this.currentIndex + 1) % this.keys.length;
        console.log(`Switched to API Key index ${this.currentIndex}`);
    }

    async generateContentWithRotation(promptParts, personaText) {
        let lastError = null;

        for (let attempt = 0; attempt < this.keys.length; attempt++) {
            try {
                const ai = this.getClient();
                
                // Construct the config based on the new SDK documentation
                const config = {
                    systemInstruction: personaText,
                    tools: [{ googleSearch: {} }], // Explicitly enables Grounding with Google Search
                    safetySettings: [
                        { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_NONE" },
                        { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_NONE" },
                        { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_NONE" },
                        { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_NONE" }
                    ]
                };

                const response = await ai.models.generateContent({
                    model: 'gemini-2.5-flash',
                    contents: promptParts,
                    config: config
                });

                return response;

            } catch (error) {
                // If it's a 429 Too Many Requests/Quota Exhausted, rotate the key and try again
                if (error.status === 429) {
                    console.log(`Key index ${this.currentIndex} is rate limited/exhausted.`);
                    lastError = error;
                    this.rotateKey();
                } else {
                    // Re-throw non-retriable errors
                    throw error;
                }
            }
        }
        
        throw new Error(`All Gemini API keys exhausted. Last error: ${lastError?.message}`);
    }
}

module.exports = new GeminiManager();