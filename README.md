# 💎 Gembot - Advanced Gemini Discord Bot

Gembot is a powerful, context-aware Discord bot powered by Google's **Gemini 2.5 Flash** model. It features robust conversational memory, dynamic API key rotation, multi-modal attachment reading (images and text files), and automated long-term user note generation.

## ✨ Features

* **Gemini 2.5 Flash Integration:** Fast, high-quality responses with built-in Google Search grounding for real-time information.
* **Dual-Tier Memory System:**
    * *Short-Term Memory:* Remembers the last 1000 messages per channel for contextual conversations.
    * *Long-Term Memory:* Runs a background AI task to summarize and update persistent notes on individual users based on their interactions.
* **Multi-Modal Support:** Send images or text file attachments, and the bot will read and analyze them natively.
* **API Key Rotation:** Provide multiple Gemini API keys. If one hits a rate limit (429), the bot automatically rotates to the next key to ensure zero downtime.
* **Channel Whitelisting:** Restrict the bot to specific channels or allow it to operate globally via DMs and mentions.
* **Automatic Message Chunking:** Seamlessly handles responses longer than Discord's 2000-character limit.

---

## 🚀 Prerequisites

Before you begin, ensure you have the following installed and set up:
1.  **Node.js:** v18.0.0 or higher is required (v20+ recommended).
2.  **Discord Bot Token:** Create an application in the [Discord Developer Portal](https://discord.com/developers/applications).
    * **CRITICAL:** You *must* enable the **Message Content Intent**, **Server Members Intent**, and **Presence Intent** in the "Bot" tab of the developer portal.
3.  **Gemini API Key(s):** Obtain at least one API key from [Google AI Studio](https://aistudio.google.com/).

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd gembot
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Environment Variables (`.env`)
Create a file named `.env` in the root directory. This file is ignored by git to keep your credentials safe.
```env
# Your Discord Bot Token
DISCORD_TOKEN=your_discord_bot_token_here

# Comma-separated list of Gemini API keys for automatic rotation
GEMINI_KEYS=your_first_api_key,your_second_api_key,your_third_api_key
```

### 4. Create Configuration Files
The bot relies on several local files to manage state, memory, and permissions. Create the following files in the root directory of your project:

**`persona.txt`**
This dictates the main personality of your bot. Write your system prompt here.
> *Example:* `You are Gembot, a helpful, sarcastic, and highly intelligent AI assistant. Keep responses concise unless asked for details.`

**`meta_persona.txt`**
This dictates the personality of the *background task* that writes notes about users. 
> *Example:* `You are an analytical profiler. Summarize the user's personality, interests, and key facts based on the conversation history. Keep it under 2000 characters and use markdown.`

**`channel_ids.json`**
Defines which channels the bot is allowed to read and respond in. If you want the bot to only work in DMs, leave the array empty.
```json
{
  "allowed_channel_ids": [
    "123456789012345678",
    "987654321098765432"
  ]
}
```

**`chat_history.json`**
Initializes the short-term memory storage. Just create the file with an empty JSON object.
```json
{}
```

**`user_notes.json`**
Initializes the long-term user memory storage.
```json
{}
```

---

## 💻 Running the Bot

Once all dependencies are installed and configuration files are created, you can start the bot:

```bash
node index.js
```
*You should see `✅ Logged in as [YourBotName]!` in the console.*

---

## 💡 Tips for New Users

### How the Bot Triggers
To prevent spam, Gembot does not respond to every message. It will only respond if:
1.  You send it a Direct Message (DM).
2.  You explicitly `@mention` the bot in a whitelisted channel.
3.  You reply directly to one of the bot's messages.
4.  You use a trigger word defined in the code (currently set to `"gem"` or the bot's hardcoded ID).

### Managing Rate Limits
The Gemini API has rate limits (Requests Per Minute). If you have high traffic, Gembot is designed to handle this gracefully. Simply generate multiple free API keys from different Google accounts in AI Studio and add them all to the `GEMINI_KEYS` variable in your `.env` file, separated by commas. The bot will automatically cycle through them when one gets exhausted.

### Attachments & Multi-Modal Processing
You can drag and drop images or text files (like code snippets, `.txt`, or `.csv`) into Discord along with your prompt. The bot automatically intercepts these, converts images to base64, reads text files directly, and feeds the entire context into the Gemini model.

### Security Note
A `utils/securityWrapper.js` file is included in the codebase. If you plan to make this bot public, you should integrate this wrapper into `messageCreate.js` to sandbox user inputs and prevent prompt-injection attacks (where users try to force the bot to reveal its `persona.txt` or system instructions).

### Resetting Memory
If the bot gets confused or the memory files get too large:
* To wipe chat history: Replace the contents of `chat_history.json` with `{}`.
* To wipe user profiles: Replace the contents of `user_notes.json` with `{}`.
* *Note: Restart the bot after modifying these files manually.*
```
