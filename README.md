You need to make your own persona.txt, meta_persona.txt, user_notes.json, chat_history.json, channel_ids.json, and keys.json files.
I recommend giving ai gembot.py and asking it how to help you set-up.

For persona.txt you can write whatever you want the persona prompt to be. Every api call sent will include these as persona instructions to Gemini.

Ask ai to help you create meta_persona.txt, what it does is it parses every message and is the persona that handles writing the user notes.

You can leave user_notes.json empty on creation. Same with chat_history.json

Channel ids are the discord channel ids that the bot is allowed to speak in.

Keys.json holds both your discord bot key, and your api keys.
You can store multiple api keys and if one fails, it will automatically attempt to use the next one. If all keys fail, it outputs the error message into the discord chat.
