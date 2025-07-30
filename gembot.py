import discord
from discord.ext import commands
import os
import json
import asyncio

# --- Configuration Loading ---
# Get the absolute path of the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define file paths for external configuration using absolute paths
KEYS_FILE = os.path.join(SCRIPT_DIR, "keys.json")
PERSONA_FILE = os.path.join(SCRIPT_DIR, "persona.txt")
META_PERSONA_FILE = os.path.join(SCRIPT_DIR, "meta_persona.txt")
CHANNEL_IDS_FILE = os.path.join(SCRIPT_DIR, "channel_ids.json")
CHAT_HISTORY_FILE = os.path.join(SCRIPT_DIR, "chat_history.json")
USER_NOTES_FILE = os.path.join(SCRIPT_DIR, "user_notes.json")

# It's good practice to define constants for file paths
# so cogs can import them if needed.
CONFIG_FILES = {
    "keys": KEYS_FILE,
    "persona": PERSONA_FILE,
    "meta_persona": META_PERSONA_FILE,
    "channels": CHANNEL_IDS_FILE,
    "history": CHAT_HISTORY_FILE,
    "notes": USER_NOTES_FILE
}


def load_keys(filepath):
    """Loads Discord and Gemini keys from the keys.json file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            keys = json.load(f)
            discord_token = keys.get("discord_token")
            gemini_keys = keys.get("gemini_keys", [])
            if not discord_token or not gemini_keys:
                raise KeyError("Ensure 'discord_token' and 'gemini_keys' are present in keys.json")
            return discord_token, gemini_keys
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading {filepath}: {e}. Please ensure the file exists and is correctly formatted.")
        exit()

# --- Bot Setup ---
class Gembot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store file paths in the bot instance for easy access in cogs
        self.config_files = CONFIG_FILES
        self.script_dir = SCRIPT_DIR

    async def setup_hook(self):
        """The setup_hook is called when the bot logs in."""
        print("Running setup hook...")
        # Load all cogs from the 'cogs' directory
        for filename in os.listdir(os.path.join(self.script_dir, 'cogs')):
            if filename.endswith('.py') and not filename.startswith('__'):
                try:
                    await self.load_extension(f'cogs.{filename[:-3]}')
                    print(f'Successfully loaded cog: {filename}')
                except Exception as e:
                    print(f'Failed to load cog {filename}.')
                    print(f'[ERROR] {e}')
        
        # Sync commands
        try:
            # You can sync to a specific guild for faster testing
            # guild = discord.Object(id=YOUR_DEV_GUILD_ID)
            # await self.tree.sync(guild=guild)
            # print(f"Synced commands to development guild.")
            
            # Or sync globally
            await self.tree.sync()
            print("Synced commands globally.")

        except Exception as e:
            print(f"Error syncing commands: {e}")


async def main():
    """Main function to load configs and run the bot."""
    # Load critical tokens and keys
    discord_token, _ = load_keys(KEYS_FILE)

    # Set up intents
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True # Recommended for user-related commands

    # Create bot instance
    bot = Gembot(command_prefix="/", intents=intents)

    # Start the bot
    async with bot:
        await bot.start(discord_token)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot shut down by user.")
