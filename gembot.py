'''import discord
from discord.ext import commands
import os
import json
import asyncio

# --- Configuration Loading ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KEYS_FILE = os.path.join(SCRIPT_DIR, "keys.json")

# Define a dictionary of config file paths for easy access in cogs
CONFIG_FILES = {
    "keys": KEYS_FILE,
    "persona": os.path.join(SCRIPT_DIR, "persona.txt"),
    "meta_persona": os.path.join(SCRIPT_DIR, "meta_persona.txt"),
    "channels": os.path.join(SCRIPT_DIR, "channel_ids.json"),
    "history": os.path.join(SCRIPT_DIR, "chat_history.json"),
    "notes": os.path.join(SCRIPT_DIR, "user_notes.json")
}

def load_keys(filepath):
    """Loads the Discord token from the keys.json file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            keys = json.load(f)
            discord_token = keys.get("discord_token")
            if not discord_token:
                raise KeyError("'discord_token' not found in keys.json")
            return discord_token
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Critical Error loading {filepath}: {e}. The bot cannot start.")
        exit()

# --- Bot Definition ---
class Gembot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Attach config paths and script directory to the bot instance
        self.config_files = CONFIG_FILES
        self.script_dir = SCRIPT_DIR
        self.remove_command('help')

    async def setup_hook(self):
        """This hook is called after the bot logs in but before it is ready."""
        print("Running setup_hook: Loading cogs...")
        
        # Load all Python files in the 'cogs' directory as extensions
        cogs_dir = os.path.join(self.script_dir, 'cogs')
        for filename in os.listdir(cogs_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                try:
                    await self.load_extension(f'cogs.{filename[:-3]}')
                    print(f'✅ Successfully loaded cog: {filename}')
                except Exception as e:
                    print(f'❌ Failed to load cog {filename}.')
                    print(f'[ERROR] {e}')

    async def on_ready(self):
        """Called when the bot is fully logged in and ready."""
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('Bot is ready and online!')
        try:
            synced = await self.tree.sync()
            print(f"Synced {len(synced)} application commands globally.")
        except Exception as e:
            print(f"Error syncing commands: {e}")

# --- Main Execution ---
async def main():
    """Main function to configure and run the bot."""
    discord_token = load_keys(KEYS_FILE)

    # Define the bot's intents
    intents = discord.Intents.default()
    intents.message_content = True  # Required for on_message event
    intents.members = True          # Recommended for user-related data

    # Create and run the bot instance
    bot = Gembot(command_prefix="!", intents=intents)

    # Add a sync command that only the bot owner can use
    @bot.command(name='sync', description='Sync slash commands with Discord.')
    @commands.is_owner()
    async def sync(ctx: commands.Context):
        await ctx.send("Syncing commands...")
        try:
            synced = await bot.tree.sync()
            await ctx.send(f"Synced {len(synced)} commands successfully.")
            print(f"Synced {len(synced)} commands.")
        except Exception as e:
            await ctx.send(f"Error syncing commands: {e}")
            print(f"Error syncing commands: {e}")

    async with bot:
        await bot.start(discord_token)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("
Bot shut down by user.")''