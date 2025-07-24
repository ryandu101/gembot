import discord
from discord.ext import commands
import os
import google.generativeai as genai
# Import the correct exception types
from google.generativeai.types import generation_types
# Import the core API exceptions to catch more error types
from google.api_core import exceptions as api_core_exceptions
# Import libraries for image processing, JSON, regex, and datetime
import io
import json
import re
from PIL import Image
from datetime import datetime, timezone

# --- Configuration Loading ---
# Get the absolute path of the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define file paths for external configuration using absolute paths
KEYS_FILE = os.path.join(SCRIPT_DIR, "keys.json")
PERSONA_FILE = os.path.join(SCRIPT_DIR, "persona.txt")
USER_NOTES_FILE = os.path.join(SCRIPT_DIR, "user_notes.json")
META_PERSONA_FILE = os.path.join(SCRIPT_DIR, "meta_persona.txt")
CHAT_HISTORY_FILE = os.path.join(SCRIPT_DIR, "chat_history.json")
CHANNEL_IDS_FILE = os.path.join(SCRIPT_DIR, "channel_ids.json")

DEV_GUILD_ID = 977180402743672902 # Replace with your actual Server ID
# Set the short-term memory window (in conversation turns)
SHORT_TERM_MEMORY_TURNS = 500


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

def load_text_file(filepath):
    """Loads text content from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        exit()

def load_allowed_channels(filepath):
    """Loads the list of allowed channel IDs from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("allowed_channel_ids", [])
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not load {filepath}. The on_message handler will be disabled in all channels.")
        return []

# Load configurations from files
DISCORD_TOKEN, GEMINI_API_KEYS = load_keys(KEYS_FILE)
PERSONA = load_text_file(PERSONA_FILE)
META_PERSONA = load_text_file(META_PERSONA_FILE)
ALLOWED_CHANNEL_IDS = load_allowed_channels(CHANNEL_IDS_FILE)


# --- State Management for API Keys ---
current_api_key_index = 0

# --- Safety Settings Configuration ---
safety_settings = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
}

# --- Gemini API Setup ---
def create_model(persona=PERSONA):
    """Creates a new GenerativeModel instance with the current key and safety settings."""
    if not GEMINI_API_KEYS:
        print("Error: No Gemini API keys found in keys.json. Exiting.")
        exit()
    api_key = GEMINI_API_KEYS[current_api_key_index]
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        'gemini-2.5-pro',
        system_instruction=persona,
        safety_settings=safety_settings
    )

model = create_model()

# --- User Notes & History Persistence Functions ---
def load_user_notes(user_id):
    """Loads the notes for a specific user from the JSON file."""
    try:
        with open(USER_NOTES_FILE, 'r') as f:
            all_notes = json.load(f)
        return all_notes.get(str(user_id), "No notes on this user yet.")
    except (FileNotFoundError, json.JSONDecodeError):
        return "No notes on this user yet."

def save_user_notes(user_id, notes):
    """Saves the updated notes for a specific user to the JSON file."""
    try:
        with open(USER_NOTES_FILE, 'r') as f:
            all_notes = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_notes = {}
    
    all_notes[str(user_id)] = notes
    
    with open(USER_NOTES_FILE, 'w') as f:
        json.dump(all_notes, f, indent=4)

async def update_notes_with_gemini(user_id, conversation_summary):
    """Uses Gemini to intelligently update the notes for a user."""
    print(f"Updating notes for user {user_id}...")
    existing_notes = load_user_notes(user_id)
    
    meta_model = create_model(persona=META_PERSONA)
    
    update_prompt = (
        f"**Existing Notes on User <@{user_id}>:**\n{existing_notes}\n\n"
        f"**Recent Conversation Summary:**\n{conversation_summary}\n\n"
        "Please update the notes based on this new information. Keep it concise."
    )
    
    try:
        response = await model.generate_content_async(update_prompt)
        if response.text:
            new_notes = response.text
            save_user_notes(user_id, new_notes)
            print(f"Successfully updated notes for user {user_id}.")
        else:
            print(f"Meta-model failed to generate updated notes for user {user_id}.")
    except Exception as e:
        print(f"An error occurred during note update for user {user_id}: {e}")

# --- REWRITTEN CHAT HISTORY LOGIC ---
def log_to_chat_history(channel_id, user_author, user_timestamp, user_content, bot_response_text):
    """Logs the user prompt and bot response to chat_history.json using the new structured format."""
    try:
        with open(CHAT_HISTORY_FILE, 'r') as f:
            all_histories = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_histories = {}

    channel_id_str = str(channel_id)
    if channel_id_str not in all_histories:
        all_histories[channel_id_str] = []

    # Create the structured entry for the user's message
    user_entry = {
        "role": "user",
        "author_id": user_author.id,
        "author_name": user_author.display_name,
        "timestamp": user_timestamp.isoformat(),
        "content": user_content
    }
    all_histories[channel_id_str].append(user_entry)
    
    # Create the structured entry for the bot's response
    bot_entry = {
        "role": "model",
        "author_id": bot.user.id,
        "author_name": bot.user.display_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content": bot_response_text
    }
    all_histories[channel_id_str].append(bot_entry)

    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(all_histories, f, indent=4)

# --- Discord Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)

def rotate_api_key():
    """Switches to the next API key in the list and re-creates the model."""
    global current_api_key_index, model
    current_api_key_index = (current_api_key_index + 1) % len(GEMINI_API_KEYS)
    print(f"Switching to API Key index {current_api_key_index}")
    model = create_model()

# --- UPDATED CONTEXT BUILDING FUNCTION ---
def build_context_prompt(user_id, channel_id):
    """Builds the prompt context using long-term notes and short-term history from the new format."""
    user_notes = load_user_notes(user_id)
    
    try:
        with open(CHAT_HISTORY_FILE, 'r') as f:
            all_histories = json.load(f)
        
        channel_history = all_histories.get(str(channel_id), [])
        # Each "turn" is now one entry, but we still grab user + model pairs, so *2 is correct.
        recent_history = channel_history[-(SHORT_TERM_MEMORY_TURNS * 2):]
        
        history_lines = []
        for entry in recent_history:
            # Check for the new structured format
            if isinstance(entry, dict) and 'author_name' in entry and 'content' in entry:
                author = entry['author_name']
                content = entry['content'].replace('\n', '\n' + ' ' * (len(author) + 2)) # Indent multiline content
                history_lines.append(f"{author}: {content}")
            else:
                # Fallback for old format or malformed data, can be removed later
                print(f"Warning: Skipping malformed/old history entry in channel {channel_id}: {entry}")

        short_term_memory = "\n".join(history_lines)

    except (FileNotFoundError, json.JSONDecodeError):
        short_term_memory = "No chat history found."

    context_parts = [
        f"### My Long-Term Notes on <@{user_id}>:\n{user_notes}\n",
        f"### Recent Conversation (Short-Term Memory):\n{short_term_memory}\n"
    ]
    
    return context_parts

# --- Helper function to send long messages ---
async def send_long_message(ctx, text, files=None):
    """Splits a long message into chunks and sends them."""
    if not text:
        if isinstance(ctx, discord.Interaction):
            await ctx.followup.send(files=files)
        else:
            await ctx.reply(files=files)
        return

    chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
    for i, chunk in enumerate(chunks):
        # Send files only with the first chunk
        attached_files = files if i == 0 else None
        if isinstance(ctx, discord.Interaction):
            # For interactions, followup.send creates new messages
            if i == 0:
                 await ctx.followup.send(content=chunk, files=attached_files)
            else:
                 await ctx.channel.send(content=chunk)
        else:
            # For on_message, we can only reply once. Send subsequent chunks to the channel.
            if i == 0:
                await ctx.reply(chunk, files=attached_files)
            else:
                await ctx.channel.send(chunk)

# --- UPDATED on_ready ---
@bot.event
async def on_ready():
    try:
        # Sync commands globally first
        await bot.tree.sync()
        print("Synced commands globally.")

        # Then, sync to the development guild for immediate testing
        if DEV_GUILD_ID:
            guild = discord.Object(id=DEV_GUILD_ID)
            await bot.tree.sync(guild=guild)
            print(f"Synced commands to development guild: {DEV_GUILD_ID}")
            
    except Exception as e:
        print(e)
    print(f'Logged in as {bot.user}')
    print(f'Using API Key index {current_api_key_index}')
    print('------')

async def generate_with_full_rotation(prompt_parts):
    """Generates content, rotating through all available API keys on ResourceExhausted errors."""
    num_keys = len(GEMINI_API_KEYS)
    if num_keys == 0:
        raise ValueError("Cannot generate content: No Gemini API keys found.")
    last_exception = None
    for attempt in range(num_keys):
        try:
            print(f"Attempting API call with key index {current_api_key_index} (Attempt {attempt + 1}/{num_keys})")
            response = await model.generate_content_async(prompt_parts)
            return response
        except api_core_exceptions.ResourceExhausted as e:
            print(f"Key index {current_api_key_index} is over quota.")
            last_exception = e
            rotate_api_key()
        except Exception as e:
            print(f"Encountered a non-retriable error: {e}")
            raise e
    print("All available API keys are exhausted.")
    if last_exception:
        raise last_exception

async def process_and_send_response(ctx, response):
    """Processes model response for text, code, and images, then sends to Discord."""
    if not response or not response.parts:
        return "" 

    files_to_send = []
    
    image_parts = [p for p in response.parts if hasattr(p, 'inline_data') and p.inline_data.data]
    if image_parts:
        for i, part in enumerate(image_parts):
            image_data = part.inline_data.data
            files_to_send.append(discord.File(io.BytesIO(image_data), filename=f"generated_image_{i+1}.png"))
    
    full_text = "".join([p.text for p in response.parts if p.text])
    
    if files_to_send:
        bot_response_text = re.sub(r'\[Image of[^\]]+\]', '', full_text).strip()
        if not bot_response_text:
            bot_response_text = "Here is the image you requested!"
    else:
        code_block_match = re.search(r"```(?:\w+)?\n([\s\S]+?)\n```", full_text)
        if code_block_match:
            code_content = code_block_match.group(1)
            lang_match = re.search(r"```(\w+)", full_text)
            extension = lang_match.group(1) if lang_match else "txt"
            
            code_file = io.BytesIO(code_content.encode('utf-8'))
            files_to_send.append(discord.File(code_file, filename=f"generated_code.{extension}"))
            
            bot_response_text = "I've generated the code you asked for and attached it as a file."
        else:
            bot_response_text = full_text

    await send_long_message(ctx, bot_response_text, files=files_to_send if files_to_send else None)
    return bot_response_text

@bot.tree.command(name="gemini", description="Ask a question to the Gemini model, or continue the conversation.")
async def gemini(interaction: discord.Interaction, prompt: str = None, attachment: discord.Attachment = None):
    await interaction.response.defer()
    user_id = interaction.user.id
    channel_id = interaction.channel.id
    
    final_prompt_parts = build_context_prompt(user_id, channel_id)
    user_content_for_log = prompt if prompt else ""

    if prompt:
        final_prompt_parts.append(f"\n{interaction.user.display_name} said: {prompt}")
    
    if attachment:
        user_content_for_log += f" [Attached file: {attachment.filename}]"
        if "image" in attachment.content_type:
            try:
                image_bytes = await attachment.read()
                img = Image.open(io.BytesIO(image_bytes))
                final_prompt_parts.append(img)
                if not prompt:
                     final_prompt_parts.append(f"\n{interaction.user.display_name} sent an image. Please describe or react to it.")
            except Exception as e:
                await interaction.followup.send(content=f"I had trouble reading that image file. Error: {e}")
                return
        else:
            final_prompt_parts.append(f"\n(User also attached a file named '{attachment.filename}' that cannot be viewed.)")

    if not prompt and not attachment:
        final_prompt_parts.append(f"\n{interaction.user.display_name} has continued the conversation without adding a new message.")
        user_content_for_log = "[Continuation without text]"
    
    print(f"{interaction.user.display_name} ({user_id}) used /gemini in channel {channel_id}")

    try:
        response = await generate_with_full_rotation(final_prompt_parts)
        
        if not response.parts:
            await interaction.followup.send(content="My response was blocked or empty. This might be due to the prompt or safety filters.")
            return

        final_bot_text = await process_and_send_response(interaction, response)
        
        # Use the new logging function
        log_to_chat_history(channel_id, interaction.user, interaction.created_at, user_content_for_log.strip(), final_bot_text)

        summary = f"User Prompt: '{user_content_for_log.strip()}'\nBot Response: '{final_bot_text}'"
        await update_notes_with_gemini(user_id, summary)
            
    except api_core_exceptions.ResourceExhausted as e:
        await interaction.followup.send(content=f"I'm sorry, but all of my available API keys are currently over quota. Please try again in a few minutes.\n`{e}`")
    except (generation_types.BlockedPromptException, api_core_exceptions.InvalidArgument) as e:
        await interaction.followup.send(content=f"There was an issue with the prompt (it might be too long, empty, or blocked).\n`{e}`")
    except Exception as e:
        await interaction.followup.send(content=f"An unexpected error occurred: `{e}`")


@bot.event
async def on_message(message):
    if message.guild and message.channel.id not in ALLOWED_CHANNEL_IDS or message.author == bot.user:
        return

    mention_triggers = ['gemini', 'gem']
    natural_trigger_phrases = [
        "how do you", "how does", "how do i", "what is the best way to", "what's the difference between",
        "what are the pros and cons", "what happens if", "can anyone explain", "can someone explain",
        "i don't understand why", "can someone tell me", "can someone help", "does anybody know", 
        "i wonder why", "i wonder if", "is it possible to", "what if we", "i'm curious about", 
        "what do you guys think about", "is it worth it to", "should i use", "any thoughts on"
    ]
    content_lower = message.content.lower()

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_reply_to_bot = message.reference and message.reference.resolved and message.reference.resolved.author == bot.user
    contains_mention = any(word in content_lower for word in mention_triggers)
    contains_natural_trigger = any(content_lower.startswith(phrase) for phrase in natural_trigger_phrases)

    if is_dm or is_reply_to_bot or contains_mention or contains_natural_trigger:
        user_id = message.author.id
        channel_id = message.channel.id
        
        prompt_parts = build_context_prompt(user_id, channel_id)
        
        if message.reference and isinstance(message.reference.resolved, discord.Message):
            original_message = message.reference.resolved
            prompt_parts.append(f"\nContext: In reply to a message from {original_message.author.display_name} that said: '{original_message.content}'")
            # Image handling for replies...
            prompt_parts.append(f"\n\n{message.author.display_name} replied: '{message.content}'")
        else:
            prompt_parts.append(f"\n{message.author.display_name} said: {message.content}")

        for attachment in message.attachments:
            if "image" in attachment.content_type:
                try:
                    image_bytes = await attachment.read()
                    prompt_parts.append(Image.open(io.BytesIO(image_bytes)))
                except Exception as e:
                    print(f"Could not process new image: {e}")
            else:
                prompt_parts.append(f"\n(User also attached a file named '{attachment.filename}' that cannot be viewed.)")
        
        if not message.content.strip() and not any(isinstance(p, Image.Image) for p in prompt_parts):
             print("Ignoring message with no text or valid images.")
             return

        print(f"{message.author.display_name} ({user_id}) triggered bot in channel {channel_id}")

        try:
            async with message.channel.typing():
                response = await generate_with_full_rotation(prompt_parts)

                if not response.parts:
                    await message.reply("My response was blocked or empty. This might be due to the prompt or safety filters.")
                    return
                
                final_bot_text = await process_and_send_response(message, response)

                # Use the new logging function
                log_to_chat_history(channel_id, message.author, message.created_at, message.content, final_bot_text)

                summary = f"User Prompt: '{message.content}'\nBot Response: '{final_bot_text}'"
                await update_notes_with_gemini(user_id, summary)

        except api_core_exceptions.ResourceExhausted as e:
            await message.reply(f"I'm sorry, but all of my available API keys are currently over quota. Please try again in a few minutes.\n`{e}`")
        except (generation_types.BlockedPromptException, api_core_exceptions.InvalidArgument) as e:
            await message.reply(f"There was an issue with the prompt (it might be too long, empty, or blocked).\n`{e}`")
        except Exception as e:
            await message.reply(f"An unexpected error occurred: `{e}`")

# --- Owner-Only Commands ---

@bot.tree.command(name="build_chat_history", description="[Owner Only] Rebuilds the chat history for this channel from scratch.")
@commands.is_owner()
async def build_chat_history(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    channel_id_str = str(interaction.channel.id)
    print(f"Starting chat history rebuild for channel {channel_id_str}...")

    try:
        with open(CHAT_HISTORY_FILE, 'r') as f:
            all_histories = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_histories = {}

    new_channel_history = []
    message_count = 0
    print("Fetching channel message history...")
    
    async for message in interaction.channel.history(limit=10000, oldest_first=True):
        if not message.content and not message.attachments:
            continue # Skip messages with no text or attachments

        role = 'model' if message.author.bot else 'user'
        
        entry = {
            "role": role,
            "author_id": message.author.id,
            "author_name": message.author.display_name,
            "timestamp": message.created_at.isoformat(),
            "content": message.content
        }
        new_channel_history.append(entry)
        message_count += 1
        if message_count % 500 == 0:
            print(f"Processed {message_count} messages...")

    all_histories[channel_id_str] = new_channel_history

    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(all_histories, f, indent=4)
    
    print(f"Finished rebuilding history for channel {channel_id_str}.")
    await interaction.followup.send(f"Successfully rebuilt chat history for this channel, processing {message_count} messages.")

@bot.tree.command(name="initialize_notes", description="[Owner Only] Creates initial user notes from chat history.")
@commands.is_owner()
async def initialize_notes(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    print("Starting user notes initialization from new history format...")

    try:
        with open(CHAT_HISTORY_FILE, 'r') as f:
            all_histories = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        await interaction.followup.send(f"Error: The `{CHAT_HISTORY_FILE}` could not be read.")
        return

    user_conversations = {}

    for channel_id, messages in all_histories.items():
        for message in messages:
            # Ensure entry is valid and from a non-bot user
            if isinstance(message, dict) and message.get('role') == 'user':
                user_id = str(message.get('author_id'))
                if not user_id: continue
                
                if user_id not in user_conversations:
                    user_conversations[user_id] = []
                
                # Add the content to the user's conversation list
                content = message.get('content', '')
                if content:
                    user_conversations[user_id].append(content)

    if not user_conversations:
        await interaction.followup.send("No user conversations found in the chat history file.")
        return

    meta_model = create_model(persona=META_PERSONA)
    initialized_users = 0
    for user_id, texts in user_conversations.items():
        print(f"Generating notes for user {user_id}...")
        full_conversation = "\n".join(texts)
        
        prompt = (
            f"Please analyze the following conversation history for user <@{user_id}> and create a concise set of initial notes about them.\n\n"
            f"**Conversation History:**\n{full_conversation}"
        )
        
        try:
            response = await meta_model.generate_content_async(prompt)
            if response.text:
                save_user_notes(user_id, response.text)
                print(f"Successfully created notes for user {user_id}.")
                initialized_users += 1
            else:
                print(f"Failed to generate notes for user {user_id} (empty response).")
        except Exception as e:
            print(f"An error occurred while generating notes for user {user_id}: {e}")
            
    await interaction.followup.send(f"Initialization complete! Created or updated notes for {initialized_users} user(s).")


@bot.tree.command(name="condense_all_notes", description="[Owner Only] Condenses all user notes to reduce size.")
@commands.is_owner()
async def condense_all_notes(interaction: discord.Interaction):
    # This command does not need changes as it only interacts with user_notes.json
    await interaction.response.defer(ephemeral=True)
    # ... (rest of the function is unchanged)

@bot.tree.command(name="createnotes", description="[Owner Only] Creates/rewrites notes for a user from this channel's history.")
@commands.is_owner()
async def createnotes(interaction: discord.Interaction, user: discord.User):
    # This command does not need changes as it reads history directly
    await interaction.response.defer(ephemeral=True)
    # ... (rest of the function is unchanged)


bot.run(DISCORD_TOKEN)