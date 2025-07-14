import discord
from discord.ext import commands
import os
import google.generativeai as genai
# Import the correct exception types
from google.generativeai.types import generation_types
# Import the core API exceptions to catch more error types
from google.api_core import exceptions as api_core_exceptions
# Import libraries for image processing and JSON
import io
import json
from PIL import Image

# --- ✅ NEW: Configuration Loading ---
# Define file paths for external configuration
KEYS_FILE = "keys.json"
PERSONA_FILE = "persona.txt"
HISTORY_FILE = "chat_history.json"

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

def load_persona(filepath):
    """Loads the persona from a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: The persona file '{filepath}' was not found.")
        exit()

# Load configurations from files
DISCORD_TOKEN, GEMINI_API_KEYS = load_keys(KEYS_FILE)
PERSONA = load_persona(PERSONA_FILE)
# --- END OF NEW CONFIGURATION LOGIC ---


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
def create_model():
    """Creates a new GenerativeModel instance with the current key and safety settings."""
    api_key = GEMINI_API_KEYS[current_api_key_index]
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        'gemini-2.5-flash',
        system_instruction=PERSONA,
        safety_settings=safety_settings
    )

model = create_model()

# --- Chat Session Management ---
chat_sessions = {}

# --- Chat History Persistence Functions ---
def save_chat_history(channel_id, history):
    """Saves the chat history for a specific channel to the JSON file."""
    try:
        with open(HISTORY_FILE, 'r') as f:
            all_histories = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_histories = {}

    serializable_history = []
    for content in history:
        text_parts = [part.text for part in content.parts if hasattr(part, 'text')]
        if text_parts:
            serializable_history.append({'role': content.role, 'parts': text_parts})

    all_histories[str(channel_id)] = serializable_history

    with open(HISTORY_FILE, 'w') as f:
        json.dump(all_histories, f, indent=4)

def load_chat_history(channel_id):
    """Loads the chat history for a specific channel from the JSON file."""
    try:
        with open(HISTORY_FILE, 'r') as f:
            all_histories = json.load(f)
        return all_histories.get(str(channel_id), [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []

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
    chat_sessions.clear()

@bot.event
async def on_ready():
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(e)
    print(f'Logged in as {bot.user}')
    print(f'Using API Key index {current_api_key_index}')
    print('------')

@bot.tree.command(name="gemini", description="Ask a question to the Gemini model, or continue the conversation.")
async def gemini(interaction: discord.Interaction, prompt: str = None, attachment: discord.Attachment = None):
    await interaction.response.defer()
    channel_id = interaction.channel.id

    history_prompt = ""
    try:
        history_prompt_list = []
        async for message in interaction.channel.history(limit=5, before=interaction.created_at):
            if message.author != bot.user:
                 history_prompt_list.append(f"<@{message.author.id}> said: {message.content}")
        history_prompt_list.reverse()
        history_prompt = "\n".join(history_prompt_list)
    except discord.Forbidden:
        print(f"Warning: Cannot read history in channel '{interaction.channel.name}' ({interaction.channel.id}).")
        pass
    except Exception as e:
        print(f"An unexpected error occurred while fetching history: {e}")
        pass
    
    final_prompt_parts = []
    if history_prompt:
        final_prompt_parts.append(f"Here is the recent conversation history:\n{history_prompt}\n")

    if prompt:
        final_prompt_parts.append(f"<@{interaction.user.id}> said: {prompt}")
    
    if attachment:
        if "image" in attachment.content_type:
            try:
                image_bytes = await attachment.read()
                img = Image.open(io.BytesIO(image_bytes))
                final_prompt_parts.append(img)
                if not prompt:
                     final_prompt_parts.insert(1, f"<@{interaction.user.id}> sent an image. Please describe or react to it.")
            except Exception as e:
                await interaction.followup.send(content=f"I had trouble reading that image file. Error: {e}")
                return
        else:
            await interaction.followup.send(content="Sorry, I can only process image files right now.", ephemeral=True)
            return

    if not final_prompt_parts:
        await interaction.followup.send(content="Please provide a prompt or an image.", ephemeral=True)
        return
    
    if channel_id not in chat_sessions:
        history = load_chat_history(channel_id)
        chat_sessions[channel_id] = model.start_chat(history=history)

    chat = chat_sessions[channel_id]
    print(f"Slash command prompt parts: {final_prompt_parts}")

    try:
        response = await chat.send_message_async(final_prompt_parts)
        if not response.parts:
            print("API response was empty. Rotating key.")
            rotate_api_key()
            await interaction.followup.send(content="There was a hiccup with the API. I've switched to a backup key, please try your prompt again.")
        else:
            await interaction.followup.send(content=response.text)
            save_chat_history(channel_id, chat.history)
    except (generation_types.BlockedPromptException, api_core_exceptions.InvalidArgument) as e:
        print(f"API Error: {e}. Rotating key.")
        rotate_api_key()
        await interaction.followup.send(content="There was a hiccup with the API. I've switched to a backup key, please try your prompt again.")
    except Exception as e:
        print(f"An unexpected API error occurred: {e}. Rotating key.")
        rotate_api_key()
        await interaction.followup.send(content="An unexpected API error occurred. I've switched to a backup key, please try again.")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    mention_triggers = ['gemini', 'gem']
    natural_trigger_phrases = [
        "how do you", "how does", "how do i", "what is the best way to", "what's the difference between",
        "what are the pros and cons", "what happens if", "can anyone explain", "can someone explain",
        "i don't understand why", "can someone tell me", "the reason for this is", "can someone help",
        "does anybody know", "does anyone have experience with", "has anyone tried", "i wonder why",
        "i wonder if", "is it possible to", "what if we", "i'm curious about", "what do you guys think about",
        "is it worth it to", "should i use", "any thoughts on"
    ]
    content_lower = message.content.lower()

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_reply_to_bot = message.reference and message.reference.resolved and message.reference.resolved.author == bot.user
    contains_mention = any(word in content_lower for word in mention_triggers)
    contains_natural_trigger = any(phrase in content_lower for phrase in natural_trigger_phrases)

    if is_dm or is_reply_to_bot or contains_mention or contains_natural_trigger:
        channel_id = message.channel.id
        
        prompt_parts = []
        if message.reference and isinstance(message.reference.resolved, discord.Message):
            original_message = message.reference.resolved
            prompt_parts.append(f"<@{original_message.author.id}> said: '{original_message.content}'\n\n"
                                f"<@{message.author.id}> replied: '{message.content}'")
        else:
            prompt_parts.append(f"<@{message.author.id}> said: {message.content}")

        for attachment in message.attachments:
            if "image" in attachment.content_type:
                try:
                    image_bytes = await attachment.read()
                    img = Image.open(io.BytesIO(image_bytes))
                    prompt_parts.append(img)
                except Exception as e:
                    print(f"Could not process image attachment: {e}")
            else:
                prompt_parts.append(f"\n(User also attached a file named '{attachment.filename}' that cannot be viewed.)")
        
        if not message.content.strip() and not any(isinstance(p, Image.Image) for p in prompt_parts):
             print("Ignoring message with no text or valid images.")
             return

        if channel_id not in chat_sessions:
            history = load_chat_history(channel_id)
            chat_sessions[channel_id] = model.start_chat(history=history)

        chat = chat_sessions[channel_id]
        print(f"Message prompt parts: {prompt_parts}")

        try:
            async with message.channel.typing():
                response = await chat.send_message_async(prompt_parts)
                if not response.parts:
                    print("API response was empty. Rotating key.")
                    rotate_api_key()
                    await message.reply("There was a hiccup with the API. I've switched to a backup key, please try your prompt again.")
                else:
                    await message.reply(response.text)
                    save_chat_history(channel_id, chat.history)
        except (generation_types.BlockedPromptException, api_core_exceptions.InvalidArgument) as e:
            print(f"API Error: {e}. Rotating key.")
            rotate_api_key()
            await message.reply("There was a hiccup with the API. I've switched to a backup key, please try your prompt again.")
        except Exception as e:
            print(f"An unexpected API error occurred: {e}. Rotating key.")
            rotate_api_key()
            await message.reply("An unexpected API error occurred. I've switched to a backup key, please try again.")

bot.run(DISCORD_TOKEN)
