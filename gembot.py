import discord
from discord.ext import commands
import os
import google.generativeai as genai
# Import the correct exception types
from google.generativeai.types import generation_types
# Import the core API exceptions to catch more error types
from google.api_core import exceptions as api_core_exceptions
# Import libraries for image processing, JSON, and regex
import io
import json
import re
from PIL import Image

# --- Configuration Loading ---
# Define file paths for external configuration
KEYS_FILE = "keys.json"
PERSONA_FILE = "persona.txt"
USER_NOTES_FILE = "user_notes.json"
META_PERSONA_FILE = "meta_persona.txt"
# ✅ Re-introduced the chat history file for permanent logging
CHAT_HISTORY_FILE = "chat_history.json"
DEV_GUILD_ID = 123456789012345678 # Replace with your actual Server ID

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

# Load configurations from files
DISCORD_TOKEN, GEMINI_API_KEYS = load_keys(KEYS_FILE)
PERSONA = load_text_file(PERSONA_FILE)
META_PERSONA = load_text_file(META_PERSONA_FILE)


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
        'gemini-2.5-flash',
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
        response = await meta_model.generate_content_async(update_prompt)
        if response.text:
            new_notes = response.text
            save_user_notes(user_id, new_notes)
            print(f"Successfully updated notes for user {user_id}.")
        else:
            print(f"Meta-model failed to generate updated notes for user {user_id}.")
    except Exception as e:
        print(f"An error occurred during note update for user {user_id}: {e}")

# ✅ NEW: Function to log raw conversations for archival
def log_to_chat_history(channel_id, user_prompt_parts, bot_response_text):
    """Logs the raw user prompt and bot response to the chat_history.json file."""
    try:
        with open(CHAT_HISTORY_FILE, 'r') as f:
            all_histories = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_histories = {}

    channel_id_str = str(channel_id)
    if channel_id_str not in all_histories:
        all_histories[channel_id_str] = []

    # Format user prompt for logging (text parts only)
    user_log_parts = [part for part in user_prompt_parts if isinstance(part, str)]
    
    # Append the user message and bot response
    all_histories[channel_id_str].append({'role': 'user', 'parts': user_log_parts})
    all_histories[channel_id_str].append({'role': 'model', 'parts': [bot_response_text]})

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

@bot.event
async def on_ready():
    try:
        if DEV_GUILD_ID:
            guild = discord.Object(id=DEV_GUILD_ID)
            bot.tree.copy_global_to(guild=guild)
            await bot.tree.sync(guild=guild)
            print(f"Synced commands to development guild: {DEV_GUILD_ID}")
        else:
            await bot.tree.sync()
            print("Synced commands globally.")
            
    except Exception as e:
        print(e)
        
    print(f'Logged in as {bot.user}')
    print(f'Using API Key index {current_api_key_index}')
    print('------')

@bot.tree.command(name="gemini", description="Ask a question to the Gemini model, or continue the conversation.")
async def gemini(interaction: discord.Interaction, prompt: str = None, attachment: discord.Attachment = None):
    await interaction.response.defer()
    user_id = interaction.user.id
    channel_id = interaction.channel.id
    
    user_notes = load_user_notes(user_id)
    final_prompt_parts = [f"Here are my notes on this user: {user_notes}\n\n"]

    if prompt:
        final_prompt_parts.append(f"<@{user_id}> said: {prompt}")
    
    if attachment:
        if "image" in attachment.content_type:
            try:
                image_bytes = await attachment.read()
                img = Image.open(io.BytesIO(image_bytes))
                final_prompt_parts.append(img)
                if not prompt:
                     final_prompt_parts.append(f"\n<@{user_id}> sent an image. Please describe or react to it.")
            except Exception as e:
                await interaction.followup.send(content=f"I had trouble reading that image file. Error: {e}")
                return
        else:
            await interaction.followup.send(content="Sorry, I can only process image files right now.", ephemeral=True)
            return

    if len(final_prompt_parts) <= 1 and not attachment:
        await interaction.followup.send(content="Please provide a prompt or an image.", ephemeral=True)
        return
    
    print(f"Slash command prompt parts: {final_prompt_parts}")

    try:
        response = await model.generate_content_async(final_prompt_parts)
        if not response.parts:
            print("API response was empty. Rotating key.")
            rotate_api_key()
            await interaction.followup.send(content="There was a hiccup with the API. I've switched to a backup key, please try your prompt again.")
        else:
            bot_response_text = response.text
            await interaction.followup.send(content=bot_response_text)
            # Log the raw interaction
            log_to_chat_history(channel_id, final_prompt_parts, bot_response_text)
            # Update the user's notes
            user_prompt_text = prompt or "Sent an image"
            summary = f"User Prompt: '{user_prompt_text}'\nBot Response: '{bot_response_text}'"
            await update_notes_with_gemini(user_id, summary)
            
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
        user_id = message.author.id
        channel_id = message.channel.id
        
        user_notes = load_user_notes(user_id)
        prompt_parts = [f"Here are my notes on this user: {user_notes}\n\n"]
        
        if message.reference and isinstance(message.reference.resolved, discord.Message):
            original_message = message.reference.resolved
            prompt_parts.append(f"<@{original_message.author.id}> said: '{original_message.content}'\n\n"
                                f"<@{user_id}> replied: '{message.content}'")
        else:
            prompt_parts.append(f"<@{user_id}> said: {message.content}")

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

        print(f"Message prompt parts: {prompt_parts}")

        try:
            async with message.channel.typing():
                response = await model.generate_content_async(prompt_parts)
                if not response.parts:
                    print("API response was empty. Rotating key.")
                    rotate_api_key()
                    await message.reply("There was a hiccup with the API. I've switched to a backup key, please try your prompt again.")
                else:
                    bot_response_text = response.text
                    await message.reply(bot_response_text)
                    # Log the raw interaction
                    log_to_chat_history(channel_id, prompt_parts, bot_response_text)
                    # Update the user's notes
                    summary = f"User Prompt: '{message.content}'\nBot Response: '{bot_response_text}'"
                    await update_notes_with_gemini(user_id, summary)

        except (generation_types.BlockedPromptException, api_core_exceptions.InvalidArgument) as e:
            print(f"API Error: {e}. Rotating key.")
            rotate_api_key()
            await message.reply("There was a hiccup with the API. I've switched to a backup key, please try your prompt again.")
        except Exception as e:
            print(f"An unexpected API error occurred: {e}. Rotating key.")
            rotate_api_key()
            await message.reply("An unexpected API error occurred. I've switched to a backup key, please try again.")

@bot.tree.command(name="initialize_notes", description="[Owner Only] Creates initial user notes from chat history.")
@commands.is_owner()
async def initialize_notes(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    print("Starting user notes initialization...")

    try:
        with open(CHAT_HISTORY_FILE, 'r') as f:
            old_history = json.load(f)
    except FileNotFoundError:
        await interaction.followup.send(f"Error: The `{CHAT_HISTORY_FILE}` was not found. No notes were created.")
        return
    except json.JSONDecodeError:
        await interaction.followup.send(f"Error: Could not read `{CHAT_HISTORY_FILE}`. It may be corrupted.")
        return

    user_conversations = {}
    user_id_pattern = re.compile(r"<@(\d+)>")

    for channel_id, messages in old_history.items():
        for message in messages:
            if message['role'] == 'user':
                full_text = " ".join(message['parts'])
                match = user_id_pattern.search(full_text)
                if match:
                    user_id = match.group(1)
                    if user_id not in user_conversations:
                        user_conversations[user_id] = []
                    user_conversations[user_id].append(full_text)

    if not user_conversations:
        await interaction.followup.send("No user conversations found in the chat history file.")
        return

    meta_model = create_model(persona=META_PERSONA)
    initialized_users = 0
    for user_id, texts in user_conversations.items():
        print(f"Generating notes for user {user_id}...")
        full_conversation = "\n".join(texts)
        
        prompt = (
            f"Please analyze the following conversation history for user <@{user_id}> and create a concise set of initial notes about their personality, "
            "interests, and key topics they discuss. Focus on creating a summary that would be useful for future interactions.\n\n"
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
            
    # ✅ Removed the "temporary" wording from the confirmation message
    await interaction.followup.send(f"Initialization complete! Created or updated notes for {initialized_users} user(s).")

bot.run(DISCORD_TOKEN)
