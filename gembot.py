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
CHAT_HISTORY_FILE = "chat_history.json"
DEV_GUILD_ID = 123456789012345678 # Replace with your actual Server ID
# Set the short-term memory window (in conversation turns)
SHORT_TERM_MEMORY_TURNS = 10 


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
    # Ensure the index is always valid
    safe_index = current_api_key_index % len(GEMINI_API_KEYS)
    api_key = GEMINI_API_KEYS[safe_index]
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
        response = await meta_model.generate_content_async(update_prompt)
        if response.text:
            new_notes = response.text
            save_user_notes(user_id, new_notes)
            print(f"Successfully updated notes for user {user_id}.")
        else:
            print(f"Meta-model failed to generate updated notes for user {user_id}.")
    except Exception as e:
        print(f"An error occurred during note update for user {user_id}: {e}")

def log_to_chat_history(channel_id, user_prompt_parts, bot_response_text, message_id=None):
    """Logs the raw user prompt and bot response to the chat_history.json file."""
    try:
        with open(CHAT_HISTORY_FILE, 'r') as f:
            all_histories = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_histories = {}

    channel_id_str = str(channel_id)
    if channel_id_str not in all_histories:
        all_histories[channel_id_str] = []

    user_log_parts = [part for part in user_prompt_parts if isinstance(part, str)]
    
    user_entry = {'role': 'user', 'parts': user_log_parts}
    if message_id:
        user_entry['id'] = message_id

    all_histories[channel_id_str].append(user_entry)
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

# --- Unified Context Building Function ---
def build_context_prompt(user_id, channel_id):
    """Builds the prompt context using long-term notes and short-term history."""
    user_notes = load_user_notes(user_id)
    
    try:
        with open(CHAT_HISTORY_FILE, 'r') as f:
            all_histories = json.load(f)
        
        channel_history = all_histories.get(str(channel_id), [])
        recent_history = channel_history[-(SHORT_TERM_MEMORY_TURNS * 2):]
        
        history_lines = []
        for entry in recent_history:
            role = "Gem" if entry['role'] == 'model' else "User"
            content = " ".join(entry['parts'])
            history_lines.append(f"{role}: {content}")
        
        short_term_memory = "\n".join(history_lines)

    except (FileNotFoundError, json.JSONDecodeError):
        short_term_memory = "No chat history found."

    context_parts = [
        f"### My Long-Term Notes on <@{user_id}>:\n{user_notes}\n",
        f"### Recent Conversation (Short-Term Memory):\n{short_term_memory}\n"
    ]
    
    return context_parts

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
    
    final_prompt_parts = build_context_prompt(user_id, channel_id)
    
    log_message = f"<@{user_id}> used /gemini"

    if prompt:
        final_prompt_parts.append(f"\n<@{user_id}> said: {prompt}")
        log_message += f" with prompt: '{prompt}'"
    
    if attachment:
        if "image" in attachment.content_type:
            try:
                image_bytes = await attachment.read()
                img = Image.open(io.BytesIO(image_bytes))
                final_prompt_parts.append(img)
                log_message += " and an image attachment."
                if not prompt:
                     final_prompt_parts.append(f"\n<@{user_id}> sent an image. Please describe or react to it.")
            except Exception as e:
                await interaction.followup.send(content=f"I had trouble reading that image file. Error: {e}")
                return
        else:
            await interaction.followup.send(content="Sorry, I can only process image files right now.", ephemeral=True)
            return

    if not prompt and not attachment:
        final_prompt_parts.append(f"\n<@{user_id}> has continued the conversation without adding a new message. Please respond based on the context.")
        log_message += " to continue the conversation."
    
    print(log_message)

    # --- ✅ NEW: Automatic Retry Logic ---
    response = None
    for attempt in range(len(GEMINI_API_KEYS)):
        try:
            response = await model.generate_content_async(final_prompt_parts)
            # If the call succeeds, break the loop
            break 
        except api_core_exceptions.ResourceExhausted as e:
            print(f"Quota exceeded on key {current_api_key_index}. Rotating key. Attempt {attempt + 1}/{len(GEMINI_API_KEYS)}")
            rotate_api_key()
            # If this was the last key, send a failure message
            if attempt == len(GEMINI_API_KEYS) - 1:
                await interaction.followup.send(content=f"All API keys have reached their quota. Please try again later.\n`{e}`")
                return
        except (generation_types.BlockedPromptException, api_core_exceptions.InvalidArgument) as e:
            print(f"Prompt-related API Error: {e}")
            await interaction.followup.send(content=f"There was an issue with the prompt (it might be too long, empty, or blocked).\n`{e}`")
            return # Don't retry for these errors
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            await interaction.followup.send(content=f"An unexpected error occurred: `{e}`")
            return # Don't retry for other unexpected errors

    # After the loop, check if we got a valid response
    if response and response.parts:
        bot_response_text = response.text
        await interaction.followup.send(content=bot_response_text)
        log_to_chat_history(channel_id, [log_message], bot_response_text)
        user_prompt_text = prompt or "Sent an image"
        summary = f"User Prompt: '{user_prompt_text}'\nBot Response: '{bot_response_text}'"
        await update_notes_with_gemini(user_id, summary)
    elif not response:
         # This case is hit if all keys failed with quota errors
         print("All API keys failed.")
         # The failure message is already sent inside the loop
    else: # Response was received but was empty
        await interaction.followup.send(content="My response was blocked or empty. This might be due to the prompt or safety filters.")


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
        
        prompt_parts = build_context_prompt(user_id, channel_id)
        
        log_message = f"<@{user_id}> said: '{message.content}'"
        
        if message.reference and isinstance(message.reference.resolved, discord.Message):
            original_message = message.reference.resolved
            prompt_parts.append(f"\n<@{original_message.author.id}> said: '{original_message.content}'")
            for attachment in original_message.attachments:
                if "image" in attachment.content_type:
                    try:
                        image_bytes = await attachment.read()
                        prompt_parts.append(Image.open(io.BytesIO(image_bytes)))
                        log_message += " (replying to an image)"
                    except Exception as e:
                        print(f"Could not process replied image: {e}")
            prompt_parts.append(f"\n\n<@{user_id}> replied: '{message.content}'")
        else:
            prompt_parts.append(f"\n<@{user_id}> said: {message.content}")

        for attachment in message.attachments:
            if "image" in attachment.content_type:
                try:
                    image_bytes = await attachment.read()
                    prompt_parts.append(Image.open(io.BytesIO(image_bytes)))
                    log_message += " (with an image attachment)"
                except Exception as e:
                    print(f"Could not process new image: {e}")
            else:
                prompt_parts.append(f"\n(User also attached a file named '{attachment.filename}' that cannot be viewed.)")
        
        if not message.content.strip() and not any(isinstance(p, Image.Image) for p in prompt_parts):
             print("Ignoring message with no text or valid images.")
             return

        print(log_message)

        # --- ✅ NEW: Automatic Retry Logic ---
        response = None
        for attempt in range(len(GEMINI_API_KEYS)):
            try:
                async with message.channel.typing():
                    response = await model.generate_content_async(prompt_parts)
                break # Success
            except api_core_exceptions.ResourceExhausted as e:
                print(f"Quota exceeded on key {current_api_key_index}. Rotating key. Attempt {attempt + 1}/{len(GEMINI_API_KEYS)}")
                rotate_api_key()
                if attempt == len(GEMINI_API_KEYS) - 1:
                    await message.reply(f"All API keys have reached their quota. Please try again later.\n`{e}`")
                    return
            except (generation_types.BlockedPromptException, api_core_exceptions.InvalidArgument) as e:
                print(f"Prompt-related API Error: {e}")
                await message.reply(f"There was an issue with the prompt (it might be too long, empty, or blocked).\n`{e}`")
                return
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                await message.reply(f"An unexpected error occurred: `{e}`")
                return

        if response and response.parts:
            bot_response_text = response.text
            await message.reply(bot_response_text)
            log_to_chat_history(channel_id, [log_message], bot_response_text, message.id)
            summary = f"User Prompt: '{message.content}'\nBot Response: '{bot_response_text}'"
            await update_notes_with_gemini(user_id, summary)
        elif not response:
            print("All API keys failed.")
        else:
            await message.reply("My response was blocked or empty. This might be due to the prompt or safety filters.")


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
            
    await interaction.followup.send(f"Initialization complete! Created or updated notes for {initialized_users} user(s).")

@bot.tree.command(name="condense_all_notes", description="[Owner Only] Condenses all user notes to reduce size.")
@commands.is_owner()
async def condense_all_notes(interaction: discord.Interaction):
    """
    Loads all user notes, asks the meta_persona to condense each one, and saves the results.
    """
    await interaction.response.defer(ephemeral=True)
    print("Starting manual note condensation for all users...")

    try:
        with open(USER_NOTES_FILE, 'r') as f:
            all_notes = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        await interaction.followup.send("Could not find or read the user_notes.json file.")
        return

    if not all_notes:
        await interaction.followup.send("There are no user notes to condense.")
        return

    meta_model = create_model(persona=META_PERSONA)
    condensed_count = 0
    error_count = 0

    for user_id, existing_notes in all_notes.items():
        print(f"Condensing notes for user {user_id}...")
        condensation_prompt = (
            f"The following notes for user <@{user_id}> may be too long. "
            "Your task is to aggressively summarize and condense them. Retain the most critical information about their personality, "
            "interests, and key facts, but significantly reduce the overall length. Rewrite the notes to be as efficient as possible.\n\n"
            f"**Existing Notes to Condense:**\n{existing_notes}"
        )

        try:
            response = await meta_model.generate_content_async(condensation_prompt)
            if response.text:
                new_notes = response.text
                all_notes[user_id] = new_notes
                print(f"Successfully condensed notes for user {user_id}.")
                condensed_count += 1
            else:
                print(f"Meta-model failed to condense notes for user {user_id}.")
                error_count += 1
        except Exception as e:
            print(f"An error occurred during note condensation for user {user_id}: {e}")
            error_count += 1
    
    with open(USER_NOTES_FILE, 'w') as f:
        json.dump(all_notes, f, indent=4)

    await interaction.followup.send(f"Condensation complete! Successfully processed notes for {condensed_count} user(s). Failed to process {error_count} user(s).")


bot.run(DISCORD_TOKEN)
