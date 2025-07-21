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
# Add channel whitelist file
CHANNEL_IDS_FILE = "channel_ids.json"
DEV_GUILD_ID = 930444180520570930 # Replace with your actual Server ID
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
        # We can use a simpler retry for the background task to avoid complexity
        response = await model.generate_content_async(update_prompt)
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
    # Re-create the model with the new key
    model = create_model()

# --- Unified Context Building Function ---
def build_context_prompt(user_id, channel_id):
    """Builds the prompt context using long-term notes and short-term history."""
    user_notes = load_user_notes(user_id)
    
    try:
        with open(CHAT_HISTORY_FILE, 'r') as f:
            all_histories = json.load(f)
        
        channel_history = all_histories.get(str(channel_id), [])
        # Get the last N turns (1 turn = 1 user + 1 model)
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

# --- Helper function to send long messages ---
async def send_long_message(context, text):
    """Splits a long message into chunks of 2000 characters and sends them."""
    if len(text) <= 2000:
        # Check if context is an Interaction or a Message
        if isinstance(context, discord.Interaction):
            await context.followup.send(content=text)
        else:
            await context.reply(text)
        return

    # Split the text into chunks
    chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
    for i, chunk in enumerate(chunks):
        if i == 0:
            if isinstance(context, discord.Interaction):
                await context.followup.send(content=chunk)
            else:
                await context.reply(chunk)
        else:
            # Send subsequent chunks to the channel
            await context.channel.send(content=chunk)

@bot.event
async def on_ready():
    try:
        if DEV_GUILD_ID:
            guild = discord.Object(id=DEV_GUILD_ID)
            bot.tree.copy_global_to(guild=guild)
            await bot.tree.sync(guild=guild)
            print(f"Synced commands to development guild: {DEV_GUILD_ID}")
        else:
            # Sync commands globally if no dev guild is specified
            await bot.tree.sync()
            print("Synced commands globally.")
    except Exception as e:
        print(e)
    print(f'Logged in as {bot.user}')
    print(f'Using API Key index {current_api_key_index}')
    print('------')

# --- START MODIFICATION ---
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
            # Success! Return the response immediately.
            return response
        except api_core_exceptions.ResourceExhausted as e:
            print(f"Key index {current_api_key_index} is over quota.")
            last_exception = e
            # Rotate to the next key for the next attempt in the loop.
            rotate_api_key()
        except Exception as e:
            # For any other kind of error (e.g., BlockedPromptException), fail immediately
            # as trying another key won't help.
            print(f"Encountered a non-retriable error: {e}")
            raise e
            
    # If the loop completes, it means all keys failed. Raise the last captured exception.
    print("All available API keys are exhausted.")
    if last_exception:
        raise last_exception
# --- END MODIFICATION ---

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
                     # Add a default prompt if only an image is sent
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

    try:
        # --- START MODIFICATION ---
        response = await generate_with_full_rotation(final_prompt_parts)
        # --- END MODIFICATION ---
        
        if not response.parts:
            await interaction.followup.send(content="My response was blocked or empty. This might be due to the prompt or safety filters.")
        else:
            bot_response_text = response.text
            # --- START MODIFICATION ---
            if prompt:
                response_with_prompt = f"> {prompt}\n\n{bot_response_text}"
                await send_long_message(interaction, response_with_prompt)
            else:
                await send_long_message(interaction, bot_response_text)
            # --- END MODIFICATION ---
            log_to_chat_history(channel_id, [log_message], bot_response_text)
            
            # Create summary for note update
            user_prompt_text = prompt or "Sent an image"
            summary = f"User Prompt: '{user_prompt_text}'\nBot Response: '{bot_response_text}'"
            await update_notes_with_gemini(user_id, summary)
            
    # --- START MODIFICATION ---
    except api_core_exceptions.ResourceExhausted as e:
        # This now only triggers after all keys have been tried and failed.
        print(f"Final error after cycling through all keys: {e}.")
        await interaction.followup.send(content=f"I'm sorry, but all of my available API keys are currently over quota. Please try again in a few minutes.\n`{e}`")
    # --- END MODIFICATION ---
    except (generation_types.BlockedPromptException, api_core_exceptions.InvalidArgument) as e:
        print(f"Prompt-related API Error: {e}")
        await interaction.followup.send(content=f"There was an issue with the prompt (it might be too long, empty, or blocked).\n`{e}`")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        await interaction.followup.send(content=f"An unexpected error occurred: `{e}`")


@bot.event
async def on_message(message):
    # Ignore messages in channels not on the whitelist
    if message.guild and message.channel.id not in ALLOWED_CHANNEL_IDS:
        return

    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Define triggers for the bot to respond
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
    # Check if the message starts with a natural trigger phrase
    contains_natural_trigger = any(content_lower.startswith(phrase) for phrase in natural_trigger_phrases)

    # Respond if it's a DM, a reply to the bot, or contains a trigger word
    if is_dm or is_reply_to_bot or contains_mention or contains_natural_trigger:
        user_id = message.author.id
        channel_id = message.channel.id
        
        prompt_parts = build_context_prompt(user_id, channel_id)
        
        log_message = f"<@{user_id}> said: '{message.content}'"
        
        # Handle replies
        if message.reference and isinstance(message.reference.resolved, discord.Message):
            original_message = message.reference.resolved
            # Add context from the replied-to message
            prompt_parts.append(f"\nContext: In reply to a message from <@{original_message.author.id}> that said: '{original_message.content}'")
            # Process attachments in the replied-to message
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
            # Handle regular messages
            prompt_parts.append(f"\n<@{user_id}> said: {message.content}")

        # Process attachments in the new message
        for attachment in message.attachments:
            if "image" in attachment.content_type:
                try:
                    image_bytes = await attachment.read()
                    prompt_parts.append(Image.open(io.BytesIO(image_bytes)))
                    log_message += " (with an image attachment)"
                except Exception as e:
                    print(f"Could not process new image: {e}")
            else:
                # Inform the model about non-viewable attachments
                prompt_parts.append(f"\n(User also attached a file named '{attachment.filename}' that cannot be viewed.)")
        
        # Don't respond to empty messages (e.g., only an attachment that failed to load)
        if not message.content.strip() and not any(isinstance(p, Image.Image) for p in prompt_parts):
             print("Ignoring message with no text or valid images.")
             return

        print(log_message)

        try:
            # Show a typing indicator while generating the response
            async with message.channel.typing():
                # --- START MODIFICATION ---
                response = await generate_with_full_rotation(prompt_parts)
                # --- END MODIFICATION ---

                if not response.parts:
                    await message.reply("My response was blocked or empty. This might be due to the prompt or safety filters.")
                else:
                    bot_response_text = response.text
                    await send_long_message(message, bot_response_text)
                    # Log conversation for context and note-taking
                    log_to_chat_history(channel_id, [log_message], bot_response_text, message.id)
                    summary = f"User Prompt: '{message.content}'\nBot Response: '{bot_response_text}'"
                    await update_notes_with_gemini(user_id, summary)

        # --- START MODIFICATION ---
        except api_core_exceptions.ResourceExhausted as e:
            # This now only triggers after all keys have been tried and failed.
            print(f"Final error after cycling through all keys: {e}.")
            await message.reply(f"I'm sorry, but all of my available API keys are currently over quota. Please try again in a few minutes.\n`{e}`")
        # --- END MODIFICATION ---
        except (generation_types.BlockedPromptException, api_core_exceptions.InvalidArgument) as e:
            print(f"Prompt-related API Error: {e}")
            await message.reply(f"There was an issue with the prompt (it might be too long, empty, or blocked).\n`{e}`")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            await message.reply(f"An unexpected error occurred: `{e}`")

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
    # Regex to find user IDs in logged messages
    user_id_pattern = re.compile(r"<@(\d+)>")

    for channel_id, messages in old_history.items():
        for message in messages:
            if message['role'] == 'user':
                full_text = " ".join(message['parts'])
                # Find the user ID in the logged message
                match = user_id_pattern.search(full_text)
                if match:
                    user_id = match.group(1)
                    # Initialize a list for the user if not already present
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
                # Overwrite the old notes with the new condensed version
                all_notes[user_id] = new_notes
                print(f"Successfully condensed notes for user {user_id}.")
                condensed_count += 1
            else:
                print(f"Meta-model failed to condense notes for user {user_id}.")
                error_count += 1
        except Exception as e:
            print(f"An error occurred during note condensation for user {user_id}: {e}")
            error_count += 1
    
    # Save the updated notes back to the file
    with open(USER_NOTES_FILE, 'w') as f:
        json.dump(all_notes, f, indent=4)

    await interaction.followup.send(f"Condensation complete! Successfully processed notes for {condensed_count} user(s). Failed to process {error_count} user(s).")

@bot.tree.command(name="createnotes", description="[Owner Only] Creates/rewrites notes for a user from this channel's history.")
@commands.is_owner()
async def createnotes(interaction: discord.Interaction, user: discord.User):
    await interaction.response.defer(ephemeral=True)
    target_user_id = user.id
    print(f"Starting note creation for user {target_user_id} from channel {interaction.channel.id}...")

    user_messages = []
    # Limit to the last 5000 messages to avoid extreme processing times
    async for message in interaction.channel.history(limit=5000):
        if message.author.id == target_user_id:
            user_messages.append(message.content)

    if not user_messages:
        await interaction.followup.send(f"No recent messages found for user <@{target_user_id}> in this channel.")
        return

    # Reverse the list to get chronological order
    user_messages.reverse()
    full_conversation = "\n".join(user_messages)

    meta_model = create_model(persona=META_PERSONA)
    
    prompt = (
        f"Please analyze the following conversation history for user <@{target_user_id}> and create a concise set of initial notes about their personality, "
        "interests, and key topics they discuss. Focus on creating a summary that would be useful for future interactions.\n\n"
        f"**Conversation History:**\n{full_conversation}"
    )

    try:
        response = await meta_model.generate_content_async(prompt)
        if response.text:
            save_user_notes(target_user_id, response.text)
            print(f"Successfully created notes for user {target_user_id}.")
            await interaction.followup.send(f"Successfully created/rewrote notes for <@{target_user_id}> based on their messages in this channel.")
        else:
            print(f"Failed to generate notes for user {target_user_id} (empty response).")
            await interaction.followup.send(f"Failed to generate notes for <@{target_user_id}>.")
    except Exception as e:
        print(f"An error occurred while generating notes for user {target_user_id}: {e}")
        await interaction.followup.send(f"An error occurred while generating notes for <@{target_user_id}>. Check the console.")


bot.run(DISCORD_TOKEN)