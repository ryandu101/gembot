import discord
from discord.ext import commands
import os
import google.generativeai as genai
from google.generativeai.types import generation_types
from google.api_core import exceptions as api_core_exceptions
import io
import json
import re
from PIL import Image
from datetime import datetime, timezone

class GeminiCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config_files = bot.config_files
        
        # --- Load Configurations ---
        self.persona = self._load_text_file(self.config_files["persona"])
        self.meta_persona = self._load_text_file(self.config_files["meta_persona"])
        _, self.gemini_api_keys = self._load_keys(self.config_files["keys"])
        self.allowed_channel_ids = self._load_allowed_channels(self.config_files["channels"])

        # --- State Management ---
        self.current_api_key_index = 0
        self.short_term_memory_turns = 900

        # --- Safety Settings ---
        self.safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }

        # --- Initialize Gemini Model ---
        # The model is created on-demand now to switch personas
        print("GeminiCog loaded and initialized.")

    # --- Configuration Loading Methods ---
    def _load_keys(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                keys = json.load(f)
                return keys.get("discord_token"), keys.get("gemini_keys", [])
        except Exception as e:
            print(f"Error loading keys from {filepath}: {e}")
            return None, []

    def _load_text_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def _load_allowed_channels(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("allowed_channel_ids", [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    # --- Gemini API Methods ---
    def _create_model(self, persona):
        if not self.gemini_api_keys:
            print("Error: No Gemini API keys found.")
            return None
        # Ensure we don't go out of bounds
        self.current_api_key_index %= len(self.gemini_api_keys)
        api_key = self.gemini_api_keys[self.current_api_key_index]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            'gemini-2.5-pro',
            system_instruction=persona,
            safety_settings=self.safety_settings
        )

    def _rotate_api_key(self):
        self.current_api_key_index = (self.current_api_key_index + 1) % len(self.gemini_api_keys)
        print(f"Switching to API Key index {self.current_api_key_index}")
        # The model will be re-created on the next call to _create_model

    async def _generate_with_full_rotation(self, prompt_parts, persona):
        num_keys = len(self.gemini_api_keys)
        if num_keys == 0:
            raise ValueError("Cannot generate content: No Gemini API keys found.")
        
        last_exception = None
        for attempt in range(num_keys):
            try:
                # Create the model here to ensure the correct persona and key are used
                model = self._create_model(persona=persona)
                if model is None: raise ValueError("Model creation failed.")

                print(f"Attempting API call with key index {self.current_api_key_index} (Attempt {attempt + 1}/{num_keys})")
                response = await model.generate_content_async(prompt_parts)
                return response
            except api_core_exceptions.ResourceExhausted as e:
                print(f"Key index {self.current_api_key_index} is over quota.")
                last_exception = e
                self._rotate_api_key() # Rotate for the next attempt
            except Exception as e:
                print(f"Encountered a non-retriable error: {e}")
                raise e # Re-raise other errors immediately
        
        print("All available API keys are exhausted.")
        if last_exception:
            raise last_exception

    # --- Persistence and Context Methods ---
    def _load_user_notes(self, user_id):
        try:
            with open(self.config_files["notes"], 'r') as f:
                all_notes = json.load(f)
            return all_notes.get(str(user_id), "No notes on this user yet.")
        except (FileNotFoundError, json.JSONDecodeError):
            return "No notes on this user yet."

    def _save_user_notes(self, user_id, notes):
        try:
            with open(self.config_files["notes"], 'r') as f:
                all_notes = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_notes = {}
        all_notes[str(user_id)] = notes
        with open(self.config_files["notes"], 'w') as f:
            json.dump(all_notes, f, indent=4)

    def _log_to_chat_history(self, channel_id, user_author, user_timestamp, user_content, bot_response_text):
        # This function should only be called for allowed channels
        try:
            with open(self.config_files["history"], 'r') as f:
                all_histories = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_histories = {}

        channel_id_str = str(channel_id)
        if channel_id_str not in all_histories:
            all_histories[channel_id_str] = []

        user_entry = {
            "role": "user", "author_id": user_author.id, "author_name": user_author.display_name,
            "timestamp": user_timestamp.isoformat(), "content": user_content
        }
        all_histories[channel_id_str].append(user_entry)
        
        bot_entry = {
            "role": "model", "author_id": self.bot.user.id, "author_name": self.bot.user.display_name,
            "timestamp": datetime.now(timezone.utc).isoformat(), "content": bot_response_text
        }
        all_histories[channel_id_str].append(bot_entry)

        with open(self.config_files["history"], 'w') as f:
            json.dump(all_histories, f, indent=4)
    
    async def _update_notes_with_gemini(self, user_id, conversation_summary):
        # This function should only be called for allowed channels
        print(f"Updating notes for user {user_id}...")
        existing_notes = self._load_user_notes(user_id)
        
        update_prompt = (
            f"**Existing Notes on User <@{user_id}>:**\n{existing_notes}\n\n"
            f"**Recent Conversation Summary:**\n{conversation_summary}\n\n"
            "Please update the notes based on this new information. Keep it concise."
        )
        
        try:
            # Use the meta_persona for this specific task
            response = await self._generate_with_full_rotation([update_prompt], persona=self.meta_persona)
            if response and response.text:
                self._save_user_notes(user_id, response.text.strip())
                print(f"Successfully updated notes for user {user_id}.")
        except Exception as e:
            print(f"An error occurred during note update for user {user_id}: {e}")

    def _build_context_prompt(self, user_id, channel_id):
        """Builds a secure and structured prompt context to prevent injection."""
        user_notes = self._load_user_notes(user_id)
        
        # Define clear, non-natural separators for the history block
        history_header = "--- BEGIN CONVERSATION HISTORY ---"
        history_footer = "--- END CONVERSATION HISTORY ---"
        
        try:
            with open(self.config_files["history"], 'r', encoding='utf-8') as f:
                all_histories = json.load(f)
            
            channel_history = all_histories.get(str(channel_id), [])
            recent_history = channel_history[-(self.short_term_memory_turns * 2):]
            
            history_lines = []
            for entry in recent_history:
                # Use a simple "Author: Message" format for clarity
                if isinstance(entry, dict) and 'role' in entry and 'content' in entry:
                    author_name = entry.get('author_name', 'Unknown')
                    content = entry.get('content', '')
                    history_lines.append(f"{author_name}: {content}")
            
            short_term_memory = "\n".join(history_lines)

        except (FileNotFoundError, json.JSONDecodeError):
            short_term_memory = "No chat history found for this channel."

        # This new structure clearly separates each piece of context for the model.
        # The most important part is the "CURRENT USER PROMPT" block, which isolates the user's new message.
        context_string = (
            f"### My Long-Term Notes on <@{user_id}>:\n{user_notes}\n\n"
            f"{history_header}\n"
            f"{short_term_memory}\n"
            f"{history_footer}\n\n"
            "--- BEGIN CURRENT USER PROMPT ---" # This wrapper is key to the security
        )
        
        # The base context is returned. The calling function will add the actual user message.
        return [context_string]

    # --- Helper and Response Processing Methods ---
    async def _send_long_message(self, ctx, text, files=None):
        chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
        for i, chunk in enumerate(chunks):
            attached_files = files if i == 0 else None
            # Use followup.send for interactions, as defer() was used
            if isinstance(ctx, discord.Interaction):
                if i == 0:
                     await ctx.followup.send(content=chunk, files=attached_files)
                else:
                     await ctx.channel.send(content=chunk) # Subsequent messages can't be followups
            else: # It's a regular message context
                if i == 0:
                    await ctx.reply(chunk, files=attached_files)
                else:
                    await ctx.channel.send(chunk)

    async def _process_and_send_response(self, ctx, response):
        if not response or not response.parts: return ""

        files_to_send = []
        # Check for image data in response parts
        image_parts = [p for p in response.parts if hasattr(p, 'inline_data') and p.inline_data.data]
        if image_parts:
            for i, part in enumerate(image_parts):
                files_to_send.append(discord.File(io.BytesIO(part.inline_data.data), filename=f"generated_image_{i+1}.png"))
        
        full_text = "".join([p.text for p in response.parts if p.text])
        
        # If sending an image, clean up the text. Otherwise, check for code blocks.
        if files_to_send:
            bot_response_text = re.sub(r'\[Image of[^\]]+\]', '', full_text).strip() or "Here is the image you requested!"
        else:
            code_block_match = re.search(r"```(?:\w+)?\n([\s\S]+?)\n```", full_text)
            if code_block_match:
                code_content = code_block_match.group(1)
                lang_match = re.search(r"```(\w+)", full_text)
                extension = lang_match.group(1) if lang_match else "txt"
                files_to_send.append(discord.File(io.BytesIO(code_content.encode('utf-8')), filename=f"code.{extension}"))
                bot_response_text = "I've generated the code you asked for and attached it as a file."
            else:
                bot_response_text = full_text

        await self._send_long_message(ctx, bot_response_text, files=files_to_send if files_to_send else None)
        return bot_response_text

    # --- REBUILT /gemini COMMAND ---
    @commands.command(name="gemini", description="Ask a question to the Gemini model, or continue the conversation.")
    async def gemini(self, ctx: commands.Context, *, prompt: str = None, attachment: discord.Attachment = None):
        # Always defer for slash commands; it's good practice.
        await ctx.defer()
        
        user = ctx.author
        channel = ctx.channel
        
        # --- Context-Aware Logic ---
        # A DM channel won't have a guild attribute.
        is_dm = ctx.guild is None
        # Check if the channel is whitelisted. DMs are not in the whitelist.
        is_allowed_channel = not is_dm and channel.id in self.allowed_channel_ids

        final_prompt_parts = []
        
        # 1. Build context ONLY if in an allowed channel
        if is_allowed_channel:
            print(f"Context-aware request in allowed channel: {channel.id}")
            final_prompt_parts.extend(self._build_context_prompt(user.id, channel.id))
        else:
            print(f"Pure prompt request from user {user.id} (DM: {is_dm}, Channel: {channel.id if not is_dm else 'N/A'})")
            # For pure prompts, we don't add any history or notes.

        # 2. Add the user's current message/input
        user_content_for_log = prompt if prompt else ""
        current_input_dict = {
            "role": "user", "timestamp": ctx.message.created_at.isoformat(),
            "author_id": user.id, "author_name": user.display_name, "content": prompt or ""
        }
        
        # 3. Handle attachments
        if attachment:
            user_content_for_log += f" [Attached file: {attachment.filename}]"
            # We only process images for now
            if "image" in attachment.content_type:
                try:
                    img_bytes = await attachment.read()
                    img = Image.open(io.BytesIO(img_bytes))
                    final_prompt_parts.append(img)
                    if not prompt: current_input_dict["content"] = "[User sent an image.]"
                except Exception as e:
                    await ctx.followup.send(f"I had trouble reading that image file. Error: {e}")
                    return
            else:
                current_input_dict["content"] += f" (Attached file: '{attachment.filename}' that cannot be viewed)"

        if not prompt and not attachment:
            current_input_dict["content"] = "[Continuation without text]"
            user_content_for_log = "[Continuation without text]"

        # Add the final user input as a JSON string for clarity in the prompt
        final_prompt_parts.append(json.dumps(current_input_dict))
        
        print(f"{user.display_name} ({user.id}) used /gemini in channel {channel.id}")

        try:
            # 4. Generate the response
            response = await self._generate_with_full_rotation(final_prompt_parts, persona=self.persona)
            if not response or not response.parts:
                await ctx.followup.send("My response was blocked or empty. This might be due to safety filters.")
                return

            # 5. Process and send the response message
            final_bot_text = await self._process_and_send_response(ctx, response)
            
            # 6. Log and update notes ONLY if in an allowed channel
            if is_allowed_channel:
                self._log_to_chat_history(channel.id, user, ctx.message.created_at, user_content_for_log.strip(), final_bot_text)
                summary = f"User Prompt: '{user_content_for_log.strip()}'\nBot Response: '{final_bot_text}'"
                await self._update_notes_with_gemini(user.id, summary)
                
        except Exception as e:
            await ctx.followup.send(f"An unexpected error occurred: `{e}`")

    # --- Event Listener (on_message) ---
    # The on_message listener is inherently context-aware because it only
    # triggers in allowed channels, so it doesn't need changes.
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # Standard checks: ignore bots, ensure channel is allowed
        if message.author.bot or (message.guild and message.channel.id not in self.allowed_channel_ids):
            return

        # --- CORRECTED TRIGGER LOGIC ---
        # Define the keywords and natural phrases that should trigger the bot
        mention_triggers = ['1392960230228492508', 'gem']
        content_lower = message.content.lower()

        is_dm = isinstance(message.channel, discord.DMChannel)
        is_reply_to_bot = message.reference and message.reference.resolved and message.reference.resolved.author == self.bot.user
        
        # Check for direct @mention OR if 'gem'/'gemini' are in the message
        contains_mention = self.bot.user.mentioned_in(message) or any(word in content_lower for word in mention_triggers)

        # Only trigger if one of the conditions is met
        if not (is_dm or is_reply_to_bot or contains_mention):
            return
        # --- END OF CORRECTION ---

        async with message.channel.typing():
            user_id = message.author.id
            channel_id = message.channel.id
            
            # 1. Build the secure base context
            prompt_parts = self._build_context_prompt(user_id, channel_id)
            
            # 2. Safely prepare the user's content and handle any attachments
            user_content = message.content
            for attachment in message.attachments:
                if "image" in attachment.content_type:
                    try:
                        prompt_parts.append(Image.open(io.BytesIO(await attachment.read())))
                    except Exception as e:
                        print(f"Could not process image in on_message: {e}")
                else:
                    user_content += f" (Note: User also attached a non-image file named '{attachment.filename}')"

            # 3. Add the current user's message, safely wrapped
            prompt_parts.append(f"\n{message.author.display_name} (<@{user_id}>): {user_content}\n--- END CURRENT USER PROMPT ---")

            print(f"Secure prompt initiated by {message.author.display_name} ({user_id}) in channel {channel_id}")

            try:
                response = await self._generate_with_full_rotation(prompt_parts, persona=self.persona)
                if not response or not response.parts:
                    await message.reply("My response was blocked or empty. This might be due to safety filters.")
                    return
                
                final_bot_text = await self._process_and_send_response(message, response)
                
                # Log history and update notes
                self._log_to_chat_history(channel_id, message.author, message.created_at, message.content, final_bot_text)
                summary = f"User Prompt: '{message.content}'\nBot Response: '{final_bot_text}'"
                await self._update_notes_with_gemini(user_id, summary)

            except Exception as e:
                await message.reply(f"An unexpected error occurred: `{e}`")

async def setup(bot: commands.Bot):
    await bot.add_cog(GeminiCog(bot))