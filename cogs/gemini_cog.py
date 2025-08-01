import discord
from discord.ext import commands
import os
import google.generativeai as genai
from google.generativeai.types import generation_types
from google.generativeai.types import content_types
from google.generativeai import protos
from google.api_core import exceptions as api_core_exceptions
import io
import json
import re
from PIL import Image
from datetime import datetime, timezone

# --- NEW: Import the LastFm class from your new cog ---
from cogs.lastfm_cog import LastFm 

class GeminiCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config_files = bot.config_files
        
        # --- Load Configurations ---
        self.persona = self._load_text_file(self.config_files["persona"])
        self.meta_persona = self._load_text_file(self.config_files["meta_persona"])
        
        # --- CORRECTED PART ---
        # Call _load_keys once and correctly unpack all three expected values
        _, self.gemini_api_keys, self.lastfm_api_key = self._load_keys(self.config_files["keys"])
        # --- END OF CORRECTION ---

        self.allowed_channel_ids = self._load_allowed_channels(self.config_files["channels"])
        
        # --- Load Last.fm API and create an instance ---
        self.last_fm = LastFm(self.lastfm_api_key)

        # --- Define the Last.fm tool for Gemini ---
        self.lastfm_tool = content_types.Tool(
            function_declarations=[
                content_types.FunctionDeclaration(
                    name='get_top_artists',
                    description='Fetches the top 5 most played artists for a given Last.fm username.',
                    parameters=protos.Schema(
                        type=protos.Type.OBJECT,
                        properties={
                            'username': protos.Schema(type=protos.Type.STRING, description='The username on Last.fm')
                        },
                        required=['username']
                    )
                )
            ]
        )
        
        # --- State Management ---
        self.current_api_key_index = 0
        self.short_term_memory_turns = 800

        # --- Safety Settings ---
        self.safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }

        # --- Initialize Gemini Model ---
        self.model = self._create_model(persona=self.persona)
        print("GeminiCog loaded and initialized.")
        self.bot = bot
        self.config_files = bot.config_files
        
        # --- Load Configurations ---
        self.persona = self._load_text_file(self.config_files["persona"])
        self.meta_persona = self._load_text_file(self.config_files["meta_persona"])
        _, self.gemini_api_keys = self._load_keys(self.config_files["keys"])
        self.allowed_channel_ids = self._load_allowed_channels(self.config_files["channels"])
        
        # --- NEW: Load Last.fm API key and create an instance ---
        self.lastfm_api_key = self._load_keys(self.config_files["keys"])[2] # Assuming it's the 3rd key
        self.last_fm = LastFm(self.lastfm_api_key)

        # --- NEW: Define the Last.fm tool for Gemini ---
        self.lastfm_tool = content_types.Tool(
            function_declarations=[
                content_types.FunctionDeclaration(
                    name='get_top_artists',
                    description='Fetches the top 5 most played artists for a given Last.fm username.',
                    parameters=protos.Schema(
                        type=protos.Type.OBJECT,
                        properties={
                            'username': protos.Schema(type=protos.Type.STRING, description='The username on Last.fm')
                        },
                        required=['username']
                    )
                )
            ]
        )
        
        # --- State Management ---
        self.current_api_key_index = 0
        self.short_term_memory_turns = 800

        # --- Safety Settings ---
        self.safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }

        # --- Initialize Gemini Model ---
        self.model = self._create_model(persona=self.persona)
        print("GeminiCog loaded and initialized.")

    # --- Modified Configuration Loading to get the lastfm key ---
    def _load_keys(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                keys = json.load(f)
                return keys.get("discord_token"), keys.get("gemini_keys", []), keys.get("lastfm_api_key")
        except Exception as e:
            print(f"Error loading keys from {filepath}: {e}")
            return None, [], None

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
        self.model = self._create_model(persona=self.persona)
    
    # --- MODIFIED: The generate function now accepts and passes the tools ---
    async def _generate_with_full_rotation(self, prompt_parts, tools=None):
        num_keys = len(self.gemini_api_keys)
        if num_keys == 0:
            raise ValueError("Cannot generate content: No Gemini API keys found.")
        
        last_exception = None
        for attempt in range(num_keys):
            try:
                print(f"Attempting API call with key index {self.current_api_key_index} (Attempt {attempt + 1}/{num_keys})")
                # Pass the tools to the model
                response = await self.model.generate_content_async(prompt_parts, tools=tools)
                return response
            except api_core_exceptions.ResourceExhausted as e:
                print(f"Key index {self.current_api_key_index} is over quota.")
                last_exception = e
                self._rotate_api_key()
            except Exception as e:
                print(f"Encountered a non-retriable error: {e}")
                raise e
        
        print("All available API keys are exhausted.")
        if last_exception:
            raise last_exception
            
    # --- NEW: Function to handle tool calls from the model ---
    async def _handle_tool_call(self, response):
        # This function checks if the model wants to call a tool
        if not response.candidates or not response.candidates[0].content.parts:
            return None, None
            
        part = response.candidates[0].content.parts[0]
        if part.function_call:
            function_call = part.function_call
            if function_call.name == 'get_top_artists':
                # Extract arguments the model provided
                args = {key: value for key, value in function_call.args.items()}
                username = args.get("username")
                print(f"Gemini requested to call 'get_top_artists' for user: {username}")

                # Call the actual Python function
                tool_result = await self.last_fm.get_top_artists(username)

                # Package the result in the format the model expects
                return content_types.to_content(
                    content_types.Part(
                        function_response=content_types.FunctionResponse(
                            name='get_top_artists',
                            response={'result': tool_result}
                        )
                    )
                ), tool_result # Return both the model-formatted part and the raw result for logging

        return None, None # No function call was made

    # --- Persistence and Context Methods (no changes needed here) ---
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
        print(f"Updating notes for user {user_id}...")
        existing_notes = self._load_user_notes(user_id)
        meta_model = self._create_model(persona=self.meta_persona)
        
        update_prompt = (
            f"**Existing Notes on User <@{user_id}>:**\n{existing_notes}\n\n"
            f"**Recent Conversation Summary:**\n{conversation_summary}\n\n"
            "Please update the notes based on this new information. Keep it concise."
        )
        
        try:
            response = await meta_model.generate_content_async(update_prompt)
            if response and response.text:
                self._save_user_notes(user_id, response.text.strip())
                print(f"Successfully updated notes for user {user_id}.")
        except Exception as e:
            print(f"An error occurred during note update for user {user_id}: {e}")

    def _build_context_prompt(self, user_id, channel_id):
        user_notes = self._load_user_notes(user_id)
        
        try:
            with open(self.config_files["history"], 'r') as f:
                all_histories = json.load(f)
            
            channel_history = all_histories.get(str(channel_id), [])
            recent_history = channel_history[-(self.short_term_memory_turns * 2):]
            
            history_lines = [json.dumps(entry) for entry in recent_history if isinstance(entry, dict)]
            short_term_memory = "\n".join(history_lines)

        except (FileNotFoundError, json.JSONDecodeError):
            short_term_memory = "No chat history found."

        context_instructions = (
            "Instructions for reading conversation history:\n"
            "The following is a transcript of the recent conversation. Each message is a self-contained JSON object.\n"
            "You MUST parse these JSON objects to understand the conversation flow. The 'content' field contains the message text.\n"
            "Do not output your own responses in JSON format unless specifically asked to. Respond naturally based on the content.\n"
            "--- End of Instructions ---\n"
        )

        return [
            f"### My Long-Term Notes on <@{user_id}>:\n{user_notes}\n",
            context_instructions,
            f"### Recent Conversation (Short-Term Memory):\n{short_term_memory}\n"
        ]

    # --- Helper and Response Processing Methods ---
    async def _send_long_message(self, ctx, text, files=None):
        chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
        for i, chunk in enumerate(chunks):
            attached_files = files if i == 0 else None
            if isinstance(ctx, discord.Interaction):
                if i == 0:
                     await ctx.followup.send(content=chunk, files=attached_files)
                else:
                     await ctx.channel.send(content=chunk)
            else: # It's a message context
                if i == 0:
                    await ctx.reply(chunk, files=attached_files)
                else:
                    await ctx.channel.send(chunk)

    async def _process_and_send_response(self, ctx, response):
        if not response or not response.parts: return ""

        files_to_send = []
        image_parts = [p for p in response.parts if hasattr(p, 'inline_data') and p.inline_data.data]
        if image_parts:
            for i, part in enumerate(image_parts):
                files_to_send.append(discord.File(io.BytesIO(part.inline_data.data), filename=f"generated_image_{i+1}.png"))
        
        full_text = "".join([p.text for p in response.parts if p.text])
        
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

    # --- Main Command ---
    @commands.hybrid_command(name="gemini", description="Ask a question to the Gemini model, or continue the conversation.")
    async def gemini(self, ctx: commands.Context, *, prompt: str = None, attachment: discord.Attachment = None):
        await ctx.defer()
        
        user_id = ctx.author.id
        channel_id = ctx.channel.id
        
        final_prompt_parts = self._build_context_prompt(user_id, channel_id)
        user_content_for_log = prompt if prompt else ""
        
        current_input_dict = {
            "role": "user", "timestamp": ctx.message.created_at.isoformat(),
            "author_id": user_id, "author_name": ctx.author.display_name, "content": prompt or ""
        }
        
        if attachment:
            user_content_for_log += f" [Attached file: {attachment.filename}]"
            if "image" in attachment.content_type:
                try:
                    img = Image.open(io.BytesIO(await attachment.read()))
                    final_prompt_parts.append(img)
                    if not prompt: current_input_dict["content"] = "[User sent an image.]"
                except Exception as e:
                    await ctx.send(f"I had trouble reading that image file. Error: {e}")
                    return
            else:
                current_input_dict["content"] += f" (Attached file: '{attachment.filename}')"

        if not prompt and not attachment:
            current_input_dict["content"] = "[Continuation without text]"
            user_content_for_log = "[Continuation without text]"

        final_prompt_parts.append(json.dumps(current_input_dict))
        
        print(f"{ctx.author.display_name} ({user_id}) used /gemini in channel {channel_id}")

        try:
            # --- MODIFIED: Tool calling workflow ---
            # 1. First call to the model with the tool definition
            response = await self._generate_with_full_rotation(final_prompt_parts, tools=[self.lastfm_tool])
            
            # 2. Check if the model decided to call the function
            tool_part, raw_tool_result = await self._handle_tool_call(response)

            # 3. If a tool was called, send the result back to the model
            if tool_part:
                print("Tool was called. Sending result back to Gemini for a natural response.")
                # Add the tool's result to the conversation history for context
                final_prompt_parts.append(response.candidates[0].content) # Model's function call request
                final_prompt_parts.append(tool_part) # Your function's result
                
                # Call the model again to get a conversational summary of the tool's output
                response = await self._generate_with_full_rotation(final_prompt_parts)

            # --- End of modification ---

            if not response or not response.parts:
                await ctx.send("My response was blocked or empty. This might be due to safety filters.")
                return

            final_bot_text = await self._process_and_send_response(ctx, response)
            self._log_to_chat_history(channel_id, ctx.author, ctx.message.created_at, user_content_for_log.strip(), final_bot_text)
            
            # Add tool result to the summary if it exists
            summary_note = f"Tool Result: {raw_tool_result}" if raw_tool_result else ""
            summary = f"User Prompt: '{user_content_for_log.strip()}'\nBot Response: '{final_bot_text}'\n{summary_note}"
            await self._update_notes_with_gemini(user_id, summary)
                
        except Exception as e:
            await ctx.send(f"An unexpected error occurred: `{e}`")

    # --- Event Listener ---
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.guild and message.channel.id not in self.allowed_channel_ids or message.author == self.bot.user:
            return

        mention_triggers = ['gemini', 'gem', str(self.bot.user.id)]
        natural_triggers = ["how do", "what is", "can someone", "i wonder", "is it possible"]
        content_lower = message.content.lower()

        is_dm = isinstance(message.channel, discord.DMChannel)
        is_reply_to_bot = message.reference and message.reference.resolved and message.reference.resolved.author == self.bot.user
        contains_mention = any(trigger in content_lower for trigger in mention_triggers)
        contains_natural = any(content_lower.startswith(phrase) for phrase in natural_triggers)

        if not (is_dm or is_reply_to_bot or contains_mention or contains_natural):
            return

        async with message.channel.typing():
            user_id = message.author.id
            channel_id = message.channel.id
            
            prompt_parts = self._build_context_prompt(user_id, channel_id)
            
            current_input_dict = {
                "role": "user", "timestamp": message.created_at.isoformat(),
                "author_id": user_id, "author_name": message.author.display_name, "content": message.content
            }
            
            for attachment in message.attachments:
                if "image" in attachment.content_type:
                    try:
                        prompt_parts.append(Image.open(io.BytesIO(await attachment.read())))
                    except Exception as e:
                        print(f"Could not process image in on_message: {e}")
                else:
                    current_input_dict["content"] += f" (Attached file: '{attachment.filename}')"

            prompt_parts.append(json.dumps(current_input_dict))

            print(f"{message.author.display_name} ({user_id}) triggered bot in channel {channel_id}")

            try:
                # --- MODIFIED: Tool calling workflow for on_message ---
                response = await self._generate_with_full_rotation(prompt_parts, tools=[self.lastfm_tool])
                tool_part, raw_tool_result = await self._handle_tool_call(response)

                if tool_part:
                    print("Tool was called from on_message. Sending result back to Gemini.")
                    prompt_parts.append(response.candidates[0].content)
                    prompt_parts.append(tool_part)
                    response = await self._generate_with_full_rotation(prompt_parts)
                # --- End of modification ---

                if not response or not response.parts:
                    await message.reply("My response was blocked or empty.")
                    return
                
                final_bot_text = await self._process_and_send_response(message, response)
                self._log_to_chat_history(channel_id, message.author, message.created_at, message.content, final_bot_text)
                
                summary_note = f"Tool Result: {raw_tool_result}" if raw_tool_result else ""
                summary = f"User Prompt: '{message.content}'\nBot Response: '{final_bot_text}'\n{summary_note}"
                await self._update_notes_with_gemini(user_id, summary)

            except Exception as e:
                await message.reply(f"An unexpected error occurred: `{e}`")

async def setup(bot: commands.Bot):
    await bot.add_cog(GeminiCog(bot))