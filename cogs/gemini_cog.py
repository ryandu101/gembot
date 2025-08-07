import discord
from discord.ext import commands
import os
import google.generativeai as genai
from google.generativeai.types import generation_types, Tool
from google.api_core import exceptions as api_core_exceptions
import io
import json
import re
from PIL import Image
from datetime import datetime, timezone
from tools.google_search import google_search_impl, GoogleSearchError

class GeminiCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config_files = bot.config_files
        
        # --- Load Configurations ---
        self.persona = self._load_text_file(self.config_files["persona"])
        self.meta_persona = self._load_text_file(self.config_files["meta_persona"])
        (
            _,
            self.gemini_api_keys,
            self.google_cse_id,
            self.google_cse_api_key,
        ) = self._load_keys(self.config_files["keys"])
        self.allowed_channel_ids = self._load_allowed_channels(self.config_files["channels"])

        # --- State Management ---
        self.current_api_key_index = 0
        self.short_term_memory_turns = 2000

        # --- Safety Settings ---
        self.safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }

        # --- Initialize Gemini Model ---
        self.google_search_tool = {
            "function_declarations": [
                {
                    "name": "google_search_impl",
                    "description": "Performs a Google search and returns a list of results. Use this for recent information or when you need to look something up.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "query": {"type": "STRING"},
                            "num_results": {"type": "INTEGER"},
                        },
                        "required": ["query"],
                    },
                }
            ]
        }
        print("GeminiCog loaded and initialized.")

    # --- Configuration Loading Methods ---
    def _load_keys(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                keys = json.load(f)
                return (
                    keys.get("discord_token"),
                    keys.get("gemini_keys", []),
                    keys.get("google_cse_id"),
                    keys.get("google_cse_api_key"),
                )
        except Exception as e:
            print(f"Error loading keys from {filepath}: {e}")
            return None, [], None, None

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

        # Only provide the tool if the necessary keys are available
        tools = [self.google_search_tool] if (self.google_cse_id and self.google_cse_api_key) else None

        return genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=persona,
            safety_settings=self.safety_settings,
            tools=tools
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
            model = self._create_model(persona=persona)
            if model is None:
                raise ValueError("Model creation failed.")

            print(f"Attempting API call with key index {self.current_api_key_index} (Attempt {attempt + 1}/{num_keys})")

            try:
                # Start the generation loop
                for i in range(5): # Limit to 5 tool calls to prevent infinite loops
                    response = await model.generate_content_async(prompt_parts)

                    # Check for tool calls
                    if not response.candidates or not response.candidates[0].content.parts:
                        return response # No content, return as is

                    part = response.candidates[0].content.parts[0]
                    if not hasattr(part, 'function_call'):
                        return response # It's a final answer

                    # --- Handle Tool Call ---
                    function_call = part.function_call
                    tool_name = function_call.name

                    print(f"Model requested to use tool: {tool_name}")

                    if tool_name == "google_search_impl":
                        try:
                            # Extract args and call the tool function
                            args = {key: value for key, value in function_call.args.items()}
                            print(f"Tool args: {args}")

                            # Call the actual search function with required auth
                            search_results = await google_search_impl(
                                api_key=self.google_cse_api_key,
                                cse_id=self.google_cse_id,
                                query=args.get("query", ""),
                                num_results=args.get("num_results", 5)
                            )

                            # Append the results back to the prompt for the next turn
                            prompt_parts.append({"parts": [
                                {"tool_response": {"name": "google_search_impl", "response": search_results}}]})

                        except GoogleSearchError as se:
                            print(f"Google Search Error: {se}")
                            # Inform the model the tool call failed
                            prompt_parts.append({"parts": [
                                {"tool_response": {"name": "google_search_impl", "response": {"error": str(se)}}}]})
                        except Exception as e:
                            print(f"An unexpected error occurred during tool call: {e}")
                            # Pass a generic error back to the model
                            prompt_parts.append({"parts": [
                                {"tool_response": {"name": "google_search_impl", "response": {"error": f"Tool execution failed: {e}"}}}]})
                    else:
                        print(f"Warning: Model called unknown tool '{tool_name}'")
                        # Inform the model the tool is not available
                        prompt_parts.append({"parts": [
                            {"tool_response": {"name": tool_name, "response": {"error": "Tool not found."}}}]})

                # If the loop finishes, it means we hit the tool call limit
                raise Exception("Exceeded maximum tool call limit.")

            except api_core_exceptions.ResourceExhausted as e:
                print(f"Key index {self.current_api_key_index} is over quota.")
                last_exception = e
                self._rotate_api_key()  # Rotate for the next attempt and continue the outer loop
            except Exception as e:
                print(f"Encountered a non-retriable error: {e}")
                raise e  # Re-raise other critical errors immediately

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
        
        # Create a more focused summary for note updating
        # Extract just the key points from the conversation
        summary_lines = conversation_summary.split('\n')
        user_prompt_line = next((line for line in summary_lines if line.startswith("User Prompt:")), "")
        bot_response_line = next((line for line in summary_lines if line.startswith("Bot Response:")), "")
        
        # Create a more concise summary
        concise_summary = f"{user_prompt_line}\n{bot_response_line}".strip()
        
        update_prompt = (
            f"**Existing Notes on User <@{user_id}>:**\n{existing_notes}\n\n"
            f"**Recent Conversation Summary:**\n{concise_summary}\n\n"
            "Please update the notes based on this new information. "
            "Follow your persona guidelines strictly: keep notes under 2000 characters, "
            "summarize rather than quote, focus on key facts and personality traits, "
            "and maintain the structured format with markdown headings."
        )
        
        try:
            # Use the meta_persona for this specific task
            response = await self._generate_with_full_rotation([update_prompt], persona=self.meta_persona)
            if response and response.text:
                new_notes = response.text.strip()
                # Ensure notes don't exceed 2000 characters
                if len(new_notes) > 2000:
                    new_notes = new_notes[:1997] + "..."
                self._save_user_notes(user_id, new_notes)
                print(f"Successfully updated notes for user {user_id}.")
        except Exception as e:
            print(f"An error occurred during note update for user {user_id}: {e}")

    def _build_context_prompt(self, user_id, channel_id):
        """Builds a secure and structured prompt context to prevent injection."""
        user_notes = self._load_user_notes(user_id)
        
        history_header = "--- BEGIN CONVERSATION HISTORY ---"
        history_footer = "--- END CONVERSATION HISTORY ---"
        
        try:
            with open(self.config_files["history"], 'r', encoding='utf-8') as f:
                all_histories = json.load(f)
            
            channel_history = all_histories.get(str(channel_id), [])
            recent_history = channel_history[-(self.short_term_memory_turns * 2):]
            
            history_lines = []
            for entry in recent_history:
                if isinstance(entry, dict) and 'role' in entry and 'content' in entry:
                    author_name = entry.get('author_name', 'Unknown')
                    content = entry.get('content', '')
                    timestamp = entry.get('timestamp', '')
                    # Parse the timestamp to make it more readable
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                            history_lines.append(f"[{formatted_time}] {author_name}: {content}")
                        except:
                            history_lines.append(f"{author_name}: {content}")
                    else:
                        history_lines.append(f"{author_name}: {content}")
            
            short_term_memory = "\n".join(history_lines)

        except (FileNotFoundError, json.JSONDecodeError):
            short_term_memory = "No chat history found for this channel."

        context_string = (
            f"### My Long-Term Notes on <@{user_id}>:\n{user_notes}\n\n"
            f"{history_header}\n"
            f"{short_term_memory}\n"
            f"{history_footer}\n\n"
            "--- BEGIN CURRENT USER PROMPT ---"
        )
        
        return [context_string]

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
            else: 
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
        bot_response_text = full_text # Default to the full response
        
        if files_to_send: # This block is for generated images
            bot_response_text = re.sub(r'\[Image of[^\]]+\]', '', full_text).strip() or "Here is the image you requested!"
        else: # This block is for text, including code
            code_block_match = re.search(r"```(?:\w+)?\n([\s\S]+?)\n```", full_text)
            if code_block_match:
                code_content = code_block_match.group(1)
                lang_match = re.search(r"```(\w+)", full_text)
                extension = lang_match.group(1) if lang_match else "txt"
                files_to_send.append(discord.File(io.BytesIO(code_content.encode('utf-8')), filename=f"code.{extension}"))
                # The response text is already the full text from the AI.
                # This sends the explanation and the code, plus the file.

        await self._send_long_message(ctx, bot_response_text, files=files_to_send if files_to_send else None)
        return bot_response_text

    # --- Commands ---
    @commands.command(name="gemini", description="Ask a question to the Gemini model, or continue the conversation.")
    async def gemini(self, ctx: commands.Context, *, prompt: str = None):
        await ctx.defer()
        
        user = ctx.author
        channel = ctx.channel
        is_dm = ctx.guild is None
        is_allowed_channel = not is_dm and channel.id in self.allowed_channel_ids

        final_prompt_parts = []
        if is_allowed_channel:
            final_prompt_parts.extend(self._build_context_prompt(user.id, channel.id))

        user_content = prompt or ""
        # Handle attachments
        for attachment in ctx.message.attachments:
            # Handle images
            if attachment.content_type and 'image' in attachment.content_type:
                try:
                    img_bytes = await attachment.read()
                    final_prompt_parts.append(Image.open(io.BytesIO(img_bytes)))
                    if not prompt: user_content = "[User sent an image.]"
                except Exception as e:
                    await ctx.followup.send(f"I had trouble reading that image file. Error: {e}")
                    return
            # Handle text-based files
            elif attachment.content_type and attachment.content_type.startswith('text/'):
                try:
                    file_bytes = await attachment.read()
                    file_text = file_bytes.decode('utf-8')
                    user_content += f"\n\n--- ATTACHED FILE: {attachment.filename} ---\n{file_text}\n--- END FILE ---"
                except Exception as e:
                    await ctx.followup.send(f"I had trouble reading the text file '{attachment.filename}'. Error: {e}")
                    return
        
        if not user_content and not any(isinstance(p, Image.Image) for p in final_prompt_parts):
            user_content = "[Continuation without text]"

        final_prompt_parts.append(f"\n{user.display_name} (<@{user.id}>): {user_content}\n--- END CURRENT USER PROMPT ---")
        
        try:
            response = await self._generate_with_full_rotation(final_prompt_parts, persona=self.persona)
            if not response or not response.parts:
                await ctx.followup.send("My response was blocked or empty. This might be due to safety filters.")
                return

            final_bot_text = await self._process_and_send_response(ctx, response)
            
            if is_allowed_channel:
                self._log_to_chat_history(channel.id, user, ctx.message.created_at, user_content, final_bot_text)
                # Create a more concise summary for note updating
                user_prompt_preview = user_content[:200] + "..." if len(user_content) > 200 else user_content
                bot_response_preview = final_bot_text[:200] + "..." if len(final_bot_text) > 200 else final_bot_text
                summary = f"User Prompt: '{user_prompt_preview}'\nBot Response: '{bot_response_preview}'"
                await self._update_notes_with_gemini(user.id, summary)
                
        except Exception as e:
            await ctx.followup.send(f"An unexpected error occurred: `{e}`")

    # --- Event Listener ---
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or (message.guild and message.channel.id not in self.allowed_channel_ids):
            return

        mention_triggers = ['1392960230228492508', 'gem']
        content_lower = message.content.lower()
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_reply_to_bot = message.reference and message.reference.resolved and message.reference.resolved.author == self.bot.user
        contains_mention = self.bot.user.mentioned_in(message) or any(word in content_lower for word in mention_triggers)

        if not (is_dm or is_reply_to_bot or contains_mention):
            return

        async with message.channel.typing():
            user_id = message.author.id
            channel_id = message.channel.id
            
            prompt_parts = self._build_context_prompt(user_id, channel_id)
            
            user_content = message.content
            # Handle attachments
            for attachment in message.attachments:
                # Handle images
                if attachment.content_type and 'image' in attachment.content_type:
                    try:
                        prompt_parts.append(Image.open(io.BytesIO(await attachment.read())))
                    except Exception as e:
                        print(f"Could not process image in on_message: {e}")
                        user_content += f"\n[Attachment Error: Failed to read image '{attachment.filename}']"
                # Handle text files
                elif attachment.content_type and attachment.content_type.startswith('text/'):
                    try:
                        file_bytes = await attachment.read()
                        file_text = file_bytes.decode('utf-8')
                        user_content += f"\n\n--- ATTACHED FILE: {attachment.filename} ---\n{file_text}\n--- END FILE ---"
                    except Exception as e:
                        print(f"Could not process text file in on_message: {e}")
                        user_content += f"\n[Attachment Error: Failed to read text file '{attachment.filename}']"
                else:
                    user_content += f" (Note: User also attached a non-image file named '{attachment.filename}')"

            prompt_parts.append(f"\n{message.author.display_name} (<@{user_id}>): {user_content}\n--- END CURRENT USER PROMPT ---")

            try:
                response = await self._generate_with_full_rotation(prompt_parts, persona=self.persona)
                if not response or not response.parts:
                    await message.reply("My response was blocked or empty. This might be due to safety filters.")
                    return
                
                final_bot_text = await self._process_and_send_response(message, response)
                
                self._log_to_chat_history(channel_id, message.author, message.created_at, message.content, final_bot_text)
                # Create a more concise summary for note updating
                user_prompt_preview = message.content[:200] + "..." if len(message.content) > 200 else message.content
                bot_response_preview = final_bot_text[:200] + "..." if len(final_bot_text) > 200 else final_bot_text
                summary = f"User Prompt: '{user_prompt_preview}'\nBot Response: '{bot_response_preview}'"
                await self._update_notes_with_gemini(user_id, summary)

            except Exception as e:
                await message.reply(f"An unexpected error occurred: `{e}`")

async def setup(bot: commands.Bot):
    await bot.add_cog(GeminiCog(bot))