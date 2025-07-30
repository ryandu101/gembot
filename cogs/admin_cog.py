import discord
from discord.ext import commands
import json
import os
import google.generativeai as genai

class AdminCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config_files = bot.config_files
        
        # --- Load necessary configs ---
        self.meta_persona = self._load_text_file(self.config_files["meta_persona"])
        _, self.gemini_api_keys = self._load_keys(self.config_files["keys"])

        # --- Safety Settings ---
        self.safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
        print("AdminCog loaded.")

    # --- Helper Methods ---
    def _load_text_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: return f.read()
        except FileNotFoundError: return ""

    def _load_keys(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                keys = json.load(f)
                return keys.get("discord_token"), keys.get("gemini_keys", [])
        except Exception: return None, []

    def _create_meta_model(self):
        # Admin commands can use the first key, assuming it has quota
        api_key = self.gemini_api_keys[0]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            'gemini-2.5-pro',
            system_instruction=self.meta_persona,
            safety_settings=self.safety_settings
        )

    def _save_user_notes(self, user_id, notes):
        try:
            with open(self.config_files["notes"], 'r') as f:
                all_notes = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_notes = {}
        all_notes[str(user_id)] = notes
        with open(self.config_files["notes"], 'w') as f:
            json.dump(all_notes, f, indent=4)

    # --- Owner-Only Commands ---

    @commands.hybrid_command(name="reload", description="[Owner] Reloads a specified cog or all cogs.")
    @commands.is_owner()
    async def reload(self, ctx: commands.Context, cog: str = None):
        """Reloads cogs dynamically."""
        await ctx.defer(ephemeral=True)
        
        if cog:
            # Reload a specific cog
            cog_name = f"cogs.{cog}"
            try:
                await self.bot.reload_extension(cog_name)
                await ctx.send(f"✅ Successfully reloaded cog: `{cog_name}`", ephemeral=True)
                print(f"Cog reloaded by {ctx.author}: {cog_name}")
            except commands.ExtensionNotLoaded:
                await ctx.send(f"❌ Error: Cog `{cog_name}` is not loaded.", ephemeral=True)
            except commands.ExtensionNotFound:
                await ctx.send(f"❌ Error: Cog `{cog_name}` not found. Check the name.", ephemeral=True)
            except Exception as e:
                await ctx.send(f"An error occurred while reloading `{cog_name}`: \n`{e}`", ephemeral=True)
        else:
            # Reload all cogs
            reloaded_cogs = []
            failed_cogs = []
            for filename in os.listdir(os.path.join(self.bot.script_dir, 'cogs')):
                if filename.endswith('.py') and not filename.startswith('__'):
                    cog_name = f"cogs.{filename[:-3]}"
                    try:
                        await self.bot.reload_extension(cog_name)
                        reloaded_cogs.append(f"`{cog_name}`")
                    except Exception as e:
                        failed_cogs.append(f"`{cog_name}`")
                        print(f"Failed to reload {cog_name}: {e}")
            
            response = ""
            if reloaded_cogs:
                response += f"✅ **Reloaded:** {', '.join(reloaded_cogs)}\n"
            if failed_cogs:
                response += f"❌ **Failed:** {', '.join(failed_cogs)}"
            
            await ctx.send(response or "No cogs found to reload.", ephemeral=True)
            print(f"All cogs reloaded by {ctx.author}.")


    @commands.hybrid_command(name="build_chat_history", description="[Owner] Rebuilds chat history for this channel.")
    @commands.is_owner()
    async def build_chat_history(self, ctx: commands.Context):
        await ctx.defer(ephemeral=True)
        channel_id_str = str(ctx.channel.id)
        print(f"Starting chat history rebuild for channel {channel_id_str}...")

        try:
            with open(self.config_files["history"], 'r') as f:
                all_histories = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_histories = {}

        new_channel_history = []
        message_count = 0
        async for message in ctx.channel.history(limit=10000, oldest_first=True):
            if not message.content and not message.attachments: continue
            role = 'model' if message.author.bot else 'user'
            entry = {
                "role": role, "author_id": message.author.id, "author_name": message.author.display_name,
                "timestamp": message.created_at.isoformat(), "content": message.content
            }
            new_channel_history.append(entry)
            message_count += 1
            if message_count % 500 == 0: print(f"Processed {message_count} messages...")

        all_histories[channel_id_str] = new_channel_history
        with open(self.config_files["history"], 'w') as f:
            json.dump(all_histories, f, indent=4)
        
        await ctx.send(f"Successfully rebuilt chat history, processing {message_count} messages.", ephemeral=True)

    @commands.hybrid_command(name="initialize_notes", description="[Owner] Creates initial user notes from all chat history.")
    @commands.is_owner()
    async def initialize_notes(self, ctx: commands.Context):
        await ctx.defer(ephemeral=True)
        print("Starting user notes initialization from history...")

        try:
            with open(self.config_files["history"], 'r') as f:
                all_histories = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            await ctx.send(f"Error: The chat history file could not be read.", ephemeral=True)
            return

        user_conversations = {}
        for channel_id, messages in all_histories.items():
            for message in messages:
                if isinstance(message, dict) and message.get('role') == 'user':
                    user_id = str(message.get('author_id'))
                    if user_id not in user_conversations: user_conversations[user_id] = []
                    if content := message.get('content'): user_conversations[user_id].append(content)

        if not user_conversations:
            await ctx.send("No user conversations found in chat history.", ephemeral=True)
            return

        meta_model = self._create_meta_model()
        initialized_users = 0
        for user_id, texts in user_conversations.items():
            print(f"Generating notes for user {user_id}...")
            full_conversation = "\n".join(texts)
            prompt = f"Analyze the following conversation from user <@{user_id}> and create concise notes.\n\n{full_conversation}"
            
            try:
                response = await meta_model.generate_content_async(prompt)
                if response.text:
                    self._save_user_notes(user_id, response.text)
                    initialized_users += 1
            except Exception as e:
                print(f"Error generating notes for user {user_id}: {e}")
                
        await ctx.send(f"Initialization complete! Created/updated notes for {initialized_users} user(s).", ephemeral=True)

    @commands.hybrid_command(name="createnotes", description="[Owner] Creates/rewrites notes for a specific user.")
    @commands.is_owner()
    async def createnotes(self, ctx: commands.Context, user: discord.User):
        await ctx.defer(ephemeral=True)
        target_user_id = user.id
        print(f"Starting note creation for user {target_user_id} from channel {ctx.channel.id}...")

        user_messages = []
        async for message in ctx.channel.history(limit=5000):
            if message.author.id == target_user_id:
                user_messages.append(message.content)

        if not user_messages:
            await ctx.send(f"No recent messages found for user <@{target_user_id}> in this channel.", ephemeral=True)
            return

        user_messages.reverse()
        full_conversation = "\n".join(user_messages)
        meta_model = self._create_meta_model()
        prompt = f"Analyze the following conversation from user <@{target_user_id}> and create concise notes.\n\n{full_conversation}"

        try:
            response = await meta_model.generate_content_async(prompt)
            if response.text:
                self._save_user_notes(target_user_id, response.text)
                await ctx.send(f"Successfully created/rewrote notes for <@{target_user_id}>.", ephemeral=True)
            else:
                await ctx.send(f"Failed to generate notes for <@{target_user_id}> (empty response).", ephemeral=True)
        except Exception as e:
            await ctx.send(f"An error occurred while generating notes: {e}", ephemeral=True)

async def setup(bot: commands.Bot):
    await bot.add_cog(AdminCog(bot))
