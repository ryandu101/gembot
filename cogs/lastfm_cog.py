import discord
from discord.ext import commands
import aiohttp
import json

# This class will hold the logic for fetching data from the Last.fm API.
# It can be used by both the direct command and the Gemini tool.
class LastFm:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://ws.audioscrobbler.com/2.0/"

    async def get_top_artists(self, username: str, limit: int = 5):
        """Fetches the top artists for a given Last.fm username."""
        if not self.api_key:
            return {"error": "Last.fm API key is not configured."}

        params = {
            'method': 'user.gettopartists',
            'user': username,
            'api_key': self.api_key,
            'format': 'json',
            'limit': limit
        }
        
        headers = {
            'User-Agent': 'Gembot/1.0 (Discord Bot)'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'error' in data:
                            return {"error": f"Last.fm API Error: {data['message']}"}
                        
                        # Extract and format the relevant information
                        artists = data.get('topartists', {}).get('artist', [])
                        formatted_artists = [
                            f"{artist['@attr']['rank']}. {artist['name']} ({artist['playcount']} plays)"
                            for artist in artists
                        ]
                        return {"top_artists": formatted_artists}
                    else:
                        return {"error": f"Failed to fetch data from Last.fm. Status: {response.status}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {e}"}

# The cog that adds the slash command to Discord
class LastFmCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.last_fm_api_key = self._load_lastfm_key(self.bot.config_files["keys"])
        self.last_fm = LastFm(self.last_fm_api_key)

    def _load_lastfm_key(self, filepath):
        """Loads the Last.fm API key from the keys.json file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                keys = json.load(f)
                return keys.get("lastfm_api_key")
        except Exception:
            return None

    @commands.hybrid_command(name="lastfm", description="Fetch top artists from Last.fm for a user.")
    async def lastfm(self, ctx: commands.Context, username: str):
        """Allows a user to directly query their Last.fm top artists."""
        await ctx.defer()
        
        result = await self.last_fm.get_top_artists(username)

        if "error" in result:
            await ctx.send(result["error"])
        else:
            artists_list = "\n".join(result['top_artists'])
            embed = discord.Embed(
                title=f"Top Artists for {username}",
                description=artists_list,
                color=discord.Color.red()
            )
            await ctx.send(embed=embed)

async def setup(bot: commands.Bot):
    await bot.add_cog(LastFmCog(bot))