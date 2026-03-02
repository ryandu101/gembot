require('dotenv').config();
const { Client, GatewayIntentBits } = require('discord.js');
const fs = require('fs');
const path = require('path');

// --- Configuration Loading ---
const SCRIPT_DIR = __dirname;
const CONFIG_FILES = {
    persona: path.join(SCRIPT_DIR, 'persona.txt'),
    meta_persona: path.join(SCRIPT_DIR, 'meta_persona.txt'),
    channels: path.join(SCRIPT_DIR, 'channel_ids.json'),
    history: path.join(SCRIPT_DIR, 'chat_history.json'),
    notes: path.join(SCRIPT_DIR, 'user_notes.json')
};

// --- Bot Definition ---
const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent, 
        GatewayIntentBits.GuildMembers,
    ]
});

client.configFiles = CONFIG_FILES;

// Load allowed channels into memory
try {
    const channelData = JSON.parse(fs.readFileSync(CONFIG_FILES.channels, 'utf8'));
    client.allowedChannels = channelData.allowed_channel_ids || [];
} catch (e) {
    client.allowedChannels = [];
    console.log("No allowed channels found or channel_ids.json is missing.");
}

// --- Load Events ---
const eventsPath = path.join(__dirname, 'events');
if (fs.existsSync(eventsPath)) {
    const eventFiles = fs.readdirSync(eventsPath).filter(file => file.endsWith('.js'));
    for (const file of eventFiles) {
        const filePath = path.join(eventsPath, file);
        const event = require(filePath);
        if (event.once) {
            client.once(event.name, (...args) => event.execute(...args, client));
        } else {
            client.on(event.name, (...args) => event.execute(...args, client));
        }
    }
}

client.once('clientReady', () => {
    console.log(`✅ Logged in as ${client.user.tag}!`);
});

client.login(process.env.DISCORD_TOKEN);