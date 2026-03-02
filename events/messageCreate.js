const fs = require('fs').promises;
const { AttachmentBuilder } = require('discord.js');
const geminiManager = require('../utils/geminiManager');

module.exports = {
    name: 'messageCreate',
    async execute(message, client) {
        if (message.author.bot) return;

        const isDM = !message.guild;
        const isAllowedChannel = isDM || client.allowedChannels.includes(message.channel.id);
        
        if (!isAllowedChannel) return;

        const mentionTriggers = ['1392960230228492508', 'gem'];
        const contentLower = message.content.toLowerCase();
        
        // Ensure message.reference exists before checking resolved
        const isReplyToBot = message.reference?.messageId && 
                             message.mentions.repliedUser?.id === client.user.id;
                             
        const containsMention = message.mentions.has(client.user) || 
                                mentionTriggers.some(word => contentLower.includes(word));

        if (!isDM && !isReplyToBot && !containsMention) return;

        // Visual indicator that the bot is thinking
        await message.channel.sendTyping();

        try {
            const userId = message.author.id;
            const channelId = message.channel.id;

            // 1. Load Persona
            let persona = await fs.readFile(client.configFiles.persona, 'utf8').catch(() => "");
            
            // 2. Build Context (Short-term and Long-term memory)
            const contextString = await buildContextPrompt(userId, channelId, client.configFiles);
            let promptParts = [contextString];

            // 3. Process the current message and attachments
            let userContent = message.content || "[Continuation without text]";
            
            for (const [id, attachment] of message.attachments) {
                if (attachment.contentType && attachment.contentType.startsWith('image/')) {
                    // Fetch image and convert to base64 for the new SDK
                    const response = await fetch(attachment.url);
                    const arrayBuffer = await response.arrayBuffer();
                    const buffer = Buffer.from(arrayBuffer);
                    
                    promptParts.push({
                        inlineData: {
                            data: buffer.toString('base64'),
                            mimeType: attachment.contentType
                        }
                    });
                } else if (attachment.contentType && attachment.contentType.startsWith('text/')) {
                    const response = await fetch(attachment.url);
                    const textContent = await response.text();
                    userContent += `\n\n--- ATTACHED FILE: ${attachment.name} ---\n${textContent}\n--- END FILE ---`;
                }
            }

            promptParts.push(`\n${message.author.displayName} (<@${userId}>): ${userContent}\n--- END CURRENT USER PROMPT ---`);

            // 4. Generate Response
            const response = await geminiManager.generateContentWithRotation(promptParts, persona);
            
            if (!response || !response.text) {
                return message.reply("My response was blocked or empty. This might be due to safety filters.");
            }

            const botText = response.text;

            // 5. Send Chunked Message (Discord 2000 character limit)
            await sendLongMessage(message, botText);

            // 6. Async background tasks: Save history and update notes
            logToChatHistory(channelId, message.author, userContent, botText, client.configFiles).catch(console.error);
            updateNotesWithGemini(userId, userContent, botText, client.configFiles).catch(console.error);

        } catch (error) {
            console.error("Error in messageCreate:", error);
            message.reply(`An error occurred: ${error.message}`);
        }
    }
};

// --- Helper Functions ---

async function buildContextPrompt(userId, channelId, configFiles) {
    let userNotes = "No notes on this user yet.";
    try {
        const notesData = JSON.parse(await fs.readFile(configFiles.notes, 'utf8'));
        userNotes = notesData[userId] || userNotes;
    } catch (e) {}

    let shortTermMemory = "No chat history found for this channel.";
    try {
        const historyData = JSON.parse(await fs.readFile(configFiles.history, 'utf8'));
        const channelHistory = historyData[channelId] || [];
        // Grab the last 1000 messages (500 turns)
        const recentHistory = channelHistory.slice(-1000); 
        
        shortTermMemory = recentHistory.map(entry => {
            const time = new Date(entry.timestamp).toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
            return `[${time}] ${entry.author_name}: ${entry.content}`;
        }).join('\n');
    } catch (e) {}

    return `### My Long-Term Notes on <@${userId}>:\n${userNotes}\n\n--- BEGIN CONVERSATION HISTORY ---\n${shortTermMemory}\n--- END CONVERSATION HISTORY ---\n\n--- BEGIN CURRENT USER PROMPT ---`;
}

async function sendLongMessage(message, text) {
    // Basic regex to strip out image generation tags if necessary, just like your python code
    const cleanText = text.replace(/\[Image of[^\]]+\]/g, '').trim() || "Done!";
    
    // Chunking to 2000 characters
    const chunks = cleanText.match(/[\s\S]{1,1999}/g) || [];
    for (let i = 0; i < chunks.length; i++) {
        if (i === 0) await message.reply(chunks[i]);
        else await message.channel.send(chunks[i]);
    }
}

async function logToChatHistory(channelId, author, userContent, botText, configFiles) {
    let history = {};
    try { history = JSON.parse(await fs.readFile(configFiles.history, 'utf8')); } catch (e) {}

    if (!history[channelId]) history[channelId] = [];

    history[channelId].push({
        role: "user", author_id: author.id, author_name: author.displayName,
        timestamp: new Date().toISOString(), content: userContent
    });
    
    history[channelId].push({
        role: "model", author_id: "bot", author_name: "Gembot",
        timestamp: new Date().toISOString(), content: botText
    });

    await fs.writeFile(configFiles.history, JSON.stringify(history, null, 4));
}

async function updateNotesWithGemini(userId, userContent, botText, configFiles) {
    // Read meta persona
    let metaPersona = await fs.readFile(configFiles.meta_persona, 'utf8').catch(() => "");
    
    let existingNotes = "No notes on this user yet.";
    let allNotes = {};
    try { 
        allNotes = JSON.parse(await fs.readFile(configFiles.notes, 'utf8')); 
        existingNotes = allNotes[userId] || existingNotes;
    } catch (e) {}

    const conciseSummary = `User Prompt: '${userContent.substring(0, 200)}'\nBot Response: '${botText.substring(0, 200)}'`;
    
    const updatePrompt = `**Existing Notes on User <@${userId}>:**\n${existingNotes}\n\n**Recent Conversation Summary:**\n${conciseSummary}\n\nPlease update the notes based on this new information. Follow your persona guidelines strictly: keep notes under 2000 characters, summarize rather than quote, focus on key facts and personality traits, and maintain the structured format with markdown headings.`;

    try {
        const response = await geminiManager.generateContentWithRotation([updatePrompt], metaPersona);
        if (response && response.text) {
            let newNotes = response.text.trim();
            if (newNotes.length > 2000) newNotes = newNotes.substring(0, 1997) + "...";
            allNotes[userId] = newNotes;
            await fs.writeFile(configFiles.notes, JSON.stringify(allNotes, null, 4));
        }
    } catch (e) {
        console.error(`Failed to update notes for ${userId}:`, e);
    }
}