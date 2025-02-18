import os
from dotenv import load_dotenv
import discord
import telegram
from telegram.ext import Updater, MessageHandler, Filters
import asyncio
from heyoo import WhatsApp

# Load environment variables
load_dotenv()

# Get tokens from .env
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
WHATSAPP_TOKEN = os.getenv('WHATSAPP_TOKEN')

# Discord bot setup
class DiscordBot(discord.Client):
    async def on_ready(self):
        print(f'{self.user} has connected to Discord!')

    async def on_message(self, message):
        if message.author == self.user:
            return
        await message.channel.send(f'Received message: {message.content}')

# Telegram bot setup
def telegram_message_handler(update, context):
    message = update.message.text
    update.message.reply_text(f'Received message: {message}')

def run_telegram_bot():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.text, telegram_message_handler))
    updater.start_polling()
    print('Telegram bot is running!')

# WhatsApp bot setup
def run_whatsapp_bot():
    messenger = WhatsApp(WHATSAPP_TOKEN, phone_number_id=os.getenv('WHATSAPP_NUMBER'))
    print('WhatsApp bot is running!')
    return messenger

# Main function to run all bots
async def main():
    # Initialize Discord bot
    discord_client = DiscordBot(intents=discord.Intents.default())
    
    # Run all bots
    await asyncio.gather(
        discord_client.start(DISCORD_TOKEN),
        asyncio.to_thread(run_telegram_bot),
        asyncio.to_thread(run_whatsapp_bot)
    )

if __name__ == "__main__":
    asyncio.run(main())
