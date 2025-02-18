from bot.discord_bot import DiscordBot
from bot.telegram_bot import TelegramBot
from bot.whatsapp_bot import WhatsAppBot

def main():
    # Initialize the bots
    discord_bot = DiscordBot()
    telegram_bot = TelegramBot()
    whatsapp_bot = WhatsAppBot()

    # Start the bots
    discord_bot.run()
    telegram_bot.start()
    whatsapp_bot.start()

if __name__ == "__main__":
    main()