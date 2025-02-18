# ai-multiplatform-bot

This project is an AI-powered bot that operates across multiple platforms, including Discord, Telegram, and WhatsApp. It utilizes the DeepSeek AI framework to provide intelligent responses and interactions.

## Project Structure

```
ai-multiplatform-bot
├── src
│   ├── bot
│   │   ├── __init__.py
│   │   ├── discord_bot.py
│   │   ├── telegram_bot.py
│   │   └── whatsapp_bot.py
│   ├── ai
│   │   ├── __init__.py
│   │   └── deepseek.py
│   ├── config
│   │   ├── __init__.py
│   │   └── settings.py
│   └── main.py
├── requirements.txt
├── .env
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-multiplatform-bot
   ```

2. **Install dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Create a `.env` file in the root directory and add your API keys and tokens for Discord, Telegram, and WhatsApp.

4. **Run the bot:**
   Execute the main script to start the bot:
   ```bash
   python src/main.py
   ```

## Usage

- The bot will connect to the specified platforms and start listening for messages.
- You can interact with the bot on Discord, Telegram, and WhatsApp.
- The AI logic is handled by the `DeepSeekAI` class, which processes input and generates responses.

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements for the project.