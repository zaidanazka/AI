class TelegramBot:
    def __init__(self, token):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{self.token}/"

    def send_message(self, chat_id, text):
        url = f"{self.base_url}sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text
        }
        response = requests.post(url, json=payload)
        return response.json()

    def get_updates(self):
        url = f"{self.base_url}getUpdates"
        response = requests.get(url)
        return response.json()

    def handle_messages(self):
        updates = self.get_updates()
        for update in updates['result']:
            chat_id = update['message']['chat']['id']
            message_text = update['message']['text']
            # Process the message and generate a response
            response_text = self.process_message(message_text)
            self.send_message(chat_id, response_text)

    def process_message(self, message):
        # Implement AI logic here using DeepSeek AI
        return f"You said: {message}"  # Placeholder response