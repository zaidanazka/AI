class DiscordBot:
    def __init__(self, token):
        self.token = token
        self.client = None

    def connect(self):
        import discord
        self.client = discord.Client()

        @self.client.event
        async def on_ready():
            print(f'Logged in as {self.client.user}')

        @self.client.event
        async def on_message(message):
            if message.author == self.client.user:
                return

            # Add your message handling logic here

        self.client.run(self.token)

    def send_message(self, channel_id, message):
        channel = self.client.get_channel(channel_id)
        if channel:
            self.client.loop.create_task(channel.send(message))