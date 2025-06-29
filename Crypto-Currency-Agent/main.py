import chainlit as cl
import requests
import os
from dotenv import load_dotenv
from agents import Runner, Agent, AsyncOpenAI, OpenAIChatCompletionsModel,RunConfig, function_tool

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# 🛠️ Tool 1: Show top 10 coin prices
@function_tool
def show_top_prices(dummy: str = "") -> str:
    """Fetch and return top 10 crypto prices from Binance."""
    try:
        response = requests.get("https://api.binance.com/api/v3/ticker/price")
        response.raise_for_status()
        data = response.json()

        top_10 = data[:10]
        return "\n".join([f"💰 {item['symbol']}: ${item['price']}" for item in top_10])

    except requests.exceptions.RequestException as e:
        return f"❌ Error fetching prices: {str(e)}"

# 🛠️ Tool 2: Get specific coin price
@function_tool
def show_specific_coin_price(symbol: str) -> str:
    """Fetch and return price of a specific crypto like BTCUSDT."""
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return f"📌 Price of {symbol.upper()}: ${data['price']}"
        else:
            return f"❌ Coin '{symbol.upper()}' not found."

    except requests.exceptions.RequestException as e:
        return f"❌ Error fetching price: {str(e)}"

# 🤖 Crypto Assistant Agent
crypto_agent = Agent(
    name="Crypto Assistant",
    instructions="""
You are a real-time crypto assistant. Help users by:
- Showing live prices of top cryptocurrencies
- Giving price of specific coins like BTCUSDT
Be concise, polite, and ensure information is accurate and updated using the tools provided.
""",
    tools=[show_top_prices, show_specific_coin_price],
    model=model
)

# 💬 Handle Chainlit user messages
@cl.on_message
async def handle_message(message: cl.Message):
    result = Runner.run_sync(
        starting_agent=crypto_agent,
        input=message.content,
        run_config=config
    )

    await cl.Message(
        content=f"🧠 **Crypto Assistant Response:**\n\n{result.final_output}"
    ).send()
