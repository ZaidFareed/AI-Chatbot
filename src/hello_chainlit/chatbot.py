import chainlit as cl
import os
from openai.types.responses import ResponseTextDeltaEvent

from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Set up the OpenAI provider for Gemini
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Define the model configuration
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# Define your agent behavior
agent1 = Agent(
    instructions="""You are an AI Professor. Users ask you to explain or explore AI-related concepts. 
    You provide clear, concise, and engaging explanations that help learners grasp complex topics easily.
    Ask clarifying questions if needed, and guide users to think critically. Keep a friendly and approachable tone.Give a medium length answer 
    Try to give a complete answer in a single response when possible.Don't mention any religious word.Act like a professor while greeting also
    """,
    name="AI Professor"
)

# Test run (optional – remove in production)
result = Runner.run_sync(
    input="What is the capital of France?",
    run_config=run_config,
    starting_agent=agent1
)

print(result)

# Chainlit event: on chat start
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    
    # Display a heading and welcome message
    await cl.Message(content="## 👨‍🏫 ProfAI: Making Complex AI Concepts Simple\n_Ask me anything about AI, machine learning, or data science!_").send()

# Chainlit event: on new message
@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    msg = cl.Message(content="")
    await msg.send()

    history.append({"role": "user", "content": message.content})
    result = Runner.run_streamed(
        agent1,
        input=history,
        run_config=run_config,
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    # Save assistant response to history
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
