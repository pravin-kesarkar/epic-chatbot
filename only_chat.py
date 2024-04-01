from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.llms.bedrock import Bedrock
import boto3
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(openai_api_key="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")


memory = ConversationBufferWindowMemory(k=8)  # Store the last 5 interactions

chat_history = [
    ("Human", "Hello, how are you?"),
    ("AI", "I'm doing well, thanks for asking!"),
    ("Human", "That's good to hear. Can you tell me about your capabilities?"),
    ("AI", "Sure, I'm an AI assistant trained to help with a wide range of tasks such as writing, analysis, coding, and problem-solving."),
    ("Human", "what is python"),
    ("AI", "Python is a popular programming language used for a variety of applications such as web development, data analysis, artificial intelligence, scientific computing, and more."),
    ("Human", "what is epic game unreal engine?"),
    ("AI", "Epic Games' Unreal Engine is a powerful game development platform used to create high-quality, interactive 3D and 2D games. It provides a wide range of tools and features for developers to design and build games for various platforms such as PC, console, mobile, and virtual reality. Unreal Engine is known for its advanced graphics capabilities, strong community support, and user-friendly interface.")
]

# Load the chat history into the memory
memory.load_memory_variables({})  # Initialize with an empty dictionary
for role, message in chat_history:
    if role == "Human":
        human_input = message
    else:
        ai_output = message
        memory.save_context({"human_input": human_input}, {"output": ai_output})

# Now you can access the chat history stored in the memory
print(memory.buffer)

conversation_with_memory = ConversationChain(
    llm=llm, memory=memory, verbose=True
)
response = conversation_with_memory.predict(input="who is the founder  ")
print("Claude: " + response)