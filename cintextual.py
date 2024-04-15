from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.llms.bedrock import Bedrock
import boto3
from langchain.llms import OpenAI

from langchain.prompts.prompt import PromptTemplate

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI Assistant:"""

PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

REGION_NAME = "us-west-2"
FMC_URL = "https://bedrock-runtime.us-west-2.amazonaws.com"
bedrock = boto3.client("bedrock-runtime", region_name=REGION_NAME)
llm = Bedrock(
    client=bedrock,
    model_id='anthropic.claude-v2:1',
    endpoint_url=FMC_URL,
    model_kwargs={"temperature": 0.7, "max_tokens_to_sample": 5000}
)

memory = ConversationBufferWindowMemory(k=5)  # Store the last 5 interactions

chat_history = [
    ("role", "Hello, how are you?"),
    ("content", "I'm doing well, thanks for asking!"),
    ("role", "That's good to hear. Can you tell me about your capabilities?"),
    ("content", "Sure, I'm an AI assistant trained to help with a wide range of tasks such as writing, analysis, coding, and problem-solving."),
    ("role", "what is python"),
    ("content", "Python is a popular programming language used for a variety of applications such as web development, data analysis, artificial intelligence, scientific computing, and more."),
    ("role", "what is epic game unreal engine?"),
    ("content", "Epic Games' Unreal Engine is a powerful game development platform used to create high-quality, interactive 3D and 2D games. It provides a wide range of tools and features for developers to design and build games for various platforms such as PC, console, mobile, and virtual reality. Unreal Engine is known for its advanced graphics capabilities, strong community support, and user-friendly interface.")
]

# Load the chat history into the memory
memory.load_memory_variables({})  # Initialize with an empty dictionary
for role, message in chat_history:
    if role == "role":
        human_input = message
    else:
        ai_output = message
        memory.save_context({"human_input": human_input}, {"output": ai_output})

# Now you can access the chat history stored in the memory
print(memory.buffer)

conversation_with_memory = ConversationChain(
    llm=llm, memory=memory, verbose=True
)
response = conversation_with_memory.predict(input="give me more details ")
print("Claude: " + response)