import base64
import boto3
import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import BedrockChat
from langchain_core.messages import HumanMessage


bedrock_runtime = boto3.client(service_name="bedrock-runtime",region_name="us-west-2")

model_id="anthropic.claude-3-sonnet-20240229-v1:0"

with open("image.png", "rb") as image_file:
        image_data = image_file.read()
        base64_img = base64.b64encode(image_data).decode('utf-8')


model_kwargs =  { 
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}
model = BedrockChat(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)
messages=[
{
    "role": "user",
    "content": [{"type": "text", "text": "Hello! Working on a game where you move in a large vehicle and while the code works, the movement is INCREDIBLY choppy, is there a smoother method to do this?"},
                {
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/png",
        "data": base64_img,
    },
}],
}]
messages = [
    ("system", "You are an AI assistant specializing in Unreal Engine game development. You can help users with a wide range of topics related to Unreal Engine, including:"),
    ("human", f"{base64_img}"),
]

prompt = ChatPromptTemplate.from_messages(messages)

chain = prompt

response = chain.invoke({"question": f"describe image and give me solution "})
print("response----------------",response)

