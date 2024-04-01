from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import boto3
import time
import json
import os

REGION_NAME = os.environ.get('REGION_NAME')
FMC_URL = os.environ['FMC_URL']
MODEL_ID=os.environ['MODEL_ID']

bedrock = boto3.client("bedrock-runtime","us-west-2")
 
def load_history_from_json(chat_history):
  try:
      json_chat_history=json.dumps(chat_history)
      print("json_chat_history",json_chat_history)
      return json_chat_history
  except Exception as e:
    print(f"Failed Load Json  :-  {e}")
    return "Pass valid Json"
 
def update_and_get_response(user_prompt, messages, llm_chain):
  try:
    """Processes user input, updates history, and generates response."""
    messages.append({"role": "user", "content": user_prompt})
  except Exception as e:
     print(f"Failed to append user_prompt into chat history {e} ")
     return "Failed to append user_prompt into chat history"
  
  try:
     
    if user_prompt.lower() == "stop":
      print("User stopped the conversation.")
    else:
      ai_response = llm_chain.predict(question=user_prompt)
      new_ai_message = {"role": "assistant", "content": ai_response}
      messages.append(new_ai_message)
      print(f"{new_ai_message['role']}: {new_ai_message['content']}")
  except Exception as e:
     print(f"Failed to generate ai response {e}")
     return "Model Busy ..."
 
  return ai_response,new_ai_message,messages
 
prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
        You are an AI assistant specializing in Unreal Engine game development. You can help users with a wide range of topics related to Unreal Engine, including:

        - Game development concepts and workflows
        - Blueprints and visual scripting
        - C++ programming for games
        - Asset creation and management (meshes, textures, materials, etc.)
        - Level design and world building
        - Lighting and post-processing effects
        - Animation and character setup
        - Physics and collision detection
        - Networking and multiplayer functionality
        - Performance optimization and profiling
        - Debugging and troubleshooting application issues
        - Graphical rendering problems and solutions
        - Integration with other tools and platforms
        
        - Best practices and tips for Unreal Engine development

        You have deep knowledge of the Unreal Engine codebase, documentation, and community resources. You can provide code examples, visual references, and step-by-step guidance to help users solve their problems or achieve their goals.

        When users ask questions, feel free to ask for clarification or additional details if needed. If users share code snippets, images, or error logs, you can analyze them and provide targeted advice.

        Your goal is to be a friendly, knowledgeable, and patient guide for Unreal Engine developers of all skill levels. You should encourage users to learn and experiment while offering practical solutions to their challenges.
    
        Current conversation: {chat_history}
    
        Human: {question}
        """
    )

def contextual_chat(chat_history,user_prompt):

    messages = chat_history # load_history_from_json(chat_history)
    llm = Bedrock(
    client=bedrock,
        model_id= MODEL_ID,
        endpoint_url=FMC_URL,
        model_kwargs={"temperature": 0.7, "max_tokens_to_sample": 5000}
    )

    memory = ConversationBufferWindowMemory(memory_key="chat_history")
    llm_chain = LLMChain(llm=llm, memory=memory, prompt=prompt)
    
    ai_response,new_ai_message,messages = update_and_get_response(user_prompt, messages, llm_chain)

    return ai_response,new_ai_message,messages 

def handler(event, context):
    try:
        print(f"EVENT - {event}")
        chat_history=event['chat_history']
        user_prompt=event['user_prompt']
        ai_response,new_ai_message,messages = contextual_chat(chat_history,user_prompt)
        LLM_RESPONSE={} 
        LLM_RESPONSE["ai_response"]=ai_response
        LLM_RESPONSE["conversion"]=messages
        LLM_RESPONSE["Last_prompt"]=user_prompt
        print("ai_response",LLM_RESPONSE)
        return LLM_RESPONSE
    
    except Exception as e:
        print(f"Error -- {e}")
        return "Failed to get answer From LLM Model ..."
