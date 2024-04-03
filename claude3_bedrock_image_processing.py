import base64
import boto3
import json
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

bedrock_clinet=boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
model_id="anthropic.claude-3-sonnet-20240229-v1:0"

def convert_img_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_img = base64.b64encode(image_data).decode('utf-8')
    return base64_img

def invoke_claude_3_with_text(base64_img,user_prompt):
    # Invoke Claude 3 with the text prompt
    try:
        response = bedrock_clinet.invoke_model(
            modelId=model_id,
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 5000,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": user_prompt},{
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_img,
                            },
                        }],
                        }
                    ],
                }
            ),
        )

        # Process and print the response
        result = json.loads(response.get("body").read())
        input_tokens = result["usage"]["input_tokens"]
        output_tokens = result["usage"]["output_tokens"]
        output_list = result.get("content", [])

        print("Invocation details:")
        print(f"- The input length is {input_tokens} tokens.")
        print(f"- The output length is {output_tokens} tokens.")

        print(f"- The model returned {len(output_list)} response(s):")
   
        return result['content'][0]['text']

    except Exception as err:
        print(err)


def handler(event):
    image_path=event["image_path"]
    user_prompt=event["user_prompt"]
    base64_img=convert_img_base64(image_path)
    result=invoke_claude_3_with_text(base64_img,user_prompt)

    print(f"***** IMAGE SOLUTION HERE ****** \n  {result}")


event={"image_path":"image.png","user_prompt":" You are an AI assistant specializing in Unreal Engine game development. You can help users with a wide range of topics related to Unreal Engine including Game development concepts and workflows - Blueprints and visual scripting .Hello! Working on a game where you move in a large vehicle and while the code works, the movement is INCREDIBLY choppy, is there a smoother method to do this?"}

handler(event)
