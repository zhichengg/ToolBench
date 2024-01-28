import json
import openai
from openai import OpenAI

from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
import time
import traceback
import random

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(key, messages, functions=None,function_call=None,key_pos=None, model="gpt-3.5-turbo-16k",stop=None,process_id=0, **args):
    client = OpenAI(api_key=key, base_url="https://api.01ww.xyz/v1")
    use_messages = []
    for message in messages:
        if not("valid" in message.keys() and message["valid"] == False):
            # if "function_call" in message:
            #     message["content"] = ""
            #     message["tool_calls"] = []
            #     tool_call = {
            #         "id": "null",
            #         "type": "function",
            #         "function": message["function_call"],
            #     }
            #     del message["function_call"]
            #     message["tool_calls"].append(tool_call)
            use_messages.append(message)

    json_data = {
        "model": model,
        "messages": use_messages,
        "max_tokens": 1024,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        **args
    }
    if stop is not None:
        json_data.update({"stop": stop})
    if functions is not None:
        tools = [
            {"type": "function", "function": f} for f in functions
        ]
        json_data.update({"tools": tools})
    if function_call is not None:
        if function_call == 'auto':
            tool_choice = 'auto'
        else:
            tool_choice = {"type": "function", "function": function_call}
        json_data.update({"tool_choice": tool_choice})
    
    try:
        # if model == "gpt-3.5-turbo-16k":
        #     openai.api_key = key
        # else:
        #     raise NotImplementedError
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(json.dumps(json_data))
        # import pdb; pdb.set_trace()
        openai_response = client.chat.completions.create(**json_data)
        # print(f"[process({process_id})]openai_response: {openai_response}")
        # print(openai_response)
        # json_data = json.loads(str(openai_response))
        json_data = openai_response.dict()
        return json_data 
    except KeyboardInterrupt:
        # raise KeyboardInterrupt
        exit()
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"OpenAI calling Exception: {e}")
        traceback.print_exc()
        return e

class ChatGPTFunction:
    def __init__(self, model="gpt-3.5-turbo-16k", openai_key=""):
        self.model = model
        self.conversation_history = []
        self.openai_key = openai_key
        self.time = time.time()
        self.TRY_TIME = 6

    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self,messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)

    def parse(self,functions,process_id,key_pos=None,**args):
        self.time = time.time()
        conversation_history = self.conversation_history
        for _ in range(self.TRY_TIME):
            if _ != 0:
                time.sleep(15)
            if functions != []:
                json_data = chat_completion_request(
                    self.openai_key, conversation_history, functions=functions,process_id=process_id, key_pos=key_pos, model=self.model, **args
                )
            else:
                json_data = chat_completion_request(
                    self.openai_key, conversation_history,process_id=process_id,key_pos=key_pos, seed=42, model=self.model, **args
                )
            try:
                total_tokens = json_data['usage']['total_tokens']
                message = json_data["choices"][0]["message"]
                if process_id == 0:
                    print(f"[process({process_id})]total tokens: {json_data['usage']['total_tokens']}")

                if message['function_call'] and "." in message["function_call"]["name"]:
                # if "function_call" in message.keys() and "." in message["function_call"]["name"]:
                    message["function_call"]["name"] = message["function_call"]["name"].split(".")[-1]

                return message, 0, total_tokens
            except KeyboardInterrupt:
                exit()
            except BaseException as e:
                print(f"[process({process_id})]Parsing Exception: {repr(e)}. Try again.")
                if json_data is not None:
                    print(f"[process({process_id})]OpenAI return: {json_data}")
            

        return {"role": "assistant", "content": str(json_data)}, -1, 0

if __name__ == "__main__":
    # TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(api_base="https://openai-lcih.onrender.com/v1")'
    # openai.api_base="https://openai-lcih.onrender.com/v1"

    llm = ChatGPTFunction(openai_key="openchat",model="gpt-3.5-turbo-1106")
    prompt = '''下面这句英文可能有语病，能不能把语病都改掉？
If you think you get the result which can answer the task, call this function to give the final answer. Or, if you think you can't handle the task from this status, call this function to restart. Remember: you should ALWAYS call this function at the end of your try, and the final answer is the ONLY part that will be showed to user, so final answer should contain enough information.
没语病的形式：
'''
    messages = [
        # {"role":"system","content":""},
        {"role":"user","content":"写一个中国的故事，一句话就可以。"},
    ]
    llm.change_messages(messages)
    output,error_code,token_usage = llm.parse(functions=[],process_id=0)
    print(output)

    json_data = {
        'model': 'gpt-3.5-turbo-1106', 
        'messages': [{'role': 'system', 'content': 'You are AutoGPT, you can use many tools(functions) to do the following task.\nFirst I will give you the task description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.\nAfter the call, you will get the call result, and you are now in a new state.\nThen you will analyze your status now, then decide what to do next...\nAfter many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.\nRemember: \n1.the state change is irreversible, you can\'t go back to one of the former state, if you want to restart the task, say "I give up and restart".\n2.All the thought is short, at most in 5 sentence.\n3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.\nLet\'s Begin!\nTask description: You should use functions to help handle the real time user querys. Remember:\n1.ALWAYS call "Finish" function at the end of the task. And the final answer should contain enough information to show to the user,If you can\'t handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.\n2.Do not use origin tool names, use only subfunctions\' names.\nYou have access of the following tools:\n1.flight_fare_search: Elevate your travel game with Flight Fare Search API! Get real-time flight data, fares, and airport info for seamless travel planning. Transform your app into a powerful travel companion with Flight Fare Search.\n'}, {'role': 'user', 'content': '\nI need to book a flight from London to Dubai for a business trip. Can you provide me with the flight options available on a specific date using the Flight Search V2 API? Additionally, I would like to search for airports using a specific query using the Airport Search API.\nBegin!\n'}], 
        'max_tokens': 1024, 
        'frequency_penalty': 0, 
        'presence_penalty': 0, 
        'functions': [{'name': 'airport_arrivals_for_flight_fare_search', 'description': 'This is the subfunction for tool "flight_fare_search", you can use this tool.The description of this function is: "An Endpoint to fetch Arrivals on a given date"', 'parameters': {'type': 'object', 'properties': {'airportcode': {'type': 'string', 'description': '', 'example_value': 'LHR'}, 'carriercode': {'type': 'string', 'description': ''}, 'date': {'type': 'string', 'description': ''}}, 'required': ['airportcode'], 'optional': ['carriercode', 'date']}}, {'name': 'flight_search_v2_for_flight_fare_search', 'description': 'This is the subfunction for tool "flight_fare_search", you can use this tool.The description of this function is: "A faster, more agile Endpoint that\'s used to search flights."', 'parameters': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': ''}, 'is_from': {'type': 'string', 'description': '', 'example_value': 'LHR'}, 'adult': {'type': 'integer', 'description': '', 'example_value': '1'}, 'to': {'type': 'string', 'description': '', 'example_value': 'DXB'}, 'currency': {'type': 'string', 'description': '', 'example_value': 'USD'}, 'type': {'type': 'string', 'description': '', 'example_value': 'economy'}, 'child': {'type': 'string', 'description': ''}, 'infant': {'type': 'string', 'description': ''}}, 'required': ['date', 'is_from', 'adult', 'to'], 'optional': ['currency', 'type', 'child', 'infant']}}, {'name': 'airport_departues_for_flight_fare_search', 'description': 'This is the subfunction for tool "flight_fare_search", you can use this tool.The description of this function is: "An endpoint to get Departues in an airport"', 'parameters': {'type': 'object', 'properties': {'airportcode': {'type': 'string', 'description': '', 'example_value': 'LHR'}, 'carriercode': {'type': 'string', 'description': ''}, 'date': {'type': 'string', 'description': ''}}, 'required': ['airportcode'], 'optional': ['carriercode', 'date']}}, {'name': 'airport_search_for_flight_fare_search', 'description': 'This is the subfunction for tool "flight_fare_search", you can use this tool.The description of this function is: "An endpoint to search airports"', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': '', 'example_value': 'LHR'}}, 'required': ['query'], 'optional': []}}, {'name': 'Finish', 'description': 'If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.', 'parameters': {'type': 'object', 'properties': {'return_type': {'type': 'string', 'enum': ['give_answer', 'give_up_and_restart']}, 'final_answer': {'type': 'string', 'description': 'The final answer you want to give the user. You should have this field if "return_type"=="give_answer"'}}, 'required': ['return_type']}}]
        }
