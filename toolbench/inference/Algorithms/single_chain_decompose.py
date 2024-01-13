import re
from Tree.Tree import my_tree, tree_node
from Prompts.ReAct_prompts import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION, FORMAT_INSTRUCTIONS_USER_FUNCTION
from Algorithms.base_search import base_search_method
from copy import deepcopy
import backoff, json

DECOMPOSE_SYSTEM_PROMPT = r"Please break down the following instructions into simpler, standalone instructions while preserving the original meaning. Please also make the subtasks divided executable on its own, i.e. keep all necessary information in all subtasks. The granularity doesn't need to be too fine. Please also include the context information that is needed to complete each subtask but is shared across all subtasks.  Return the outcome in a JSON file in the following format: {\"context\": ..., \"subtasks\": [subtask1, subtask2, ...]}.Ensure all subtasks to be pure text."
DECOMPOSE_FEWSHOT = [
    {"role": "user", "content": "Query: \nI'm organizing a company event and I need to track the delivery of event materials. Can you help me track the package with the tracking number PQR678? Also, provide me with the latest status and location updates. Additionally, find the address details for the event venue using the zip code 43210."},
{"role": "assistant", "content": json.dumps({
  "context": "Organizing a company event requires tracking the delivery of event materials and obtaining venue address details.",
  "subtasks": [
    "Use the tracking number PQR678 to check the delivery status of the event materials package.",
    "Obtain the latest location updates for the package with tracking number PQR678.",
    "Find the address details for the event venue located in the area with the zip code 43210."
  ]
})}
]
DECOMPOSE_USER_PROMPT = "Query: {query}"
TASK_FORMAT_PROMPT = """\
Given the task context 
{context} 
Now please complete the following task: 
{subquery}"""
FINAL_ANSWER_PROMPT = """\
Please provide the final answer to the task given context, subqueries and answer trajectory to each subquery.
Context: {context}
Subqueries: {subquery}
Answer trajectory: {answer}
Please summarize the results and give your final answer.
"""


class single_chain_decompose(base_search_method):
    """Implement of CoT method
    """
    def __init__(self,llm,io_func,extra_prefix="",process_id=0,start_message_list=None, buffer=None, history_buffer="None", local_buffer="None", retriever=None):
        """extra_prefix and start_message_list is used in Reflection Algo"""
        super(single_chain_decompose, self).__init__(llm,io_func, process_id, callbacks=None)
        self.io_func = io_func
        self.llm = llm
        self.extra_prefix = extra_prefix
        self.start_message_list = start_message_list
        self.process_id = process_id
        self.retriever = retriever
        if buffer != None:
            self.buffer = buffer
            self.history_buffer = history_buffer
            self.local_buffer = local_buffer
        else:
            self.buffer = None

        self.restart()
    def restart(self):
        self.status = 0
        self.try_list = []
        self.terminal_node = []

        self.query_count = 0 # number of interactions with openai
        self.total_tokens = 0
        self.success_count = 0

    def to_json(self, answer=False,process=True):
        if process:
            json_obj = {
                "win": self.status == 1,
                "try_count": len(self.try_list),
                "trys": self.try_list,
                "compare_candidates": [],
                "forward_args":self.forward_args,
            }
            for node in self.terminal_node:
                if node.pruned == False: # has final answer
                    json_obj["compare_candidates"].append(node.get_chain_result_from_this_node(use_messages=False))
        else:
            json_obj = {}

        if answer:
            json_obj["answer_generation"] = {
                "valid_data": False,
                "final_answer": "",
                "function": self.io_func.functions,
                "query_count": self.query_count,
                "total_tokens": self.total_tokens,
                "train_messages": [],
                "chain": [],
            }
            for node in self.terminal_node:
                if node.pruned == False:
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["final_answer"] = node.description
                    json_obj["answer_generation"]["train_messages"] = node.get_train_messages_from_this_node()
                    break
        return json_obj

    def to_json_single(self):
        """parse the last try
        Though the nodes are formed as a tree, We still know they are actually a chain
        """
        json_obj = {}
        tree_obj = self.terminal_node[-1].get_chain_result_from_this_node()
        json_obj["chain"] = tree_obj
        json_obj["win"] = self.status == 1
        return json_obj

    def start(self,single_chain_max_step,pass_at=1,answer=1):
        self.forward_args = locals()
        if "self" in self.forward_args.keys():
            self.forward_args.pop("self")


        context, subquerys = self.decompose_task()
        self.tree = my_tree()
        self.tree.root.node_type = "Decompose"
        self.tree.root.description = f"""
Decompose tasks:
Context is: {context}
Subtasks are: {subquerys}"""
        self.tree.root.print(self.process_id)

        for _ in range(pass_at):
            results = []
            outnodes = []
            previous_node = self.tree.root
            for subquery in subquerys:
                temp_node = tree_node()
                temp_node.node_type = "Decompose"
                temp_node.io_state = deepcopy(self.io_func)
                temp_node.io_state.input_description = TASK_FORMAT_PROMPT.format(context=context, subquery=subquery)
                temp_node.messages = previous_node.messages.copy()
                temp_node.messages.append({
                    "role":"assistant",
                    "name": "Decompose",
                    "content": temp_node.io_state.input_description
                })
                if self.io_func.retriever is not None:
                    self.io_func.query_json['query'] = temp_node.io_state.input_description
                    self.io_func.refresh_tool_funcs()


                temp_node.father = previous_node
                previous_node.children.append(temp_node)
                out_node = self.do_chain(temp_node, single_chain_max_step)
                outnodes.append(out_node)
                sucess_state = out_node.io_state.check_success()
                results.append(sucess_state)
                previous_node = out_node

        
            terminal_node = tree_node()
            terminal_node.father = previous_node
            previous_node.children.append(terminal_node)
            terminal_node.node_type = "Action"
            terminal_node.description = 'Finish'
            

            final_node = tree_node()
            final_node.node_type = "Action Input"
            final_node.father = terminal_node
            terminal_node.children.append(final_node)
            
            final_message = FINAL_ANSWER_PROMPT.format(context=context, subquery=subquerys, answer=[node.description for node in outnodes])
            final_message = {"role":"user","content":final_message}
            self.llm.change_messages([final_message])
            outputs, _, _ = self.llm.parse(functions=[],process_id=0)
            terminal_node.messages = previous_node.messages.copy()
            terminal_node.messages.extend([final_message, outputs])
            # terminal_node.detailed_description = outputs["content"]
            terminal_node.print(self.process_id)
            terminal_node.pruned = False
            for node in outnodes:
                if node.pruned:
                    terminal_node.pruned = True
                    break

            final_node.pruned = terminal_node.pruned
            final_node.messages = terminal_node.messages.copy()


            # self.terminal_node.append(terminal_node)
            # self.try_list.append(self.to_json_single())


            if all(results):
                self.status = 1
                self.success_count += 1
                final_node.observation = "{\"response\":\"successfully giving the final answer.\"}"
                final_node.description = json.dumps({
                    'return_type' : 'give_answer',
                    'final_answer': terminal_node.description,
                })
                self.terminal_node.append(final_node)
                self.try_list.append(self.to_json_single())
                if self.success_count >= answer:
                    return 1
            else:
                final_node.observation = "{\"response\":\"chose to give up and restart\"}"
                final_node.description = "{\n  \"return_type\": \"give_up_and_restart\"\n}"
                self.terminal_node.append(final_node)
                self.try_list.append(self.to_json_single())
        return sum(results) / len(results)

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def decompose_task(self):
        start_task = self.io_func.input_description
        messages = [
                    {"role":"system","content": DECOMPOSE_SYSTEM_PROMPT},
                    *DECOMPOSE_FEWSHOT,
                    {"role":"user","content": DECOMPOSE_USER_PROMPT.format(query=start_task)},
                ]
        self.llm.change_messages(messages)
        responses,_,_ = self.llm.parse(functions=[],process_id=0, response_format={ "type": "json_object" })
        task_decompose = json.loads(responses["content"])
        context = task_decompose["context"]
        subquerys = task_decompose["subtasks"]
        return context, subquerys



    def do_chain(self,now_node,single_chain_max_step):

        if self.start_message_list == None:
            system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION
            system = system.replace("{task_description}",self.io_func.task_description)
            now_node.messages.append({"role":"system","content":system})

            user = FORMAT_INSTRUCTIONS_USER_FUNCTION
            user = user.replace("{input_description}",self.io_func.input_description)
            now_node.messages.append({"role":"user","content":user})
            
            if self.buffer != None and hasattr(self, "history_buffer"):
                history_prompt = self.buffer.get_history_prompt_using_instruction(instruction=user, k=2, key="query")
                now_node.messages.append({"role":"system","content":history_prompt})
            
                # print("wfff initial messages: \n%s"% str(self.tree.root.messages))

        else:
            """In Reflection Algo, we startswith former trials and reflections, so the caller will give the start messages"""
            now_node.messages = self.start_message_list
        
        # now_node = self.tree.root
        while True:
            # recursively parse message into nodes
            self.llm.change_messages(now_node.messages)
            new_message,error_code,total_tokens = self.llm.parse(functions=self.io_func.functions,process_id=self.process_id)
            self.total_tokens += total_tokens                               
            self.query_count += 1
            assert new_message["role"] == "assistant"
            if "content" in new_message.keys() and new_message["content"] != None:
                temp_node = tree_node()
                temp_node.node_type = "Thought"
                temp_node.description = new_message["content"]
                child_io_state = deepcopy(now_node.io_state)
                
                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0 
                temp_node.messages = now_node.messages.copy()
                temp_node.father = now_node
                now_node.children.append(temp_node)
                temp_node.print(self.process_id)
                now_node = temp_node

                if error_code != 0:
                    now_node.observation_code = error_code
                    now_node.pruned = True

            if "function_call" in new_message.keys() and new_message['function_call'] is not None:
                function_name = new_message["function_call"]["name"]
                temp_node = tree_node()
                temp_node.node_type = "Action"
                temp_node.description = function_name
                child_io_state = deepcopy(now_node.io_state)
                
                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0 
                temp_node.messages = now_node.messages.copy()
                temp_node.father = now_node
                now_node.children.append(temp_node)

                temp_node.print(self.process_id)
                now_node = temp_node

                function_input = new_message["function_call"]["arguments"]
                temp_node = tree_node()
                temp_node.node_type = "Action Input"
                temp_node.description = function_input
                child_io_state = deepcopy(now_node.io_state)

                observation, status = child_io_state.step(action_name=now_node.description, action_input=function_input)
                temp_node.observation = observation
                temp_node.observation_code = status

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0 
                temp_node.messages = now_node.messages.copy()
                temp_node.father = now_node
                now_node.children.append(temp_node)
                temp_node.print(self.process_id)
                now_node = temp_node

                if status != 0:
                    # return code refers to Downstream_tasks/rapidapi
                    if status == 4:
                        now_node.pruned = True
                    elif status == 1: # hallucination api name
                        assert "function_call" in new_message.keys()
                        new_message["function_call"]["name"] = "invalid_hallucination_function_name"
            
            now_node.messages.append(new_message)
            if now_node.node_type == "Thought":
                if hasattr(self, "local_buffer") and self.local_buffer == "thought":
                    local_system_message = self.buffer.get_history_prompt_using_instruction(instruction=new_message["content"], key="thought", k=1)
                    now_node.messages.append({"role":"system","content":local_system_message})
                    print("wff got local history: " + local_system_message)
            if now_node.node_type == "Action Input":
                now_node.messages.append({
                    "role":"function",
                    "name": new_message["function_call"]["name"],
                    "content": now_node.observation,
                })
            if now_node.get_depth() >= single_chain_max_step and not (now_node.is_terminal):
                now_node.pruned = True
            
            if now_node.pruned or now_node.is_terminal:
                return now_node

    

