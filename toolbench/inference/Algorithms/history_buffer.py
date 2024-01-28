from logging.handlers import BufferingHandler
from matplotlib.backend_bases import key_press_handler
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
from datasets import Dataset
import requests
import json
from requests.auth import HTTPBasicAuth
import numpy as np
import atexit
import time
import openai
import openai

# client = OpenAI(api_key="openchat", base_url="https://api.01ww.xyz/v1")
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential

class Buffer:
    """
    A class representing a buffer that stores and manages text samples with their embeddings.
    """
    def __init__(self, encoder='ada', path=None, online_update=False):
        """
        Initializes a new Buffer instance.

        Args:
            encoder (str): The encoder type to use, e.g., 'dpr'.
        """
        if online_update:
            atexit.register(self.__exit)
        self.encoder = encoder
        if encoder == 'dpr':
            torch.set_grad_enabled(False)
            self.ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        
        self.dataset_path = '/ML-A100/Home/csj/gzc/STC/sidecar/data/buffer.jsonl'
        self.local_dataset_path = '/ML-A100/Home/csj/gzc/STC/sidecar/data/local_buffer.jsonl'
        self.ds = Dataset.from_json(self.dataset_path)
        
        self.local_ds = Dataset.from_json(self.local_dataset_path)

        # self.history_list = []
        # with open(self.dataset_path, "r") as f:
        #     for line in tqdm(f.readlines(), position=0, desc="loading buffer"):
        #             obj = json.loads(line)
        #             obj["query_embedding"] = np.array(obj["query_embedding"])
        #             obj["data"] = json.dumps(obj["data"])
        #             self.history_list.append(obj)
        # self.ds = Dataset.from_list(self.history_list)

        self.RAW_HISTORY_PROMPT = " You have had following conversations in similar tasks before. Please get inspiration from the successful conversations, but avoid making the same mistakes in the failed ones. "
        self.TOOL_DESC_HISTORY_PROMPT = " You have called the following functions respectively in similar tasks before,  please utilize this information "
        
        self.LOCAL_HISTORY_PROMPT = " You have tried the following API calls in similar tasks before, please get inspiration from the successful conversations, but avoid making the same mistakes in the failed ones. "

    def save(self):
        print('saving dataset to ' + self.dataset_path)
        self.ds.to_json(self.dataset_path)

    def __exit(self):
        self.save()

    def _ada_embedding_request(self, input, model="text-embedding-ada-002"):
        # TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(api_base="https://api.01ww.xyz/v1")'
        # openai.api_base="https://api.01ww.xyz/v1"
        input = input.replace("\n", " ")
        # response = openai.embeddings.create(input = [input], model=model)
        response = openai.Embedding.create(input = [input], model=model)
        return response.data[0].embedding


    def embed(self, input):
        """
        Generates an embedding for the given input text.

        Args:
            input (str): The input text to be embedded.

        Returns:
            numpy.ndarray: The embedding vector of the input text.
        """
        if self.encoder == 'dpr':
            return (self.ctx_encoder(**self.ctx_tokenizer(input, return_tensors="pt", truncation=True))[0][0].numpy(), True)
        elif self.encoder == 'ada':
            resp =  self._ada_embedding_request(input)
            return (np.array(resp), True)

    def add_sample(self, input, metadata={}):
        if "answer_generation" not in input or "query" not in input["answer_generation"]:
            return
        if "train_messages" not in input["answer_generation"] or len(input["answer_generation"]["train_messages"]) == 0:
            return
        
         
        query = input["answer_generation"]["query"]
        success_chain = input["answer_generation"]["train_messages"][-1]
        data_record = {
            "query": query,
            "data": input,
        }
        # query_embedding, ok = self.embed(input["answer_generation"]["query"])
        # if ok:
        #     data_record['query_embedding'] = query_embedding
        print(type(data_record))
        print(type(metadata))
        self.ds = self.ds.add_item({**data_record, **metadata})
        
    
    def get_nearest_samples(self, query, k=3, buffer_info='raw'):
        string_list = []
        if buffer_info == "mock":
            samples = self.ds[0:1+k]
            string_list = samples["data"]
        else:
            query_embedding, ok = self.embed(query)
            if not self.ds.is_index_initialized('query_embedding'):
                self.ds.add_faiss_index(column='query_embedding')
            scores, samples = self.ds.get_nearest_examples('query_embedding', query_embedding, k)
            string_list = samples['data']
        

        raw_data_list = [json.loads(string) for string in string_list]
        success_chains = [raw_data["answer_generation"]["train_messages"][-1] for raw_data in raw_data_list]
        for idx in range(len(success_chains)):
            if len(success_chains[idx]) > 0 and success_chains[idx][0]["role"] == "system":
                success_chains[idx].pop(0)
        if buffer_info == "raw":
            return [" Task " + str(idx+1) + ": " + json.dumps(chain) for idx,chain in enumerate(success_chains)]
        elif buffer_info == "mock":
            return [" Task " + str(idx+1) + ": " + json.dumps(chain) for idx,chain in enumerate(success_chains)]
        elif buffer_info == "tool_desc":
            res = []
            for raw_data in raw_data_list:
                curr_actions = []
                functions = {}
                function_list = raw_data["answer_generation"]["function"]
                for function in function_list:
                    functions[function["name"]] = function
                    
                chain = raw_data["answer_generation"]["train_messages"][-1]
                for message in chain:
                    if message["role"] == "assistant" and "function_call" in message:
                        function_name = message["function_call"]["name"]
                        function_desc = functions[function_name].get("description") if function_name in functions else ""
                        function_call = {
                            "name" : message["function_call"]["name"],
                            "description" : function_desc,
                        }
                        curr_actions.append(function_call)
                res.append(curr_actions)
            return [" Task " + str(idx+1) + ": " + json.dumps(temp) for idx, temp in enumerate(res)]
        
    def get_local_nearest_samples(self, query, k=5, buffer_info='raw', key='thought'):
        """
        Retrieves the nearest samples to a given query text based on their embeddings.

        Args:
            query (str): The query text to find nearest samples for.
            k (int): The number of nearest samples to retrieve.

        Returns:
            list: A list of nearest sample contents.
        """
        res = []
        if buffer_info == "mock":
            samples = self.local_ds[0:1+k]
            res = samples["value"]
        else:
            if key == "thought":
                query_embedding, ok = self.embed(query)
                if not self.local_ds.is_index_initialized('thought_embedding'):
                    self.local_ds.add_faiss_index(column='thought_embedding')
                scores, samples = self.local_ds.get_nearest_examples('thought_embedding', query_embedding, k)
                res = samples['value']
            elif key == "chain":
                query_embedding, ok = self.embed(query)
                if not self.local_ds.is_index_initialized('chain_embedding'):
                    self.local_ds.add_faiss_index(column='chain_embedding')
                scores, samples = self.local_ds.get_nearest_examples('chain_embedding', query_embedding, k)
                res = samples['value']
                
        return [" Function Call " + str(idx+1) + ": " + json.dumps(temp) for idx, temp in enumerate(res)]
            


    def add_faiss_index(self):
        self.ds.add_faiss_index(column='query_embedding')


    def __len__(self):
        """
        Returns the number of samples in the buffer.

        Returns:
            int: The number of samples in the buffer.
        """
        return self.ds.num_rows
    
    def get_samples(self):
        """
        Returns the entire buffer's contents.

        Returns:
            Dataset: The dataset containing the buffer's samples and embeddings.
        """
        return self.ds


    def get_history_prompt_using_instruction(self, instruction, k=3, key='query', buffer_info='raw'):
        if key == "query":
            fetched_history = self.get_nearest_samples(query=instruction, k=k, buffer_info=buffer_info)
            if buffer_info == "raw":
                print('wff search raw history using instruction:%s' % instruction)
                return self.RAW_HISTORY_PROMPT + '\n' + '\n'.join(fetched_history)
            elif buffer_info == "tool_desc":
                print('wff search tool_desc history using instruction:%s' % instruction)
                return self.TOOL_DESC_HISTORY_PROMPT + '\n' + '\n'.join(fetched_history)
            elif buffer_info == "mock":
                print('wff search mock history using instruction:%s' % instruction)
                return self.RAW_HISTORY_PROMPT + '\n' + '\n'.join(fetched_history)
        elif key in ["chain", "thought"]:
            local_history = self.get_local_nearest_samples(instruction, key=key)
            if key == "thought":
                print('wff search local history using thought:%s' % instruction)
            elif key == "chain":
                local_history = self.get_local_nearest_samples(query=instruction, key="chain", k=k)
                print('wff search local history using chain:%s' % instruction)
            return self.LOCAL_HISTORY_PROMPT + '\n' + '\n'.join(local_history)



# Example usage
if __name__ == "__main__":
    buffer = Buffer()
    print(buffer.ds)

    print(buffer.get_history_prompt_using_instruction(instruction='I need to search for history information', k=2, key="chain"))
    # buffer.save()