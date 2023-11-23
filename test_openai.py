import os
import openai
openai.api_base="https://api.01ww.xyz/v1"
openai.api_key = 'openchat'


# data = {'model': 'gpt-3.5-turbo-16k-0613', 'messages': [{'role': 'user', 'content': "下面这句英文可能有语病，能不能把语病都改掉？\nIf you think you get the result which can answer the task, call this function to give the final answer. Or, if you think you can't handle the task from this status, call this function to restart. Remember: you should ALWAYS call this function at the end of your try, and the final answer is the ONLY part that will be showed to user, so final answer should contain enough information.\n没语病的形式：\n"}], 'max_tokens': 1024, }

# completion = openai.ChatCompletion.create(
# #   model="gpt-3.5-turbo",
# #   messages=[
# #     {"role": "system", "content": "You are a helpful assistant."},
# #     {"role": "user", "content": "Hello!"}
# #   ]
#     **data
# )

# print(completion.choices[0].message)

# completion = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo-16k-0613",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello!"}
#   ]
# )

# print(completion.choices[0].message)


embedding = openai.Embedding.create(
  model="text-embedding-ada-002",
  input="The food was delicious and the waiter...",
  encoding_format="float"
)
print(embedding)