from langchain.llms import HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import getpass
from pprint import pprint
import json
import csv

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
endpoint_url = "endpoint_url"
llm = HuggingFaceEndpoint(
    endpoint_url=endpoint_url,
    huggingface_api_token=HUGGINGFACEHUB_API_TOKEN
)
"""
Read each lines in the the jsonlines file and assign it to file_input.
"""
prompts = []
file_input = './data/maieutic.json'
file_output = './data/medicine-chat_output.csv'
with open(file_input) as file:
    for line in file:
        prompts.append(json.loads(line))

"""
Chain 1
"""
prompt_one = ChatPromptTemplate.from_template(
    "Decide if \"{utterance}\" contains a metaphorical comparison or not. Answer only yes or no."
)
runnable_one = prompt_one | llm | StrOutputParser()
chain_one = RunnablePassthrough.assign(decision = runnable_one)

"""
Chain 2
"""

prompt_two = ChatPromptTemplate.from_template(
    "You have said \"{decision}\" if \"{utterance}\" contains a metaphorical comparison or not. Provide a reason."
)
runnable_two = prompt_two | llm | StrOutputParser()
chain_two = chain_one | RunnablePassthrough.assign(argument = runnable_two)

"""
Chain 3
"""

prompt_three = ChatPromptTemplate.from_template(
    "Provide a counter-argument against \"{argument}\"."
)

runnable_three = prompt_three | llm | StrOutputParser()

chain_three = chain_two | RunnablePassthrough.assign(counter_argument = runnable_three)


# expected_input = ([
#         {"utterance": "It is as if someone hit my heel with a hammer."},
#         # {"utterance": "The pain feels like a gnawing ache and lasts for about a week, then subsides, eventually goes away completely, and then a month or two later comes back."},
#         # {"utterance": "The pain is constant, dull and gnawing."},
#         {"utterance": "I have a painful armpit lump."},
#     ])


"""
Structured Output
"""

# for i, prompt in enumerate(prompts):
#     print(f"Answering question {i+1}: {prompt['utterance']}")
    
response = chain_three.batch(prompts)
print(response)

fieldnames = ['utterance', 'decision', 'argument', 'counter_argument']
with open(file_output, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for row in response:
        writer.writerow(row)

print("Output saved to:", file_output)