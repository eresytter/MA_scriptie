from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import getpass
from pprint import pprint
import json
import csv

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Your API key here")
llm = ChatOpenAI(
    model = "gpt-4o", 
    temperature = 0.1,
    max_tokens = 256
)

"""
Read each lines in the the jsonlines file and assign it to file_input.
"""
prompts = []
file_input = './data/maieutic_part2.json'
file_output = './data/gpt-4o_maieutic_part_2.csv'
with open(file_input) as file:
    for line in file:
        prompts.append(json.loads(line))

"""
Chain 1
"""
prompt_one = ChatPromptTemplate.from_template(
    "Decide if {utterance} contains a metaphorical comparison or not. Limit response to yes or no."
)
runnable_one = prompt_one | llm | StrOutputParser()
chain_one = RunnablePassthrough.assign(decision = runnable_one)

"""
Chain 2
"""

prompt_two = ChatPromptTemplate.from_template(
    "You have said {decision} if \"{utterance}\" contains a metaphorical comparison or not. Provide a reason and rate how confident you are from 0 to 10."
)
runnable_two = prompt_two | llm | StrOutputParser()
chain_two = chain_one | RunnablePassthrough.assign(argument = runnable_two)

"""
Chain 3
"""

prompt_three = ChatPromptTemplate.from_template(
    "Provide a counter-argument against \"{argument}\". Rate how confident you are from 0 to 10."
)

runnable_three = prompt_three | llm | StrOutputParser()

chain_three = chain_two | RunnablePassthrough.assign(counter_argument = runnable_three)

"""
expected_input = ([
        {"utterance": "It is as if someone hit my heel with a hammer."},
        # {"utterance": "The pain feels like a gnawing ache and lasts for about a week, then subsides, eventually goes away completely, and then a month or two later comes back."},
        # {"utterance": "The pain is constant, dull and gnawing."},
        {"utterance": "I have a painful armpit lump."},
    ])
"""

"""
Structured Output
"""

for i, prompt in enumerate(prompts):
    print(f"Answering question {i+1}: {prompt['utterance']}")
    
response = chain_three.batch(prompts)

fieldnames = ['utterance', 'decision', 'argument', 'counter_argument']
with open(file_output, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for row in response:
        writer.writerow(row)

print("Output saved to", file_output)