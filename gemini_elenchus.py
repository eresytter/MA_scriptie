# pip install --upgrade --quiet  langchain-google-genai pillow
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import getpass
from pprint import pprint
import json
import csv

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Your API key here")
llm = ChatGoogleGenerativeAI(
    model = "gemini-1.0-pro", 
    temperature = 0.1,
    max_tokens = 256
)

"""
Read each lines in the the jsonlines file and assign it to file_input.
"""
prompts = []
file_input = './data/elenchus.json'
file_output = './data/gemini-1.0-pro_elenchus.csv'
with open(file_input) as file:
    for line in file:
        prompts.append(json.loads(line))

# """
# Example prompt
# """
# template = """
# Your task is to answer the question below.

# Question: {example}
# Answer:
# """

# prompt = ChatPromptTemplate.from_template(template)
# runnable = prompt | llm | StrOutputParser()

# example = ([
#         {"example": "It is as if someone hit my heel with a hammer. How big was the hammer?"},
#         {"example": "It feels as if someone has tied a rubber band at my elbow and I am just waiting for my arm to explode. Is the explosion caused by a dynamite?"},
#         {"example": "Sometimes it feels as if someone hit my head with a baseball bat. How big was the baseball bat?"},
#         # {"example": "I have a painful armpit lump."},
#     ])


"""
Prompt
"""
template = """
Your task is to answer the question below.

Question: {lure}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
runnable = prompt | llm | StrOutputParser()

for i, prompt in enumerate(prompts):
    print(f"Answering question {i+1}: {prompt['lure']}")

"""
Structured Output
"""
    
response = runnable.batch(prompts)

with open(file_output, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Response'])
    writer.writerows([[item] for item in response])
print("Output saved to", file_output)