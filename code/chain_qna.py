from langchain_community.llms import HuggingFaceHub, HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import os
import torch

HF_API_KEY = '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'
os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'HF_API_KEY'

# hub_llm = HuggingFaceEndpoint(
#     repo_id = 'google/flan-t5-large',
#     max_new_tokens = 64,
#     temperature=0.01,
#     huggingfacehub_api_token = HF_API_KEY
# )

model_id = 'BioMistral/BioMistral-7B'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id) #AutoModelForSeq2SeqLM if model is encoder-decoder

pipe = pipeline(
    "text-generation", #text2text-generation if model is encoder-decoder
    model = model,
    tokenizer = tokenizer,
    max_length = 100
)

local_llm = HuggingFacePipeline(pipeline=pipe)

multi_template = """Answer the following questions one at a time.

Questions: {questions}
Answers:
"""

prompt = PromptTemplate(
    template=multi_template,
    input_variables=['questions']
)

llm_chain = LLMChain(
    llm = local_llm,
    prompt = prompt
)

# user question:
qs_str = [
    {'questions': "Which NFL team won the Super Bowl in the 2010 season?"},
    {'questions': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
    {'questions': "Who was the 12th person on the moon?"},
    {'questions': "How many eyes does a blade of grass have?"}
]

# answer:
res = llm_chain.generate(qs_str)
print(res)
