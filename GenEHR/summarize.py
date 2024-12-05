""" import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token="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",
)

def summarize(text):
    messages = [

    {
    "role": "system",
    "content": "Summarize the following conversation in **English**, extracting **only the symptoms, diagnosis, and treatment** explicitly mentioned. Provide the result in **two brief lines** suitable for a medical report, **without any extra information** or context."
    },
    {
    "role": "user",
    "content":f"{text}"
    }



    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][-1]
 """

import ollama

def summarize(text):
    q = f"""Summarize the following conversation in **English**, extracting **only the symptoms, diagnosis, and treatment** explicitly mentioned. Provide the result in **two brief lines** suitable for a medical report, **without any extra information** or context. \n {text}"""
    response = ollama.generate(model='llama3', prompt=q)
    return response
