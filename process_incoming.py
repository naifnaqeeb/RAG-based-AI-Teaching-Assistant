import os
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed",json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()['embeddings']
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate",json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    response = r.json()
    print(response)
    return response

def inference_gemini(prompt):
    print("Thinking....")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    r = requests.post(url, json=payload)
    r.raise_for_status()

    return r.json()["candidates"][0]["content"]["parts"][0]["text"]



df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a question: ")
question_embedding = create_embedding([incoming_query])[0]

#Find similarities of question_embeddings with other embeddings
similarities = cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx]
# print(new_df[["title","number","text"]])

prompt = f'''I am teaching web development in my Sigma Web Development course. Here are the video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title","number","start","end","text"]].to_json(orient="records")}
-----------------------------------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (don't mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, then tell him you can only ask questions related to the course
'''

with open("prompt.txt",'w') as f:
    f.write(prompt)

# response = inference(prompt)["response"]
# print(response)

response = inference_gemini(prompt)
print(response)

with open("response.txt",'w') as f:
    f.write(response)

# for index, item in new_df.iterrows():
#     print(index,item["title"],item["number"],item["text"],item["start"],item["end"])