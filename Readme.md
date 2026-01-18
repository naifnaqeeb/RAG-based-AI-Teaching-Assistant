# RAG-Based AI Teaching Assistant

This project allows you to build a Retrieval-Augmented Generation (RAG) system
using your own video content.

---

## Step 1 – Collect Your Videos
Move all your video files into the `videos/` folder.

---

## Step 2 – Convert Videos to MP3
Run the `video_to_mp3.py` script to extract audio from all videos.

---

## Step 3 – Convert MP3 to JSON
Run the `mp3_to_json.py` script to transcribe audio into text and chunk it into JSON files.

---

## Step 4 – Convert JSON to Embeddings
Run `preprocess_json.py` to:
- Convert text chunks into vector embeddings
- Store them in a Pandas DataFrame
- Save the result as a `joblib` file

---

## Step 5 – Query the System (RAG)
Load the joblib file into memory and:
1. Embed the user’s question  
2. Retrieve the most relevant chunks  
3. Build a context-based prompt  
4. Send it to the LLM for an answer  

---

## Output  
The assistant answers using **only your video content**.
