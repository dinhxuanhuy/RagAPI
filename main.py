from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from Rag import RagPipeLine
import uvicorn
import os

app = FastAPI()

# Initialize RAG Pipeline
# We initialize it here so it's loaded when the app starts
try:
    rag_pipeline = RagPipeLine()
except Exception as e:
    print(f"Failed to initialize RAG Pipeline: {e}")
    rag_pipeline = None

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Welcome to StoreRagApi"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline is not initialized")
    
    try:
        # The rag_ask method expects a string question
        # and returns a dictionary with "query" and "result" (or similar depending on chain)
        # However, checking Rag.py, it calls self.rag_pipeline({"question": question})
        # RetrievalQAWithSourcesChain usually returns 'answer' and 'sources' keys.
        # But we disabled sources.
        
        response = rag_pipeline.rag_ask(request.question)
        
        # Depending on what RetrievalQAWithSourcesChain returns exactly:
        # It typically returns a dict. We should extract the answer.
        # The prompt uses "Assistant:" at the end, implying a chat completion.
        
        if isinstance(response, dict) and "answer" in response:
             return {"answer": response["answer"]}
        elif isinstance(response, dict) and "result" in response:
             return {"answer": response["result"]}
        else:
             return {"answer": str(response)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
