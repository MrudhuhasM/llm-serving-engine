from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from engine.engine import Engine
from engine.request import Request


app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100


engine = Engine("Qwen/Qwen3-0.6B")


def stream_text(request: Request):
    for token_id in engine.generate(request):
        text = engine.tokenizer.decode([token_id])
        yield text


@app.post("/generate")
async def generate(request: GenerateRequest):
    req = Request(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
    )

    return StreamingResponse(
        stream_text(req),
        media_type="text/plain",
    )