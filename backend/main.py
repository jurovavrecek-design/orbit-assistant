from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

from app.models.chat_request import ChatRequest
from app.services.chat_service import ChatService


app = FastAPI()

templates = Jinja2Templates(
    directory="templates"
)

chat_service = ChatService()


@app.get("/")
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="chat.html"
    )


#
# Existing non-streaming endpoint
#
@app.post("/chat")
def chat(request: ChatRequest):

    return chat_service.ask(
        request.question
    )


#
# New streaming endpoint
#
@app.post("/chat/stream")
def chat_stream(request: ChatRequest):

    return StreamingResponse(

        chat_service.stream(
            request.question
        ),

        media_type="text/plain"
    )