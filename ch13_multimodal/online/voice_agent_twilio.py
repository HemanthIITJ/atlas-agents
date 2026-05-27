"""
voice_agent_twilio.py — Twilio Media Stream → OpenAI Realtime API bridge.

Twilio sends telephony audio (PCMU/8kHz) over a WebSocket Media Stream.
This FastAPI server accepts that stream, converts audio to PCM16/24kHz,
forwards it to the OpenAI Realtime API, and streams audio responses back
to the caller in real time.

Run:
    uvicorn voice_agent_twilio:app --host 0.0.0.0 --port 8080

Configure Twilio:
    Set your phone number's Voice URL to:
    https://<your-host>/twiml

    The TwiML response starts the Media Stream pointed at /media-stream.
"""

import asyncio
import base64
import json
import os

try:
    import audioop                    # Python ≤ 3.12 stdlib
except ImportError:
    import audioop_lts as audioop     # pip install audioop-lts  (Python 3.13+)

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

SYSTEM_PROMPT = """You are Atlas, a helpful voice assistant.
Answer concisely — spoken answers should be under 3 sentences.
Do not say "certainly", "absolutely", or "of course".
"""


@app.get("/twiml")
async def twiml_response():
    """Tell Twilio to start a Media Stream pointing at /media-stream."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://YOUR_HOST/media-stream"/>
  </Connect>
</Response>"""
    return HTMLResponse(content=xml, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(twilio_ws: WebSocket):
    """
    Bridge between Twilio Media Stream and OpenAI Realtime API.

    Twilio sends PCMU (G.711 μ-law) at 8kHz.
    OpenAI Realtime expects PCM16 at 24kHz.
    We transcode on the fly.
    """
    await twilio_ws.accept()

    openai_headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(OPENAI_REALTIME_URL, extra_headers=openai_headers) as oai_ws:
        await _configure_session(oai_ws)
        stream_sid = None

        async def receive_from_twilio():
            nonlocal stream_sid
            try:
                async for raw in twilio_ws.iter_text():
                    msg = json.loads(raw)
                    event = msg.get("event")

                    if event == "start":
                        stream_sid = msg["start"]["streamSid"]

                    elif event == "media":
                        # Twilio sends PCMU/8kHz — convert to PCM16/24kHz
                        mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                        pcm8k = audioop.ulaw2lin(mulaw_bytes, 2)       # μ-law → PCM16
                        pcm24k = audioop.ratecv(pcm8k, 2, 1, 8000, 24000, None)[0]
                        await oai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(pcm24k).decode(),
                        }))

                    elif event == "stop":
                        break

            except WebSocketDisconnect:
                pass

        async def receive_from_openai():
            try:
                async for raw in oai_ws:
                    event = json.loads(raw)
                    etype = event.get("type")

                    if etype == "response.audio.delta" and stream_sid:
                        # OpenAI sends PCM16/24kHz — convert back to PCMU/8kHz for Twilio
                        pcm24k = base64.b64decode(event["delta"])
                        pcm8k = audioop.ratecv(pcm24k, 2, 1, 24000, 8000, None)[0]
                        mulaw = audioop.lin2ulaw(pcm8k, 2)
                        await twilio_ws.send_text(json.dumps({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": base64.b64encode(mulaw).decode()},
                        }))

                    elif etype == "input_audio_buffer.speech_started":
                        # User interrupted — cancel current response
                        await oai_ws.send(json.dumps({"type": "response.cancel"}))
                        if stream_sid:
                            # Tell Twilio to stop playing buffered audio
                            await twilio_ws.send_text(json.dumps({
                                "event": "clear",
                                "streamSid": stream_sid,
                            }))

                    elif etype == "response.function_call_arguments.done":
                        # Tool call — execute and return result
                        result = await _execute_tool(event["name"], event["arguments"])
                        await oai_ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": event["call_id"],
                                "output": json.dumps(result),
                            }
                        }))
                        await oai_ws.send(json.dumps({"type": "response.create"}))

                    elif etype == "error":
                        print(f"OpenAI error: {event['error']}")

            except websockets.exceptions.ConnectionClosed:
                pass

        # Run both directions concurrently
        await asyncio.gather(receive_from_twilio(), receive_from_openai())


async def _configure_session(ws):
    """Send session configuration to OpenAI Realtime."""
    await ws.send(json.dumps({
        "type": "session.update",
        "session": {
            "instructions": SYSTEM_PROMPT,
            "voice": "alloy",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {
                "type": "server_vad",
                "silence_duration_ms": 500,
                "threshold": 0.5,
                "prefix_padding_ms": 200,
            },
            "tools": [
                {
                    "type": "function",
                    "name": "search_knowledge_base",
                    "description": "Search the Atlas knowledge base for information.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            ],
            "input_audio_transcription": {"model": "whisper-1"},
        },
    }))


async def _execute_tool(name: str, arguments_json: str) -> dict:
    """Dispatch tool calls from the Realtime API."""
    args = json.loads(arguments_json)
    if name == "search_knowledge_base":
        # Replace with real vector search in production
        return {"result": f"No results found for: {args['query']}"}
    return {"error": f"Unknown tool: {name}"}
