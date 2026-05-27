"""
gemini_live_agent.py — Twilio Media Stream → Gemini Live API bridge.

The same telephony voice-agent use case as voice_agent_twilio.py, but
using Google's Gemini Live API instead of OpenAI Realtime.

Key differences vs the OpenAI version:
  - SDK-level connection (client.aio.live.connect) instead of raw WebSocket
  - PCM16 at 16kHz in/out (OpenAI uses 24kHz)
  - Interruption is automatic — the server sets interrupted=True; no
    explicit response.cancel needed
  - Voice names: "Puck", "Charon", "Kore", "Fenrir", "Aoede"

Twilio sends PCMU/8kHz. We resample to PCM16/16kHz for Gemini (audioop),
and resample Gemini's PCM16/16kHz output back to PCMU/8kHz for Twilio.

Run:
    uvicorn gemini_live_agent:app --host 0.0.0.0 --port 8080

Configure Twilio:
    Set your phone number's Voice URL to https://<your-host>/twiml
    The TwiML response starts the Media Stream at /media-stream.

Requires:
    pip install google-genai fastapi uvicorn websockets
    Python 3.13+: pip install audioop-lts
"""

import asyncio
import base64
import json
import os

try:
    import audioop                    # Python ≤ 3.12 stdlib
except ImportError:
    import audioop_lts as audioop     # pip install audioop-lts  (Python 3.13+)

from google import genai
from google.genai import types
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

GEMINI_MODEL = "gemini-2.0-flash-live-001"

SYSTEM_PROMPT = """You are Atlas, a helpful voice assistant.
Answer concisely — spoken answers should be under 3 sentences.
Do not say "certainly", "absolutely", or "of course".
"""

client = genai.Client()


def _make_live_config() -> types.LiveConnectConfig:
    """Build the session configuration for Gemini Live."""
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_PROMPT,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
            )
        ),
        tools=[{
            "function_declarations": [{
                "name": "search_knowledge_base",
                "description": "Search the Atlas knowledge base for information.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {"query": {"type": "STRING"}},
                    "required": ["query"],
                },
            }]
        }],
    )


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
    Bridge between Twilio Media Stream and Gemini Live API.

    Audio path in:  Twilio PCMU/8kHz → PCM16/16kHz → Gemini Live
    Audio path out: Gemini Live PCM16/16kHz → PCMU/8kHz → Twilio
    """
    await twilio_ws.accept()
    stream_sid = None

    async with client.aio.live.connect(model=GEMINI_MODEL, config=_make_live_config()) as session:

        async def receive_from_twilio():
            nonlocal stream_sid
            try:
                async for raw in twilio_ws.iter_text():
                    msg = json.loads(raw)
                    event = msg.get("event")

                    if event == "start":
                        stream_sid = msg["start"]["streamSid"]

                    elif event == "media":
                        # Twilio: PCMU/8kHz → PCM16/16kHz for Gemini
                        mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                        pcm8k = audioop.ulaw2lin(mulaw_bytes, 2)          # μ-law → PCM16
                        pcm16k = audioop.ratecv(pcm8k, 2, 1, 8000, 16000, None)[0]
                        await session.send(
                            input={"data": pcm16k, "mime_type": "audio/pcm;rate=16000"},
                            end_of_turn=False,
                        )

                    elif event == "stop":
                        break

            except WebSocketDisconnect:
                pass

        async def receive_from_gemini():
            try:
                async for response in session.receive():
                    # Audio response — convert PCM16/16kHz → PCMU/8kHz for Twilio
                    if response.data and stream_sid:
                        pcm8k = audioop.ratecv(response.data, 2, 1, 16000, 8000, None)[0]
                        mulaw = audioop.lin2ulaw(pcm8k, 2)
                        await twilio_ws.send_text(json.dumps({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": base64.b64encode(mulaw).decode()},
                        }))

                    # Server interrupted (user spoke over the response)
                    if response.server_content and response.server_content.interrupted:
                        if stream_sid:
                            # Clear any buffered audio in Twilio's playback queue
                            await twilio_ws.send_text(json.dumps({
                                "event": "clear",
                                "streamSid": stream_sid,
                            }))

                    # Tool call
                    if response.tool_call:
                        for fc in response.tool_call.function_calls:
                            result = await _execute_tool(fc.name, fc.args)
                            await session.send(
                                input=types.LiveClientToolResponse(
                                    function_responses=[
                                        types.FunctionResponse(
                                            id=fc.id,
                                            name=fc.name,
                                            response={"result": result},
                                        )
                                    ]
                                )
                            )

                    # Log errors
                    if hasattr(response, "error") and response.error:
                        print(f"Gemini error: {response.error}")

            except Exception as e:
                print(f"Gemini receive error: {e}")

        await asyncio.gather(receive_from_twilio(), receive_from_gemini())


async def _execute_tool(name: str, args: dict) -> str:
    """Dispatch tool calls from the Gemini Live session."""
    if name == "search_knowledge_base":
        # Replace with real vector search in production
        return f"No results found for: {args.get('query', '')}"
    return f"Unknown tool: {name}"
