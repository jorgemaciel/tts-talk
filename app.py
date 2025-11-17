import gradio as gr
import torch
from transformers import pipeline
from gtts import gTTS
import time
import os
import fastapi
import socketio
import asyncio

# --- 1. Configuração do Modelo (sem alterações) ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
transcribe_pipeline = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-base",
    chunk_length_s=30,
    device=device,
)

# --- 2. Configuração do Servidor WebSocket (com a correção) ---
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app_fastapi = fastapi.FastAPI()
sio_asgi_app = socketio.ASGIApp(sio)
app_fastapi.mount("/socket.io", sio_asgi_app)

audio_buffers = {}


@sio.on("connect")
async def connect(sid, environ):
    print(f"Cliente conectado: {sid}")
    audio_buffers[sid] = bytearray()


@sio.on("disconnect")
async def disconnect(sid):
    print(f"Cliente desconectado: {sid}")
    if sid in audio_buffers:
        del audio_buffers[sid]


@sio.on("audio_chunk")
async def handle_audio_chunk(sid, data):
    if sid in audio_buffers:
        audio_buffers[sid].extend(data)


# ========= INÍCIO DA CORREÇÃO =========
# Removemos o argumento 'data' que não estava sendo usado.
@sio.on("stop_streaming")
async def handle_stop_streaming(sid):
    # ========= FIM DA CORREÇÃO =========
    """Processa o áudio restante e gera a resposta final com TTS."""
    print(f"Streaming finalizado para {sid}, processando resposta final.")
    final_transcription = ""
    buffer = audio_buffers.get(sid, bytearray())
    if len(buffer) > 4096:
        temp_path = f"temp_stream_{sid}_final.webm"
        with open(temp_path, "wb") as f:
            f.write(buffer)
        try:
            result = await asyncio.to_thread(
                transcribe_pipeline,
                temp_path,
                batch_size=8,
                generate_kwargs={"language": "portuguese", "task": "transcribe"},
            )
            final_transcription = result["text"] if result else ""
            print(f"Transcrição Final ({sid}): {final_transcription}")
            await sio.emit(
                "transcription_update",
                {"text": final_transcription, "final": True},
                room=sid,
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    audio_buffers[sid] = bytearray()

    normalized_text = final_transcription.lower().strip()
    trigger_phrases = ["o que você é", "como se chama", "qual seu nome", "quem é você"]
    response_text = (
        "Desculpe, não entendi a pergunta. Por favor, tente perguntar quem eu sou."
    )
    if any(phrase in normalized_text for phrase in trigger_phrases):
        response_text = "Eu sou Robert, o assistente médico da SESA."

    tts = gTTS(text=response_text, lang="pt-br", slow=False)
    response_audio_path = f"response_{sid}_{int(time.time())}.mp3"
    tts.save(response_audio_path)

    await sio.emit("final_response", {"audio_path": response_audio_path}, room=sid)


# --- 3. Interface Gráfica com Gradio (sem alterações) ---
css = """
/* ... (mesmo CSS de antes) ... */
#transcription_display { padding: 15px; margin: 10px 0; border: 1px solid #444; border-radius: 8px; min-height: 50px; background-color: #1a1a1a; color: #f0f0f0; text-align: center; font-size: 1.1em;}
.record-button { background-color: #2c2c2c; color: white; border: 1px solid #555; border-radius: 8px; padding: 15px 30px; font-size: 18px; cursor: pointer; transition: background-color 0.2s; }
.record-button.recording { background-color: #b22222; border-color: #ff4444; }
.record-button:disabled { background-color: #555; cursor: not-allowed; }
"""

js_code = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<script>
function main() {
    const socket = io(window.location.origin, { path: "/socket.io/" });
    const button = document.getElementById('recordButton');
    const display = document.getElementById('transcription_display');

    let mediaRecorder = null;
    let isRecording = false;

    socket.on('connect', () => console.log('Conectado ao servidor WebSocket!'));
    socket.on('disconnect', () => console.log('Desconectado do servidor WebSocket.'));

    socket.on('transcription_update', (data) => {
        if (data.final) {
            display.innerText = data.text || "Não consegui ouvir nada. Tente novamente.";
        }
    });

    socket.on('final_response', (data) => {
        console.log('Recebida resposta final:', data.audio_path);
        const audio = new Audio('/file=' + data.audio_path);
        audio.play();
        
        button.disabled = false;
        button.innerText = "Pressione para Falar";
    });

    const toggleRecording = async () => {
        if (isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            button.classList.remove('recording');
            button.innerText = "Processando...";
            button.disabled = true;
            socket.emit('stop_streaming');
        } else {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                isRecording = true;
                display.innerText = "Ouvindo...";
                button.classList.add('recording');
                button.innerText = "Gravando... (Pressione para Parar)";
                
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        socket.emit('audio_chunk', event.data);
                    }
                };
                
                mediaRecorder.start(1000);
            } catch (err) {
                console.error("Erro ao acessar microfone:", err);
                alert("Erro ao acessar o microfone. Verifique as permissões.");
            }
        }
    };

    button.onclick = toggleRecording;
}

window.addEventListener('load', main);
</script>
"""

with gr.Blocks(css=css, head=js_code) as demo:
    gr.HTML(f"""<h2 style="text-align: center;">Assistente Médico Robert (SESA)</h2>""")
    gr.HTML(
        f"""<div id="transcription_display">A transcrição aparecerá aqui...</div>"""
    )
    gr.HTML(
        f"""<div style="display:flex; justify-content:center; margin-top:20px;"><button id="recordButton" class="record-button">Pressione para Falar</button></div>"""
    )

# Monta a aplicação Gradio dentro do FastAPI
app = gr.mount_gradio_app(app_fastapi, demo, path="/")

# --- 4. Como Executar (sem alterações) ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# Para executar no terminal:
# uvicorn app:app --reload
