import gradio as gr
import torch
from transformers import pipeline
from gtts import gTTS
import time
import os
import fastapi
import socketio
import asyncio
import subprocess
import numpy as np
import shutil


# --- 1. Configuração do Modelo (sem alterações) ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
transcribe_pipeline = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-base",
    chunk_length_s=30,
    device=device,
)

# --- 1.1 Configuração VAD (Silero) ---
try:
    # Carrega o modelo VAD do Silero
    # trust_repo=True é necessário para evitar avisos/erros em versões recentes
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    # Desempacota apenas o que precisamos. Ignoramos read_audio padrão pois usaremos ffmpeg
    (get_speech_timestamps, _, _, _, _) = utils
    vad_model.to(device)
    print("Modelo VAD carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar VAD: {e}")
    vad_model = None


def read_audio_ffmpeg(file_path):
    """
    Lê áudio usando ffmpeg e converte para tensor float32 (16kHz, mono).
    Isso evita dependências complexas de backend de áudio do python (torchaudio/soundfile).
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        # Fallback to default command if not found (though likely to fail if not in PATH)
        ffmpeg_path = "ffmpeg"

    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file_path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-",
    ]
    try:
        # Executa ffmpeg e captura a saída (bytes raw PCM)
        # Capture stderr to debug
        process = subprocess.run(cmd, capture_output=True, check=True)
        out = process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Erro ffmpeg ao ler {file_path}: {e}")
        print(f"FFmpeg stderr: {e.stderr.decode('utf-8', errors='replace')}")
        return None

    # Converte bytes para numpy array int16, depois para float32 normalizado entre -1 e 1
    return torch.from_numpy(
        np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    )


# --- 2. Configuração do Servidor WebSocket (Lógica de Streaming Atualizada) ---
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app_fastapi = fastapi.FastAPI()
sio_asgi_app = socketio.ASGIApp(sio)
app_fastapi.mount("/socket.io", sio_asgi_app)

# Dicionários para gerenciar o estado de cada cliente
client_states = {}


@sio.on("connect")
async def connect(sid, environ):
    print(f"Cliente conectado: {sid}")
    # Inicializa o estado para um novo cliente
    client_states[sid] = {
        "audio_buffer": bytearray(),
        "last_transcription": "",
        "processing_task": None,
    }


@sio.on("disconnect")
async def disconnect(sid):
    print(f"Cliente desconectado: {sid}")
    if sid in client_states:
        # Cancela qualquer tarefa de processamento pendente
        if client_states[sid]["processing_task"]:
            client_states[sid]["processing_task"].cancel()
        del client_states[sid]


async def process_audio_chunk_for_transcription(sid):
    """Função que executa a transcrição em um chunk de áudio."""
    state = client_states.get(sid)
    if (
        not state or len(state["audio_buffer"]) < 16384
    ):  # Não processa se for muito pequeno
        return

    print(f"Processando chunk de áudio para {sid}...")
    temp_path = f"temp_stream_{sid}.webm"
    with open(temp_path, "wb") as f:
        f.write(state["audio_buffer"])

    try:
        # --- VAD CHECK ---
        if vad_model:
            # Lê o áudio salvo usando ffmpeg para garantir robustez
            wav = read_audio_ffmpeg(temp_path)

            if wav is not None:
                # Executa VAD com threshold ajustado
                speech_timestamps = get_speech_timestamps(
                    wav, vad_model, sampling_rate=16000, threshold=0.4
                )

                if not speech_timestamps:
                    print(
                        f"VAD: Nenhuma fala detectada para {sid}. Pulando transcrição."
                    )
                    return
                else:
                    print(
                        f"VAD: Fala detectada ({len(speech_timestamps)} segmentos). Prosseguindo."
                    )
            else:
                print("VAD: Falha ao ler áudio. Abortando transcrição para evitar ruído.")
                return

        # Executa a transcrição pesada em uma thread separada
        result = await asyncio.to_thread(
            transcribe_pipeline,
            temp_path,
            batch_size=8,
            generate_kwargs={
                "language": "portuguese", 
                "task": "transcribe"
            },
        )
        transcription = result["text"].strip() if result and result["text"] else ""

        # Lógica simples para evitar repetição: só atualiza se a nova transcrição for maior
        if len(transcription) > len(state["last_transcription"]):
            print(f"Transcrição Parcial ({sid}): {transcription}")
            state["last_transcription"] = transcription
            # Envia a atualização para a UI
            await sio.emit("partial_transcription", {"text": transcription}, room=sid)

    except Exception as e:
        print(f"Erro na transcrição de chunk: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@sio.on("audio_chunk")
async def handle_audio_chunk(sid, data):
    state = client_states.get(sid)
    if state:
        state["audio_buffer"].extend(data)
        # Se nenhuma tarefa de processamento estiver rodando, inicia uma nova
        if state["processing_task"] is None or state["processing_task"].done():
            # Processa o áudio a cada 2 segundos para simular tempo real
            state["processing_task"] = asyncio.create_task(asyncio.sleep(2.0))
            state["processing_task"].add_done_callback(
                lambda _: asyncio.create_task(
                    process_audio_chunk_for_transcription(sid)
                )
            )


@sio.on("stop_streaming")
async def handle_stop_streaming(sid):
    print(f"Streaming finalizado para {sid}, gerando resposta final.")
    state = client_states.get(sid)
    if not state:
        return

    # Cancela qualquer tarefa pendente para evitar uma última transcrição desnecessária
    if state["processing_task"]:
        state["processing_task"].cancel()

    # Usa a última transcrição completa obtida
    final_transcription = state["last_transcription"]
    print(f"Transcrição Final Usada ({sid}): {final_transcription}")

    # Reinicia o buffer e a transcrição para a próxima gravação
    state["audio_buffer"] = bytearray()
    state["last_transcription"] = ""

    # Lógica para gerar a resposta de Robert (sem alterações)
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

    socket.on('connect', () => console.log('Conectado!'));

    // NOVO: Listener para transcrições parciais
    socket.on('partial_transcription', (data) => {
        console.log('Parcial:', data.text);
        display.innerText = data.text || "Ouvindo...";
    });

    socket.on('final_response', (data) => {
        const audio = new Audio('/file=' + data.audio_path);
        audio.play();
        button.disabled = false;
        button.innerText = "Pressione para Falar";
    });

    const toggleRecording = async () => {
        if (isRecording) {
            mediaRecorder.stop();
            const stream = mediaRecorder.stream;
            stream.getTracks().forEach(track => track.stop()); // Libera o microfone
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
                mediaRecorder.stream = stream; // Salva a stream para poder pará-la depois
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        socket.emit('audio_chunk', event.data);
                    }
                };
                
                // Envia o primeiro chunk rapidamente, depois a cada segundo
                mediaRecorder.start(1000); 
            } catch (err) {
                alert("Erro ao acessar o microfone.");
            }
        }
    };
    button.onclick = toggleRecording;
}
window.addEventListener('load', main);
</script>
"""

with gr.Blocks(css=css, head=js_code) as demo:
    # O HTML da UI não muda
    gr.HTML(f"""<h2 style="text-align: center;">Assistente Médico Robert (SESA)</h2>""")
    gr.HTML(
        f"""<div id="transcription_display">Pressione o botão para começar a falar...</div>"""
    )
    gr.HTML(
        f"""<div style="display:flex; justify-content:center; margin-top:20px;"><button id="recordButton" class="record-button">Pressione para Falar</button></div>"""
    )

app = gr.mount_gradio_app(app_fastapi, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
