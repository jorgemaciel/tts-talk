import asyncio
import os
import sounddevice as sd
import google.genai as genai
from google.genai.types import (
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    SpeechConfig,
    VoiceConfig,
)

# Configure sua chave de API do Google
# Certifique-se de ter a variável de ambiente GEMINI_API_KEY definida
try:
    genai.Client(api_key="AIzaSyCVTlCMt5tPYfGJUMa2okxUcz3w853Ng-w")
except KeyError:
    print("Erro: A variável de ambiente GEMINI_API_KEY não foi definida.")
    exit()

# --- Configurações de Áudio ---
SAMPLE_RATE = 16000  # Taxa de amostragem para entrada e saída
CHANNELS = 1  # Mono
BLOCKSIZE = 2048  # Tamanho do bloco para o stream de áudio

# Fila para armazenar os dados de áudio do microfone
input_queue = asyncio.Queue()


def audio_callback(indata, frames, time, status):
    """Callback para capturar áudio do microfone e colocar na fila."""
    if status:
        print(status)
    input_queue.put_nowait(bytes(indata))


async def audio_stream_generator():
    """Gera um fluxo de áudio a partir da fila de entrada."""
    while True:
        chunk = await input_queue.get()
        if chunk is None:
            break
        yield chunk


async def main():
    """Função principal para executar o bate-papo por voz."""
    print("Iniciando a sessão de bate-papo por voz com o Gemini...")
    print("Fale agora! Diga 'tchau' para encerrar.")

    # Configuração da API Live do Gemini
    config = LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=SpeechConfig(
            voice_config=VoiceConfig(
                prebuilt_voice_config=PrebuiltVoiceConfig(
                    # Você pode escolher outras vozes, como "Charon", "Kore", etc.
                    voice_name="Puck",
                )
            )
        ),
    )

    try:
        # Conecta-se à API Live usando o modelo correto
        async with (
            genai.Client.aio.live.connect(
                model="models/gemini-1.5-pro-latest",  # Modelo correto para streaming de áudio
                config=config,
            ) as session
        ):
            print("Conectado! A sessão está ativa.")

            # Inicia o stream de áudio do microfone
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=BLOCKSIZE,
                callback=audio_callback,
            ):
                # Inicia o stream de resposta do Gemini
                async for audio_chunk in session.start_stream(
                    stream=audio_stream_generator(),
                    mime_type=f"audio/pcm;rate={SAMPLE_RATE}",
                ):
                    # Reproduz o áudio recebido do Gemini
                    if audio_chunk.data:
                        sd.play(audio_chunk.data, samplerate=SAMPLE_RATE)
                        sd.wait()

                    # Condição de parada (exemplo simples)
                    # Uma implementação mais robusta usaria detecção de palavras-chave
                    if "tchau" in session.last_user_text.lower():
                        print("Encerrando a sessão...")
                        await input_queue.put(None)  # Sinaliza o fim do stream
                        break

    except Exception as e:
        print(f"Ocorreu um erro: {e}")
    finally:
        print("Sessão finalizada.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário.")
