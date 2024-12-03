import requests
import base64
import json
import os
import asyncio
import threading
from queue import Queue
from typing import Generator
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from auralis import TTSOutput


class AudioPlayer:
    def __init__(self):
        self.audio_queue = Queue()
        self.playing = True
        self.player_thread = threading.Thread(target=self._play_audio_worker)
        self.player_thread.daemon = True
        self.player_thread.start()

    def _play_audio_worker(self):
        while self.playing:
            try:
                audio_data = self.audio_queue.get()
                if audio_data is None:  # Sentinel value to stop
                    break
                TTSOutput(array=audio_data).play()
            except Exception as e:
                print(f"Error playing audio: {e}")
            finally:
                self.audio_queue.task_done()

    def add_audio(self, audio_data):
        self.audio_queue.put(audio_data)

    def stop(self):
        self.playing = False
        self.audio_queue.put(None)  # Add sentinel value
        self.player_thread.join()


def process_stream_response(url: str, headers: dict, payload: dict):
    """
    Process streaming response from TTS server with async audio playback
    """
    audio_player = AudioPlayer()

    try:
        with requests.post(url, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode('utf-8')
                if not line.startswith('data: '):
                    continue

                data_str = line[6:]  # Remove 'data: ' prefix
                if data_str == '[DONE]':
                    break

                try:
                    data = json.loads(data_str)

                    # Handle text chunks
                    if data.get("object") == "chat.completion.chunk":
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content")
                        if content:
                            print(f"{content}", end="", flush=True)

                    # Handle audio chunks
                    elif data.get("object") == "audio.chunk":
                        audio_data = base64.b64decode(data["data"])
                        audio_player.add_audio(audio_data)

                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON data: {e}")
                    continue
    finally:
        audio_player.stop()


def process_stream_with_openai(
        client: OpenAI,
        model: str,
        speaker_data: str,
        llm_runtime_url: str,
        vocalize_every_n_words: int,
        prompt: str,
        auralis_params: dict = None
):
    """
    Process stream using OpenAI SDK with Auralis parameters

    Args:
        client: OpenAI client instance
        model: Model to use for text generation
        speaker_data: Base64 encoded audio data for voice cloning
        llm_runtime_url: URL of the LLM endpoint
        vocalize_every_n_words: Number of words after which to generate audio
        prompt: Text prompt for generation
        auralis_params: Optional dictionary of Auralis-specific parameters
    """
    audio_player = AudioPlayer()

    # Default Auralis parameters
    default_auralis_params = {
        'auralis_enhance_speech': True,
        'auralis_sound_norm_refs': False,
        'auralis_max_ref_length': 60,
        'auralis_gpt_cond_len': 30,
        'auralis_gpt_cond_chunk_len': 4,
        'auralis_temperature': 0.75,
        'auralis_top_p': 0.85,
        'auralis_top_k': 50,
        'auralis_repetition_penalty': 5.0,
        'auralis_length_penalty': 1.0,
        'auralis_do_sample': True,
        'auralis_language': "auto"
    }

    # Update defaults with any provided parameters
    if auralis_params:
        default_auralis_params.update(auralis_params)

    try:
        # Start Auralis request
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            extra_body={
                'openai_api_url': llm_runtime_url,
                "speaker_files": [speaker_data],
                'vocalize_at_every_n_words': vocalize_every_n_words,
                **default_auralis_params  # Include all Auralis parameters
            }
        )

        for chunk in stream:
            if not chunk:
                continue
            if getattr(chunk, 'object', None) == 'audio.chunk':
                # Handle audio chunk
                audio_data = base64.b64decode(chunk.data)
                audio_player.add_audio(audio_data)
            elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                # Handle text chunk
                print(f"{chunk.choices[0].delta.content}", end="", flush=True)

    finally:
        audio_player.stop()



if __name__ == "__main__":
    # Read reference audio
    with open("../resources/audio_samples/female.wav", "rb") as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')

    ### REQUEST WITH CURL LIKE INTERFACE

    ## Request configuration
    #url = "http://localhost:8000/v1/chat/completions"
    #headers = {
    #    "Authorization": "Bearer your-openai-api-key"
    #}
    #payload = {
    #    "model": "meta-llama/Llama-3.2-1B-Instruct",
    #    "openai_api_url": "http://127.0.0.1:8001/v1/chat/completions",
    #    "messages": [{"role": "user", "content": "Tell me a story about a brave knight"}],
    #    "speaker_files": [audio_data],
    #    "vocalize_at_every_n_words": 25,
    #    "stream": True
    #}
    #
    ## Process stream
    #process_stream_response(url, headers, payload)

    # or with OAI
    client = OpenAI(
        api_key="your-openai-api-key",
        base_url="http://127.0.0.1:8000/v1/",  # insert the auralis endpoint, NOT the LLM generation endpoint
    )
    # Optional: customize Auralis parameters
    custom_auralis_params = {
        'auralis_temperature': 0.8,
        'auralis_language': 'en',
        'auralis_enhance_speech': True
    }
    # Process stream using OpenAI
    process_stream_with_openai(
        client=client,
        model='meta-llama/Llama-3.2-1B-Instruct',
        llm_runtime_url="http://127.0.0.1:8001/v1/chat/completions",
        speaker_data=audio_data,
        vocalize_every_n_words=40,
        prompt="Tell me a story about a brave knight",
        auralis_params=custom_auralis_params  # Optional: pass custom parameters
    )