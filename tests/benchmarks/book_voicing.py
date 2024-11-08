import time

import asyncio
import ebooklib
import torch
from ebooklib import epub
from bs4 import BeautifulSoup
import os

from fasterTTS import TTS, TTSRequest, TTSOutput


def extract_text_from_epub(epub_path, output_path=None):
    """
    Extracts text from an EPUB file and optionally saves it to a text file.

    Args:
        epub_path (str): Path to the EPUB file
        output_path (str, optional): Path where to save the text file

    Returns:
        str: The extracted text
    """
    # Load the book
    book = epub.read_epub(epub_path)

    # List to hold extracted text
    chapters = []

    # Extract text from each chapter
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Get HTML content
            html_content = item.get_content().decode('utf-8')

            # Use BeautifulSoup to extract text
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            chapters.append(text)

    # Join all chapters
    full_text = '\n\n'.join(chapters)

    # Save text if output path is specified
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)

    return full_text


def process_multiple_epubs(input_folder, output_folder):
    """
    Process all EPUB files in a folder.

    Args:
        input_folder (str): Folder containing EPUB files
        output_folder (str): Folder where to save text files
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each EPUB file in the folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.epub'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")

            try:
                print(f"Processing: {filename}")
                extract_text_from_epub(input_path, output_path)
                print(f"Successfully processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")




# Example usage
async def main():
# For a single file
    #extract_text_from_epub("/home/marco/Documenti/Harry Potter - Complete Series (J K. Rowling) (Z-Library).epub")
    with open("/home/marco/PycharmProjects/betterVoiceCraft/FasterTTS/tests/benchmarks/harry_potter_full.txt", "r") as f:
        text = f.read()

    speaker_file = "/home/marco/PycharmProjects/betterVoiceCraft/female.wav"
    # Inizializza il TTS
    tts = TTS(scheduler_max_concurrency=60).from_pretrained("AstraMindAI/xttsv2", torch_dtype=torch.float32, pipeline_parallel=3)
    requests = []
    for text_pieces in [text[i:i + 100000] for i in range(0, len(text), 100000)]:
        requests.append(TTSRequest(
            text=text_pieces,
            language="en",
            speaker_files=[speaker_file],
            stream=False
        ))
    # Definisci le coroutines per le richieste
    coroutines = [tts.generate_speech_async(req) for req in requests]
    start_time = time.time()

    # Esegui le richieste in parallelo
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    output_container = {}


     # Crea una coroutine per processare ogni risultato
    async def process_result(idx, result):
            output_container[idx] = []
            if isinstance(result, Exception):
                print(f"Si è verificato un errore nella richiesta {idx}: {result}")
            else:
                async for chunk in result:
                    output_container[idx].append(chunk)



    # Processa tutti i risultati in parallelo
    await asyncio.gather(
            *(process_result(idx, result) for idx, result in enumerate(results, 1)),
            return_exceptions=True
        )
    print(f"Tempo di esecuzione: {time.time() - start_time:.2f} secondi")
    complete_list = []
    for v in output_container.values():
        complete_list.extend(v)
    TTSOutput.combine_outputs(complete_list).save('harry_potar.wav')

if __name__ == "__main__":
    asyncio.run(main())