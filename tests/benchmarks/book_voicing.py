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
    tts = TTS(scheduler_max_concurrency=20).from_pretrained("AstraMindAI/xttsv2", torch_dtype=torch.float32)
    req = TTSRequest(
            text=text,
            language="en",
            speaker_files=[speaker_file],
            stream=False
        )

    start_time = time.time()

    # Esegui le richieste in parallelo
    results = tts.generate_speech(req)

    print(f"Tempo di esecuzione: {time.time() - start_time:.2f} secondi")
    results.save('harry_potar.wav')

if __name__ == "__main__":
    asyncio.run(main())