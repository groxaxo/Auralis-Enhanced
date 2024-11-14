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


text = """La Storia di Villa Margherita

L'antica Villa Margherita si ergeva maestosa sulla collina che dominava il piccolo paese di San Lorenzo, circondata da cipressi centenari e giardini rigogliosi. Era stata costruita nel 1876 dal Conte Alessandro Visconti per la sua giovane sposa, Margherita, da cui prese il nome.

La villa aveva attraversato diverse epoche, resistendo a due guerre mondiali e numerose trasformazioni sociali. Durante la Seconda Guerra Mondiale, era diventata un rifugio per molte famiglie che fuggivano dai bombardamenti sulla città. Le sue spesse mura di pietra avevano protetto centinaia di vite.

"Non posso credere che vogliano demolirla!" esclamò Maria, l'ultima discendente dei Visconti, durante un'accesa discussione con il consiglio comunale. "Questa villa non è solo un edificio, è la memoria vivente della nostra comunità!"

Il sindaco, seduto dietro la sua scrivania di quercia, sospirò profondamente. "Capisco il suo punto di vista, signorina Visconti, ma i costi di restauro sono proibitivi. La struttura è pericolante, e non possiamo permetterci di rischiare incidenti."

La notizia della possibile demolizione si diffuse rapidamente nel paese, suscitando reazioni contrastanti. Alcuni abitanti sostenevano che fosse giunto il momento di fare spazio a strutture più moderne, mentre altri consideravano la villa un patrimonio culturale insostituibile.

Anna, la bibliotecaria comunale che aveva trascorso l'infanzia giocando nei giardini della villa, decise di agire. Organizzò una raccolta fondi e coinvolse esperti di restauro da tutta Italia. "Se lavoriamo insieme," disse durante un'assemblea pubblica, "possiamo salvare questo pezzo di storia!"

Le settimane seguenti furono caratterizzate da un fermento straordinario. Artigiani locali si offrirono volontari per i lavori di restauro meno specializzati. Gli studenti dell'istituto d'arte realizzarono un documentario sulla storia della villa. Persino alcune aziende della regione decisero di sponsorizzare il progetto.

Il professor Martelli, storico dell'architettura dell'università vicina, scoprì nei sotterranei della villa una collezione di lettere e documenti che rivelavano dettagli inediti sulla sua storia. "Queste carte dimostrano che Villa Margherita fu un importante centro culturale all'inizio del '900," spiegò entusiasta. "Ospitò artisti, scrittori e musicisti da tutta Europa!"

Dopo mesi di lavoro instancabile, la villa cominciò a riprendere il suo antico splendore. Le pareti furono consolidate, il tetto riparato, e i giardini riportati all'originale bellezza. Durante i lavori, gli operai scoprirono affreschi nascosti sotto strati di intonaco e una piccola cappella che era stata murata.

Il giorno della riapertura, l'intera comunità si riunì per celebrare. Maria Visconti, con le lacrime agli occhi, tagliò il nastro rosso all'ingresso. "Oggi non festeggiamo solo il restauro di un edificio," disse commossa, "ma la rinascita di un luogo che continuerà a raccontare storie per le generazioni future."

La villa divenne un museo e centro culturale, ospitando mostre, concerti e laboratori per bambini. Le sue sale, che un tempo rischiavano di essere ridotte in macerie, tornarono a riempirsi di vita e di voci. I giardini diventarono un parco pubblico, dove gli abitanti del paese potevano passeggiare e rilassarsi.

"È incredibile come un edificio possa unire così tante persone," commentò Anna, guardando un gruppo di studenti che disegnavano nel giardino. "Villa Margherita non è più solo un monumento del passato, ma un simbolo di cosa possiamo realizzare quando lavoriamo insieme per un obiettivo comune."

E così, grazie all'impegno di un'intera comunità, Villa Margherita continuò a dominare la collina di San Lorenzo, non più come un peso da sostenere, ma come un faro di cultura e memoria collettiva, pronta ad accogliere nuove storie e nuovi sogni.

?!... "Davvero sorprendente!" esclamò un visitatore. "Non avrei mai immaginato che un edificio potesse racchiudere tante storie." Le sue parole echeggiarono nelle sale della villa, mescolandosi con i sussurri dei visitatori e il fruscio delle foglie dei cipressi centenari che, come sempre, montavano la guardia alla memoria di quel luogo straordinario."""


# Example usage
async def main():
# For a single file
    #extract_text_from_epub("/home/marco/Documenti/Harry Potter - Complete Series (J K. Rowling).epub")
    #with open("/home/marco/PycharmProjects/betterVoiceCraft/FasterTTS/tests/benchmarks/harry_potter_full.txt", "r") as f:
    #    text = f.read()

    speaker_file = "/home/marco/PycharmProjects/betterVoiceCraft/female.wav"
    # Inizializza il TTS
    tts = TTS(scheduler_max_concurrency=24).from_pretrained("AstraMindAI/xttsv2", torch_dtype=torch.float32)
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


if __name__ == "__main__":
    asyncio.run(main())