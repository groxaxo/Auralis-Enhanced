import argparse
import logging
from typing import Optional

from fasterTTS.core.tts import TTS
from components.serving_engine import OpenAICompatibleTTS
from components.tts_server import create_server
from fasterTTS.common.logging.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='OpenAI-compatible TTS Server')

    # Server configuration
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to run the server on')

    # Model configuration
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the pretrained TTS model')
    parser.add_argument('--model-device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run the model on')

    # Logging configuration
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')

    return parser.parse_args()


def initialize_tts(model_path: str, device: str) -> Optional[OpenAICompatibleTTS]:
    """Initialize the TTS model and wrapper."""
    try:
        # Initialize base TTS model
        tts = TTS()
        tts.from_pretrained(
            model_name_or_path=model_path,
            device=device
        )

        # Create OpenAI-compatible wrapper
        return OpenAICompatibleTTS(tts)

    except Exception as e:
        logging.error(f"Failed to initialize TTS model: {str(e)}")
        return None


def main():
    # Parse command line arguments
    args = parse_args()

    # Setup logging
    logger = setup_logger(__file__, args.log_level)

    # Initialize TTS
    logger.info(f"Initializing TTS model from {args.model_path}")
    tts_wrapper = initialize_tts(args.model_path, args.model_device)

    if tts_wrapper is None:
        logger.error("Failed to initialize TTS. Exiting.")
        return 1

    # Create and run server
    try:
        logger.info(f"Starting server on {args.host}:{args.port}")
        server = create_server(tts_wrapper)
        server.run(host=args.host, port=args.port)
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())