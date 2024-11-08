import logging
import sys
from pathlib import Path
from datetime import datetime
import colorama
from colorama import Fore, Back, Style
from typing import Optional, Union
import re
import traceback
import copy
import os

# Initialize colorama
colorama.init()


class VLLMLogOverrider:
    """Override VLLM loggers to show only performance metrics"""

    def __init__(self, target_logger: logging.Logger):
        self.target_logger = target_logger
        self.perf_pattern = re.compile(
            r"Avg prompt throughput:.+tokens/s,.+GPU KV cache usage:.+CPU KV cache usage:.+"
        )
        self._override_vllm_loggers()

    def _override_vllm_loggers(self):
        """Override all VLLM loggers to use our custom handler"""
        for name in logging.root.manager.loggerDict:
            if name.startswith('vllm'):
                vllm_logger = logging.getLogger(name)
                vllm_logger.handlers.clear()
                vllm_logger.propagate = False
                handler = self._create_redirecting_handler()
                vllm_logger.addHandler(handler)

    def _create_redirecting_handler(self):
        """Create a handler that only shows performance metrics"""

        class RedirectHandler(logging.Handler):
            def __init__(self, target_logger, perf_pattern):
                super().__init__()
                self.target_logger = target_logger
                self.perf_pattern = perf_pattern

            def emit(self, record):
                msg = str(record.msg)
                if record.args:
                    msg = msg % record.args

                # Only show performance metrics
                if self.perf_pattern.search(msg):
                    self.target_logger.info(f"Decoder performance: {msg}")
                # Silently drop all other VLLM logs

        return RedirectHandler(self.target_logger, self.perf_pattern)

class ColoredFormatter(logging.Formatter):
    """Colored formatter with structured output and file location"""

    COLORS = {
        'DEBUG': {
            'color': Fore.CYAN,
            'style': Style.DIM,
            'icon': '🔍'
        },
        'INFO': {
            'color': Fore.GREEN,
            'style': Style.NORMAL,
            'icon': 'ℹ️'
        },
        'WARNING': {
            'color': Fore.YELLOW,
            'style': Style.BRIGHT,
            'icon': '⚠️'
        },
        'ERROR': {
            'color': Fore.RED,
            'style': Style.BRIGHT,
            'icon': '❌'
        },
        'CRITICAL': {
            'color': Fore.WHITE,
            'style': Style.BRIGHT,
            'bg': Back.RED,
            'icon': '💀'
        }
    }

    def format(self, record: logging.LogRecord) -> str:
        colored_record = copy.copy(record)

        # Get color scheme
        scheme = self.COLORS.get(record.levelname, {
            'color': Fore.WHITE,
            'style': Style.NORMAL,
            'icon': '•'
        })

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]

        # Get file location
        file_location = f"{os.path.basename(record.pathname)}:{record.lineno}"

        # Build components
        components = []

        # Check if VLLM message
        is_vllm_msg = "VLLM" in str(record.msg)

        if is_vllm_msg:
            # Special VLLM formatting
            components.append( f"{Fore.BLUE}{timestamp}{Style.RESET_ALL}"
                f"{Fore.WHITE}{Style.DIM}{file_location}{Style.RESET_ALL}"
                f"{scheme['color']}{scheme['style']}{scheme['icon']} {record.levelname:8}{Style.RESET_ALL}"
                f"{scheme['color']}{record.msg}{Style.RESET_ALL}"
            )
        else:
            # Normal log formatting
            components.extend([
                f"{Fore.BLUE}{timestamp}{Style.RESET_ALL}",
                f"{Fore.WHITE}{Style.DIM}{file_location}{Style.RESET_ALL}",
                f"{scheme['color']}{scheme['style']}{scheme['icon']} {record.levelname:8}{Style.RESET_ALL}",
                f"{scheme['color']}{record.msg}{Style.RESET_ALL}"
            ])

            # Add exception info
            if record.exc_info:
                components.append(
                    f"\n{Fore.RED}{Style.BRIGHT}"
                    f"{''.join(traceback.format_exception(*record.exc_info))}"
                    f"{Style.RESET_ALL}"
                )

        return " | ".join(components)


def setup_logger(
        name: Optional[Union[str, Path]] = None,
        level: int = logging.DEBUG
) -> logging.Logger:
    """
    Setup a colored logger with VLLM override and file location

    Args:
        name: Logger name or __file__ for module name
        level: Logging level
    """
    # Get logger name from file path
    if isinstance(name, (str, Path)) and Path(name).suffix == '.py':
        name = Path(name).stem

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Only add handler if none exists
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

        # Override VLLM loggers to use our logger
        VLLMLogOverrider(logger)

    return logger


def set_vllm_logging_level(level: int = logging.INFO):
    """
    Set the logging level for VLLM loggers

    Args:
        level: Logging level to set (e.g., logging.INFO, logging.ERROR)
    """
    for name in logging.root.manager.loggerDict:
        if name.startswith('vllm'):
            logger = logging.getLogger(name)
            logger.setLevel(level)