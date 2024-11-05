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


class VLLMLogKiller(logging.Filter):
    """Completely silence all VLLM loggers except performance metrics"""

    def __init__(self):
        super().__init__()
        self.perf_pattern = re.compile(
            r"Avg prompt throughput:.+tokens/s,.+GPU KV cache usage:.+CPU KV cache usage:.+"
        )
        self._silence_vllm_loggers()

    def _silence_vllm_loggers(self):
        """Recursively find and silence all VLLM loggers"""
        for name in logging.root.manager.loggerDict:
            if name.startswith('vllm'):
                logger = logging.getLogger(name)
                logger.handlers.clear()
                logger.propagate = False
                logger.setLevel(logging.ERROR)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith('vllm'):
            return bool(self.perf_pattern.search(str(record.msg)))
        return True


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

        # Check if VLLM performance log
        is_vllm_perf = (
                record.name.startswith('vllm') and
                "throughput" in str(record.msg)
        )

        if is_vllm_perf:
            # Special VLLM perf formatting
            components.append(
                f"{Fore.BLUE}{timestamp}{Style.RESET_ALL} | "
                f"{Fore.MAGENTA}VLLM{Style.RESET_ALL} | "
                f"{Fore.CYAN}{record.msg}{Style.RESET_ALL}"
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
    Setup a colored logger with VLLM filtering and file location

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

        # Add VLLM killer to root logger if not already added
        root_logger = logging.getLogger()
        if not any(isinstance(f, VLLMLogKiller) for f in root_logger.filters):
            killer = VLLMLogKiller()
            root_logger.addFilter(killer)

        logger.addHandler(console_handler)

    return logger


# Optional utility for temporary VLLM logging
def enable_vllm_logging(enable: bool = True):
    level = logging.INFO if enable else logging.ERROR
    for name in logging.root.manager.loggerDict:
        if name.startswith('vllm'):
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.propagate = enable