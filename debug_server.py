#!/usr/bin/env python3
"""Debug script to check if tts_engine is properly initialized"""

import sys
sys.path.insert(0, '/home/op/Auralis/src')

from auralis.entrypoints import oai_server

print(f"tts_engine value: {oai_server.tts_engine}")
print(f"tts_engine type: {type(oai_server.tts_engine)}")

# Check if the global is accessible
import auralis.entrypoints.oai_server as server_module
print(f"Module tts_engine: {server_module.tts_engine}")
