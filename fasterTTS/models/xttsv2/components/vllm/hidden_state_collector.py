import threading
from typing import Optional

import torch

class HiddenStatesCollector:
    def __init__(self):
        self.outputs = {}
        self.lock = threading.Lock()

    def __call__(self, outputs: Optional[torch.Tensor], request_id: str):
        """Save outputs for a specific request"""
        with self.lock:
            if request_id not in self.outputs:
                self.outputs[request_id] = []
            self.outputs[request_id].append(outputs)

    def get_hidden_states(self, request_id) -> Optional[torch.Tensor]:
        with self.lock:
            outputs = self.outputs.pop(request_id, None)
        if outputs is not None:
            outputs = torch.cat(outputs, dim=0)
        return outputs

    def bind_to_request(self, request_id: str):
        def bound_collector(outputs: Optional[torch.Tensor], _request_id: str = None):
            self(outputs, request_id)
        return bound_collector
