# FasterTTS

This repo is intented to be a faster version of the [Coqui XTTSv2]() project. 
The main goal is to make the inference time faster, ayncronous by using a modified implementation of the vllm for the gpt part with a schedule rto manage and orchestrate requests.
this makes this library almos 8x faster than xttsv2 standard code, and it is tought to be modular, to beign able to plug and play different tts models later on

## Installation

```python
pip install fastertts
```

