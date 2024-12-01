---
name: Bug report
about: Create a report to help us improve
title: "[BUG\U0001F41B]"
labels: ''
assignees: ''

---

## Bug Description
[Provide a clear and concise description of the bug]

## Minimal Reproducible Example

```python
from auralis import TTS, TTSRequest
import logging

request = TTSRequest(text="Hello, world!", speaker_files=["speaker1.WAV"])
tts = TTS(scheduler_max_concurrency=1, vllm_logging_level=logging.ERROR).from_pretrained("AstraMindAI/xttsv2", gpt_model="AstraMindAI/xtts2-gpt")
out = tts.generate_speech(request)
```


## Expected Behavior
[Describe what you expected to happen]

## Actual Behavior
[Describe what actually happened]

## Error Logs
```
[Paste relevant error logs here, ensuring the logging level is set to DEBUG]
```

## Environment
Please run the following commands and include the output:

```bash
# OS Information
uname -a

# Python version
python --version

# Installed Python packages
pip list

# GPU Information (if applicable)
nvidia-smi

# CUDA version (if applicable)
nvcc --version
```

## Possible Solutions
[If you have ideas on how to solve the issue, include them here]

## Additional Information
[Any other information you think might be helpful for diagnosing the issue]
