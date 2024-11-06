import torch
import gc
import asyncio
from torch.profiler import profile, record_function, ProfilerActivity

from fasterTTS.common.definitions.requests import TTSRequest
from fasterTTS.models.xttsv2.XTTSv2 import XTTSv2Engine
from fasterTTS.core.tts import TTS

# Sample Text
text = """La Storia di Villa Margherita

L'antica Villa Margherita si ergeva maestosa sulla collina che dominava il piccolo paese di San Lorenzo, circondata da cipressi centenari e giardini rigogliosi. Era stata costruita nel 1876 dal Conte Alessandro Visconti per la sua giovane sposa, Margherita, da cui prese il nome.

... [Text Truncated for Brevity] ...

E così, grazie all'impegno di un'intera comunità, Villa Margherita continuò a dominare la collina di San Lorenzo, non più come un peso da sostenere, ma come un faro di cultura e memoria collettiva, pronta ad accogliere nuove storie e nuovi sogni.
"""

speaker_file = "/home/marco/PycharmProjects/betterVoiceCraft/female.wav"

# Initialize the TTS engine
tts = TTS()
tts.tts_engine = XTTSv2Engine.from_pretrained(
    "/home/marco/PycharmProjects/betterVoiceCraft/hf_converted_files/xtts2",
    torch_dtype=torch.float32
)

async def main():
    # Create TTS requests
    request1 = TTSRequest(
        text=text,
        language="it",
        speaker_files=[speaker_file],
        stream=True
    )

    request2 = TTSRequest(
        text=text,
        language="it",
        speaker_files=[speaker_file],
        stream=True
    )

    requests = [request1, request2]

    # Define coroutines for the requests
    coroutines = [tts.generate_speech_async(req) for req in requests]

    # Execute the requests in parallel
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    # Coroutine to process each result
    async def process_result(idx, result):
        if isinstance(result, Exception):
            print(f"Error in request {idx}: {result}")
        else:
            async for chunk in result:
                pass  # Process the audio chunk as needed
                # Example: Save or play the chunk
                # print(f"Received audio chunk for request {idx}: {chunk}")

    # Process all results in parallel
    await asyncio.gather(
        *(process_result(idx, result) for idx, result in enumerate(results, 1)),
        return_exceptions=True
    )
    print("All TTS requests have been processed.")

def profile_function(target_function, *args, **kwargs):
    """
    Profiles the given target function, handling both synchronous and asynchronous functions.
    """
    if asyncio.iscoroutinefunction(target_function):
        # If the target function is asynchronous
        async def async_wrapper():
            return await target_function(*args, **kwargs)

        # Define an asynchronous profiling context
        async def profile_async():
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                with record_function("Function Execution"):
                    result = await async_wrapper()

            # Print the profiler summary sorted by CUDA memory usage
            print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=30))

            # Collect all objects
            gc.collect()
            tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda]

            # Create a list of tuples (variable_name, memory_usage_in_MB)
            memory_variables = []
            for tensor in tensors:
                memoria = tensor.element_size() * tensor.nelement() / (1024 ** 2)  # Convert to MB
                memory_variables.append((f"id={id(tensor)}", memoria))

            # Sort variables by memory usage in descending order
            memory_variables.sort(key=lambda x: x[1], reverse=True)

            # Get the top 30 variables
            top_30 = memory_variables[:30]

            print("\nTop 30 Variables by VRAM Usage (in MB):")
            for name, memoria in top_30:
                print(f"{name}: {memoria:.2f} MB")

        # Run the asynchronous profiler
        asyncio.run(profile_async())
    else:
        # If the target function is synchronous
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function("Function Execution"):
                result = target_function(*args, **kwargs)

        # Print the profiler summary sorted by CUDA memory usage
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=30))

        # Collect all objects
        gc.collect()
        tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda]

        # Create a list of tuples (variable_name, memory_usage_in_MB)
        memory_variables = []
        for tensor in tensors:
            memoria = tensor.element_size() * tensor.nelement() / (1024 ** 2)  # Convert to MB
            memory_variables.append((f"id={id(tensor)}", memoria))

        # Sort variables by memory usage in descending order
        memory_variables.sort(key=lambda x: x[1], reverse=True)

        # Get the top 30 variables
        top_30 = memory_variables[:30]

        print("\nTop 30 Variables by VRAM Usage (in MB):")
        for name, memoria in top_30:
            print(f"{name}: {memoria:.2f} MB")

if __name__ == "__main__":
    profile_function(main)
