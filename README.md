import os
import psutil
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_generate_batch(model_name, prompts, cpu_range):
    # Bind current process/thread to specific CPU range
    p = psutil.Process(os.getpid())
    p.cpu_affinity(cpu_range)
    torch.set_num_threads(len(cpu_range))

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()

    # Tokenize batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids

    # Run batch generate
    outputs = model.generate(input_ids, max_new_tokens=20)

    # Decode and print each
    print(f"\n[CPU {cpu_range[0]}-{cpu_range[-1]}] Outputs:")
    for i, output in enumerate(outputs):
        decoded = tokenizer.decode(output, skip_special_tokens=True)
        print(f"  Prompt {i+1}: {decoded}")

# Inputs
batch_1 = ["Hello, how are you?", "Tell me a joke.", "What's the weather today?"]
batch_2 = ["Once upon a time", "Explain relativity.", "What is quantum physics?"]

# Two threads, two CPU ranges
thread1 = threading.Thread(target=run_generate_batch, args=("gpt2", batch_1, list(range(0, 46))))
thread2 = threading.Thread(target=run_generate_batch, args=("gpt2", batch_2, list(range(46, 92))))

# Start and wait
thread1.start()
thread2.start()
thread1.join()
thread2.join()
