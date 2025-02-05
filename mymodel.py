!pip install torch torchvision torchaudio transformers datasets accelerate bitsandbytes peft

!pip install torch --index-url https://download.pytorch.org/whl/cu118

from google.colab import files
uploaded = files.upload()

from datasets import load_dataset

dataset = load_dataset("json", data_files="questions.jsonl")
dataset = dataset["train"].train_test_split(test_size=0.1)
print(dataset)

from huggingface_hub import login
login()

!pip install transformers accelerate bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-v0.1"

# Enable 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Model loaded in 4-bit successfully!")

from huggingface_hub import login
login("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")

prompt = "What is the meaning of life?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Move to GPU if available
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))

while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=100)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Akkyaaa: {response}")
