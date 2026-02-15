from io import BytesIO
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import librosa

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", 
    torch_dtype = torch.bfloat16, 
    trust_remote_code=True,
    device_map = "auto"
    )

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct" ,trust_remote_code=True)

# test one audio file
file_path = "/projects/bgbh/datasets/l2-arctic/data/arabic/ABA_idx00000.wav"
audio_data, sr = librosa.load(file_path, sr=16000)

messages = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": file_path},
        {"type": "text", "text": "Transcribe this in English."}
    ]}
]

text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

inputs = processor(text=text, audio= audio_data, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

output = model.generate(
    **inputs, 
    max_new_tokens=128,
    output_hidden_states= True,  
    return_dict_in_generate=True
    )

prompt_length = inputs['input_ids'].size(1)
response_ids = output.sequences[:, prompt_length:]

#extract second to last hidden layer
first_step = output.hidden_states[0]
second_to_last = first_step[-2]

#process prompt
response = processor.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(f"\nModel Response: {response}")
print(f"Embedding Shape: {second_to_last}")