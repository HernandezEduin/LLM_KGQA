valid_models = [
    'gemma3',
    'gemma4', 
    'llama3', 
    'llama3.1', 
    'deepseek-r1', 
    'qwen2.5', 
    'gpt-oss', 
    'mixtral', 
    'vicuna', 
    'phi3'
]

has_instruct_versions = {
    'gemma3': False,
    'gemma4': False,
    'llama3': False, 
    'llama3.1': True, 
    'deepseek-r1': False, 
    'qwen2.5': True, 
    'gpt-oss': False, 
    'mixtral': False, 
    'phi3': False
}

has_quantized_versions = {
    'gemma3': False,
    'gemma4': False,
    'llama3': False, 
    'llama3.1': True, 
    'deepseek-r1': False, 
    'qwen2.5': True, 
    'gpt-oss': False, 
    'mixtral': False, 
    'phi3': False
}

context_window_limits = {
    'gemma3': 128*1024,
    'gemma4': 128*1024,
    'llama3': 8*1024,
    'llama3.1': 128*1024,
    'deepseek-r1': 128*1024,
    'qwen2.5': 32*1024,
    'gpt-oss': 128*1024,
    'mixtral': 32*1024,
    'vicuna': 4*1024,
    'phi3': 128*1024,
}