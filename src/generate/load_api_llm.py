from generate.openai_chat_batch import AsyncOpenAIClient, OpenAIClient

def load_api_chat_completion(model_name, async_=False, *args, **kargs):
    model_name_qwen = {
        # Qwen
        "qwen-max": "qwen3-max-preview",
        "qwen-plus": "qwen-plus",
        "qwen-turbo": "qwen-turbo",
        "QWQ-32B": "qwq-32b-preview",
        "qwen2.5-72b-instruct": "qwen2.5-72b-instruct",
        "qwen2.5-32b-instruct": "qwen2.5-32b-instruct",
        "qwen1.5-72b-chat": "qwen1.5-72b-chat",
        # # deepseek
        # "deepseek-v3": "deepseek-v3",
        # "deepseek-r1": "deepseek-r1",
    }
    model_name_deepseek = {
        # deepseek
        "deepseek-v3": "deepseek-chat",
        "deepseek-r1": "deepseek-reasoner",
    }
    model_name_openai = {
        # openai
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o",
        "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
        "gpt-4.1": "gpt-4.1",
        "gpt-4.1-nano": "gpt-4.1-nano"
    }
    model_name_01 = {
        "yi-large": "yi-large",
        "yi-34b-chat": "yi-medium"
    }
    model_name_intern = {
        "internlm2_5-20b-chat": "internlm2.5-latest"
    }


    model_name_vllm = {
        "Llama-3.1-8B-Instruct": ("Llama-3.1-8B-Instruct", 8000),
        "Llama-3.2-3B-Instruct": ("Llama-3.2-3B-Instruct", 8000),
        "Ministral-8B-Instruct": ("Ministral-8B-Instruct", 8000),
        "Qwen2.5-7B-Instruct": ("Qwen2.5-7B-Instruct", 7105),
        "qwen3-moe": ("qwen3-moe", 8000),
        "Qwen3-8B": ("Qwen3-8B", 7104),
        "qwen3-30b-moe": ("qwen3-30b-moe", 8000),
    }

    if model_name in list(model_name_vllm.keys()):
        model_name, local_port = model_name_vllm[model_name]
        api_key = "zjj"
        base_url = f"http://localhost:{local_port}/v1/"
    elif model_name == "qwen3-32b":
        api_key = "mem"
        base_url = "http://10.0.2.219:1146/v1"
    elif model_name in list(model_name_qwen.keys()):
        api_key = "sk-f00f70df2abe444a9e930313085cffe9"
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model_name = model_name_qwen[model_name]
    elif model_name in list(model_name_deepseek.keys()):
        api_key = "sk-551e1f1394f9455891c7bd9a01249d8a"
        base_url = "https://api.deepseek.com/v1"
        model_name = model_name_deepseek[model_name]
    elif model_name in list(model_name_openai.keys()):
        api_key = "CDHhGET6CaPgEj6e2itV3xQjO10kVyquZaVW7FJvBlu1j4SYHjc0JQQJ99BKACYeBjFXJ3w3AAAAACOGyMor"
        base_url = "https://foundationmodeldepartment.openai.azure.com/openai/v1"
        model_name = model_name_openai[model_name]
    elif model_name in list(model_name_01.keys()):
        api_key = "a789593637d442b2b78a63e24bac0202"
        base_url = "https://api.lingyiwanwu.com/v1"
        model_name = model_name_01[model_name]
    elif model_name in list(model_name_intern.keys()):
        api_key = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI1MjMwNDUzNCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc0MzMyODUxOCwiY2xpZW50SWQiOiJlYm1ydm9kNnlvMG5semFlazF5cCIsInBob25lIjoiMTc3NjcyNTM1NjIiLCJvcGVuSWQiOm51bGwsInV1aWQiOiJhNDUwOTFjOC1kNTczLTQ2ZmYtOTVjYi0xZTdiYWQ4MTJjYTUiLCJlbWFpbCI6IiIsImV4cCI6MTc1ODg4MDUxOH0.zX3_jTBH28ZvHwOo9QP6w6LmAP2Dae5fF2x4CRkMJRawMCg1EQx435czicQPof2paZyeCVbHpnLjMtfQtgtRRA"
        base_url = "https://chat.intern-ai.org.cn/api/v1/"
        model_name = model_name_intern[model_name]
    else:
        print("pass a wrong opt model")
        return None
    
    if not async_:
        client = OpenAIClient(api_key=api_key, base_url=base_url, model=model_name)
    else:
        client = AsyncOpenAIClient(api_key=api_key, base_url=base_url, model=model_name)

    return client
    