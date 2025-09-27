import openai

def get_response(prompt: str, api_key: str) -> str:
    """Kullanıcının API key'i ile GPT yanıtı döndürür"""
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def get_response_with_memory(prompt: str, api_key: str) -> str:
    """Memory kullanımı yok, sadece get_response çağırır"""
    return get_response(prompt, api_key)
