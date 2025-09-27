import streamlit as st
from utils.llm_backend import get_response, get_response_with_memory

st.title("💬 GPT Clone - Kullanıcı API Key ile")

api_key = st.text_input("🔑 Lütfen OpenAI API Key'inizi girin:", type="password")

if api_key:
    prompt = st.text_area("💭 Sorunuzu yazın:")
    use_memory = st.checkbox("Memory Kullan", value=False)

    if st.button("Gönder") and prompt.strip():
        with st.spinner("Yanıt alınıyor..."):
            try:
                if use_memory:
                    answer = get_response_with_memory(prompt, api_key)
                else:
                    answer = get_response(prompt, api_key)
                st.success(answer)
            except Exception as e:
                st.error(f"Hata: {str(e)}")
