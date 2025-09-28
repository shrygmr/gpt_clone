# utils/lc_chain.py
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_memory_chain(
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    system_prompt: str = "You are a helpful assistant."
):
    """
    LangChain LLMChain + kısa süreli hafıza (uyumlu prompt ile).
    """
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )

    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="input",
        return_messages=True,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False,
    )
    return chain
