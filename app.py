# app.py
import os
import time
from typing import Dict, List, Generator, Optional

import streamlit as st
from utils.llm_core import get_backend, build_messages, estimate_tokens



from utils.store import save_session_to_path, load_session_from_fileobj, serialize_session
from utils.lc_chain import get_memory_chain

st.set_page_config(
    page_title="ChatGPT Clone",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Sidebar: settings
# -----------------------------
st.sidebar.title("⚙️ Chat Settings")
backend_name = st.sidebar.selectbox("Backend", ["openai"], index=0)

# API key from user (stored only in session)
api_key_input = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Your key is kept only in this browser session (st.session_state)."
)
if api_key_input:
    st.session_state["api_key"] = api_key_input.strip()
api_key = st.session_state.get("api_key")

model = st.sidebar.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
system_prompt = st.sidebar.text_area("System prompt (optional)", value="")

# LangChain memory toggle
st.sidebar.subheader("🧠 Memory (LangChain)")
use_lc_memory = st.sidebar.toggle(
    "Enable short-term memory",
    value=True,
    help="Keeps recent conversation using LangChain ConversationBufferMemory."
)

# Token & cost settings
st.sidebar.subheader("💲 Token & Cost (est.)")
col_rate_in, col_rate_out = st.sidebar.columns(2)
with col_rate_in:
    rate_in = st.number_input(
        "Input $/1K tok",
        min_value=0.0,
        value=0.15,
        step=0.01,
        help="Edit to match your model pricing"
    )
with col_rate_out:
    rate_out = st.number_input(
        "Output $/1K tok",
        min_value=0.0,
        value=0.60,
        step=0.01,
        help="Edit to match your model pricing"
    )

# Persistence controls
st.sidebar.subheader("💾 Persistence")
filename = st.sidebar.text_input("Filename", value="history.json")
c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("Save Session"):
        save_session_to_path(
            path=filename,
            messages=st.session_state.get("messages", []),
            feedback=st.session_state.get("feedback", {}),
        )
        st.sidebar.success(f"Saved → {filename}")
with c2:
    uploaded = st.sidebar.file_uploader("Load from file", type=["json"], label_visibility="collapsed")
    if uploaded is not None:
        data = load_session_from_fileobj(uploaded)
        st.session_state["messages"] = data.get("messages", [])
        st.session_state["feedback"] = data.get("feedback", {})
        st.session_state["edit_mode"] = {}
        st.sidebar.success("Session loaded.")

col_clear, col_export, col_reset_key = st.sidebar.columns(3)
if col_clear.button("Clear Conversation"):
    st.session_state["messages"] = []
    # Reset LC chain memory as well
    st.session_state.pop("lc_chain_key", None)
    st.session_state.pop("lc_memory_chain", None)
if col_export.button("Export JSON"):
    st.sidebar.download_button(
        label="Download history.json",
        data=serialize_session(
            messages=st.session_state.get("messages", []),
            feedback=st.session_state.get("feedback", {}),
        ),
        file_name="history.json",
        mime="application/json",
        use_container_width=True,
    )
if col_reset_key.button("Reset Key"):
    st.session_state.pop("api_key", None)

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict] = []
if "feedback" not in st.session_state:
    st.session_state.feedback: Dict[int, Optional[str]] = {}
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode: Dict[int, bool] = {}

# Backend instance (uses user API key if provided)
backend = get_backend(backend_name, api_key=api_key)

# Prepare/reuse LangChain memory chain
if use_lc_memory and api_key:
    chain_key = f"{api_key[:6]}:{model}:{temperature}:{bool(system_prompt)}"
    if st.session_state.get("lc_chain_key") != chain_key:
        st.session_state["lc_memory_chain"] = get_memory_chain(
            api_key=api_key,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt or "You are a helpful assistant.",
        )
        st.session_state["lc_chain_key"] = chain_key
lc_chain = st.session_state.get("lc_memory_chain") if use_lc_memory else None

# -----------------------------
# Header / Toolbar
# -----------------------------
left, mid, right = st.columns([1, 2, 1])
with left:
    st.title("💬 ChatGPT Clone")
with right:
    if st.button("🧹 New Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["feedback"] = {}
        st.session_state["edit_mode"] = {}
        st.session_state.pop("lc_chain_key", None)
        st.session_state.pop("lc_memory_chain", None)
        st.toast("Conversation cleared.")

st.markdown("<div style='margin-top:-10px;'>Powered by OpenAI GPT 🚀</div>", unsafe_allow_html=True)

st.divider()

# -----------------------------
# Render chat history
# -----------------------------
for idx, msg in enumerate(st.session_state.messages):
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    meta = msg.get("meta", {})

    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant":
            # Meta line
            rt = meta.get("response_time")
            mdl = meta.get("model")
            edited = meta.get("edited", False)
            cap_bits = []
            if mdl:
                cap_bits.append(f"🧠 {mdl}")
            if rt is not None:
                cap_bits.append(f"⏱️ {rt:.2f}s")
            if edited:
                cap_bits.append("✏️ edited")
            # tokens & $ if present
            if all(k in meta for k in ("in_tokens", "out_tokens", "est_cost")):
                cap_bits.append(f"🧮 in:{meta['in_tokens']} out:{meta['out_tokens']} · ≈ ${meta['est_cost']:.4f}")
            if cap_bits:
                st.caption(" · ".join(cap_bits))

            # Feedback UI
            fb = st.session_state.feedback.get(idx)
            c1, c2, c3 = st.columns([1, 1, 6])
            with c1:
                if st.button("👍", key=f"up_{idx}"):
                    st.session_state.feedback[idx] = "up"
            with c2:
                if st.button("👎", key=f"down_{idx}"):
                    st.session_state.feedback[idx] = "down"
            with c3:
                if fb:
                    st.caption(f"Feedback recorded: {fb}")

            # Edit UI
            edit_on = st.session_state.edit_mode.get(idx, False)
            toggle = st.button("✏️ Edit", key=f"edit_toggle_{idx}")
            if toggle:
                st.session_state.edit_mode[idx] = not edit_on
                edit_on = not edit_on

            if edit_on:
                new_text = st.text_area(
                    "Edit assistant response:",
                    value=content,
                    key=f"edit_area_{idx}",
                )
                save = st.button("Save", key=f"save_{idx}")
                if save:
                    st.session_state.messages[idx]["content"] = new_text
                    st.session_state.messages[idx].setdefault("meta", {})["edited"] = True
                    st.session_state.edit_mode[idx] = False
                    st.rerun()

# -----------------------------
# Input box
# -----------------------------
prompt = st.chat_input("Type your message here…")

if prompt is not None and not prompt.strip():
    st.warning("Please enter a non-empty message.")

if prompt and prompt.strip():
    if not api_key:
        st.error("OpenAI API key is required. Please enter it in the sidebar.")
    else:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        # Prepare conversation history for backend
        messages_for_llm = build_messages(st.session_state.messages, system_prompt)

            # Generate assistant response burayı düzelttim
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                start = time.time()
                try:
                    if lc_chain is not None:
                        # LangChain memory path
                        response_text = lc_chain.run(prompt)
                        # çıktıyı hemen kullanıcıya yazdır
                        st.markdown(response_text)
                    else:
                        # Direct backend path (stream)
                        def stream_gen() -> Generator[str, None, None]:
                            for part in backend.generate(
                                messages=messages_for_llm,
                                model=model,
                                temperature=temperature,
                                stream=True,
                            ):
                                yield part
                        response_text = st.write_stream(stream_gen()) or ""
                except Exception as e:
                    response_text = f"⚠️ Error: {e}"
                    st.error(response_text)
                    st.stop()
                finally:
                    duration = time.time() - start

                # Estimate tokens & cost
                in_tokens = estimate_tokens(messages_for_llm, model)
                out_tokens = estimate_tokens(
                    [{"role": "assistant", "content": response_text}], model
                )
                est_cost = (in_tokens / 1000.0) * rate_in + (out_tokens / 1000.0) * rate_out
                st.caption(
                    f"⏱️ {duration:.2f}s · 🧠 {model} · 🧮 in:{in_tokens} out:{out_tokens} · ≈ ${est_cost:.4f}"
                )

        

                # Estimate tokens & cost
                in_tokens = estimate_tokens(messages_for_llm, model)
                out_tokens = estimate_tokens([{"role": "assistant", "content": response_text}], model)
                est_cost = (in_tokens / 1000.0) * rate_in + (out_tokens / 1000.0) * rate_out
                st.caption(f"⏱️ {duration:.2f}s · 🧠 {model} · 🧮 in:{in_tokens} out:{out_tokens} · ≈ ${est_cost:.4f}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "meta": {
                    "response_time": duration,
                    "model": model,
                    "in_tokens": in_tokens,
                    "out_tokens": out_tokens,
                    "est_cost": est_cost,
                    "lc_memory": bool(lc_chain is not None),
                },
            }
        )

        # Auto-save snapshot if filename set
        if filename:
            try:
                save_session_to_path(
                    path=filename,
                    messages=st.session_state.get("messages", []),
                    feedback=st.session_state.get("feedback", {}),
                )
                st.toast(f"Session autosaved → {filename}")
            except Exception as e:
                st.warning(f"Autosave failed: {e}")
