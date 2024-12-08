import streamlit as st
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

# Load Translation Model
def load_translation_model(src_lang, tgt_lang):
    if src_lang == "ko" and tgt_lang == "en":
        model_name = "Helsinki-NLP/opus-mt-ko-en"
    elif src_lang == "en" and tgt_lang == "ko":
        model_name = "Helsinki-NLP/opus-mt-en-ko"
    else:
        raise ValueError("Unsupported language pair!")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Translate Text
def translate_text(text, src_lang, tgt_lang):
    tokenizer, model = load_translation_model(src_lang, tgt_lang)
    tokenized_text = tokenizer(text, return_tensors="pt", truncation=True)
    translated_tokens = model.generate(**tokenized_text)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# Load Summarization Model
@st.cache_resource
def load_summarization_model():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

# Summarize Text
def summarize_text(text):
    tokenizer, model = load_summarization_model()
    input_ids = tokenizer.encode(
        "summarize: " + text, return_tensors="pt", max_length=512, truncation=True
    )
    summary_ids = model.generate(
        input_ids,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit App
st.title("Machine Translation and Summarization App")
st.write(
    "This app provides free text translation and summarization using open-source models."
)

# Input Section
st.header("Translation")
text_to_translate = st.text_area("Enter text to translate:")
src_lang = st.selectbox("Source Language", ['en', 'ko'], index=0)  # English or Korean
tgt_lang = st.selectbox("Target Language", ['en', 'ko'], index=1)  # English or Korean

if st.button("Translate"):
    if text_to_translate:
        try:
            translation = translate_text(text_to_translate, src_lang, tgt_lang)
            st.subheader("Translated Text:")
            st.write(translation)
        except ValueError as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter text to translate.")

st.header("Summarization")
text_to_summarize = st.text_area("Enter text to summarize:")

if st.button("Summarize"):
    if text_to_summarize:
        summary = summarize_text(text_to_summarize)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter text to summarize.")
