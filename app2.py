import streamlit as st
from transformers import MarianMTModel, MarianTokenizer, T5ForConditionalGeneration, T5Tokenizer

# Load Translation Pipeline (Using MarianMT for English-Korean translation)
@st.cache_resource
def load_translation_pipeline(src_lang, tgt_lang):
    # Use the MarianMT model from Hugging Face for translation
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"  # Ensure the model exists on Hugging Face
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Translate Text
def translate_text(text, src_lang, tgt_lang):
    model, tokenizer = load_translation_pipeline(src_lang, tgt_lang)
    # Tokenize the text
    tokenized_text = tokenizer(text, return_tensors="pt", truncation=True)
    # Generate translation
    translated_tokens = model.generate(**tokenized_text)
    # Decode the translated tokens
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Load Summarization Model (T5)
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

# Input Section for Translation
st.header("Translation")
text_to_translate = st.text_area("Enter text to translate:")

# Language selection for translation
src_lang = st.selectbox("Select Source Language", ["en", "ko"])  # English or Korean
tgt_lang = st.selectbox("Select Target Language", ["en", "ko"])  # English or Korean

if st.button("Translate"):
    if text_to_translate:
        # Use the translation function to translate the text
        translation = translate_text(text_to_translate, src_lang, tgt_lang)
        st.subheader("Translated Text:")
        st.write(translation)
    else:
        st.warning("Please enter text to translate.")

# Input Section for Summarization
st.header("Summarization")
text_to_summarize = st.text_area("Enter text to summarize:")

if st.button("Summarize"):
    if text_to_summarize:
        summary = summarize_text(text_to_summarize)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter text to summarize.")
