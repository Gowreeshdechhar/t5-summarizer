
import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model and tokenizer once
@st.cache_resource
def load_model():
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return model, tokenizer

model, tokenizer = load_model()

# UI
st.set_page_config(page_title="T5 Summarizer")
st.title("📝 T5 Text Summarizer")
st.write("Enter a paragraph and click the button to summarize.")

text = st.text_area("Enter your text:", height=300)

if st.button("Summarize"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Summarizing..."):
            input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
            output = model.generate(input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(output[0], skip_special_tokens=True)
            st.success("Here is your summary:")
            st.write(summary)
