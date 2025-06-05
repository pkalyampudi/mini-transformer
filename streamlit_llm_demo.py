# streamlit_llm_demo.py
# A simple transformer attention demo using Streamlit

import streamlit as st
import torch
import torch.nn.functional as F

st.set_page_config(page_title="LLM Transformer Demo", layout="centered")
st.title("🔍 Simple Transformer Demo")

st.write("This is a simplified demo of how attention works in models like ChatGPT.")

st.header("Step 1: Enter Your Words")
words_input = st.text_input("Enter 3 words separated by commas (e.g., hello, world, AI):", "hello, world, AI")

words = [w.strip() for w in words_input.split(",")][:3]
if len(words) < 3:
    st.warning("Please enter at least 3 words.")
    st.stop()

# Generate random word embeddings (2D vectors)
st.header("Step 2: Word Embeddings")
embedding_dict = {word: torch.randn(2) for word in words}
for word, vec in embedding_dict.items():
    st.write(f"{word}: {vec.numpy()}")

# Stack into tensor for processing
W = torch.stack(list(embedding_dict.values()))

st.header("Step 3: Self-Attention Calculation")
Q, K, V = W, W, W

def self_attention(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scaled_scores = scores / Q.size(-1)**0.5
    attention_weights = F.softmax(scaled_scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

output, attn = self_attention(Q, K, V)

st.subheader("Attention Weights (Matrix)")
st.write(attn.detach().numpy())

st.subheader("Output After Attention")
for i, word in enumerate(words):
    st.write(f"{word}: {output[i].detach().numpy()}")

st.markdown("---")
st.markdown("🧠 This is how a transformer understands the relationship between words using attention!")
