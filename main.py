import streamlit as st
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch

# Load the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Function to predict objects in an image
def predict_object(image, question):
    # Process inputs
    inputs = processor(image, question, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
        answer = processor.decode(generated_ids[0], skip_special_tokens=True)

    return answer

# Streamlit app
st.title("Image Question Answering with BLIP")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Ask for a question
    question = st.text_input("Ask a question about the image:")

    if st.button("Get Answer"):
        if question:
            # Get the answer
            answer = predict_object(image, question)
            st.write(f"**Q:** {question}")
            st.write(f"**A:** {answer}")
        else:
            st.write("Please enter a question.")
