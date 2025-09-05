import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Constants ---
# Using Salesforce's BLIP as it's a popular and effective model for this task.
MODEL_ID = "Salesforce/blip-image-captioning-large"

# --- App Configuration ---
st.set_page_config(page_title="🖼️ AI Image Captioner", layout="centered")
st.title("🖼️ AI Image Caption Generator")
st.write("Give me an image, and I'll tell you what I see!")

# --- Model Loading ---
@st.cache_resource
def get_model_and_processor():
    """
    Load the BLIP model and processor from Hugging Face.
    We cache this to avoid reloading it every time the app reruns.
    """
    print("Loading BLIP model...") # Debugging print
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    
    print("Model loaded successfully on:", device)
    return model, processor, device

# A small helper function, common in human-written code to keep the main block clean.
def display_results(image, caption):
    st.subheader("Your Image:")
    st.image(image, use_column_width=True)
    
    st.subheader("AI-Generated Caption:")
    st.markdown(f"**📝 {caption.capitalize()}**")

# --- Main Logic ---
model, processor, device = get_model_and_processor()

st.sidebar.header("How It Works")
st.sidebar.info(
    "This app uses a pre-trained AI model called BLIP (Bootstrapping Language-Image Pre-training) "
    "to analyze the content of an image and generate a text description."
)

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        raw_image = Image.open(uploaded_file).convert('RGB')
        
        with st.spinner("AI is thinking... 🧠"):
            # Preprocess the image
            inputs = processor(raw_image, return_tensors="pt").to(device)
            
            # Generate the caption
            # No need for torch.no_grad() here, inference mode is handled by HF.
            out = model.generate(**inputs, max_new_tokens=50)
            
            # Decode the result
            caption = processor.decode(out[0], skip_special_tokens=True)
            
        display_results(uploaded_file, caption)
        
    except Exception as e:
        st.error(f"Oops, something went wrong. Please try another image. Error: {e}")

else:
    st.info("Please upload an image to begin.")
