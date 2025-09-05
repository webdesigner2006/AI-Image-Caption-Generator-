import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- App Configuration ---
st.set_page_config(page_title="🖼️ AI Image Caption Generator", layout="centered")
st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image and let the AI describe it for you!")

# --- Model Loading ---
# We use st.cache_resource to load the model only once, improving performance.
@st.cache_resource
def load_model():
    """Loads the BLIP image captioning model and processor from Hugging Face."""
    model_id = "Salesforce/blip-image-captioning-large"
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    
    return model, processor, device

# Load the model and processor
model, processor, device = load_model()

# --- Image Captioning Function ---
def generate_caption(image_file):
    """Generates a caption for the given image file."""
    try:
        # Open the image file
        raw_image = Image.open(image_file).convert('RGB')
        
        # Preprocess the image
        inputs = processor(raw_image, return_tensors="pt").to(device)
        
        # Generate the caption
        with torch.no_grad(): # Disable gradient calculation for inference
            out = model.generate(**inputs, max_new_tokens=50) # Increased max_new_tokens
            
        # Decode the generated ids to text
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption, raw_image
        
    except Exception as e:
        return f"An error occurred: {e}", None

# --- Streamlit UI ---
st.sidebar.header("About")
st.sidebar.info(
    "This app uses the 'BLIP' model from Salesforce Research, a state-of-the-art "
    "model for image captioning. Simply upload your image to see it in action."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.subheader("Your Image:")
    st.image(uploaded_file, use_column_width=True)
    
    with st.spinner("AI is thinking... 🧠"):
        caption, _ = generate_caption(uploaded_file)
        
        st.subheader("Generated Caption:")
        st.markdown(f"**{caption.capitalize()}**")

else:
    st.info("Upload an image to get started.")
