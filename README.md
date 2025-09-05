# AI-Image-Caption-Generator-
This app uses a powerful pre-trained multimodal AI model from Hugging Face Transformers to generate descriptive captions for any image you upload. It's a great example of combining computer vision and natural language processing.
# 🖼️ AI Image Caption Generator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app/) This little app is like giving your computer a pair of eyes and a voice. Just show it a picture, and it will tell you what it sees! It uses a powerful AI model to generate descriptive, human-like captions for any image you upload.

See the world through the eyes of an AI!



---

## ✨ Features

-   **Easy Image Upload**: Supports JPG, JPEG, and PNG formats.
-   **Instant Captions**: Get a descriptive caption for your image in seconds.
-   **State-of-the-Art AI**: Powered by the Salesforce BLIP model, a leader in vision-language tasks.
-   **Minimalist UI**: Clean and simple interface built with Streamlit.

---

## 🛠️ Tech Stack

-   **Language**: Python
-   **Framework**: Streamlit
-   **Core AI**: Hugging Face Transformers
-   **Deep Learning**: PyTorch
-   **Image Handling**: Pillow

---

## 🚀 How to Run It Locally

Want to try it yourself? It's super easy to get started.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/image_captioning.git](https://github.com/your-username/image_captioning.git)
    cd image_captioning
    ```

2.  **Set up a Python environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The first time you run it, the app will download the AI model (about 1.7GB), so it might take a few minutes depending on your internet connection. After that, it's cached!*

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

Now you can upload your own images and see what the AI thinks!
