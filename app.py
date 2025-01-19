import os
import streamlit as st
import replicate
import requests
import base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from openai import OpenAI
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables from .env file
load_dotenv()

# Access environment variables
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
IMG_API_KEY = os.getenv("IMG_API_KEY")
FONT_PATH = os.getenv("FONT_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
IMG_UPLOAD_URL = os.getenv("IMG_UPLOAD_URL")
REPLICATE_MODEL = os.getenv("REPLICATE_MODEL")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

st.title("Comic Book Cover Generator")

# User Inputs
title = st.text_input("Enter the comic book title:")
tagline = st.text_input("Enter the comic book tagline:")
style = st.text_input("Enter the style/theme (e.g., Noir, Cyberpunk, etc.):")
uploaded_image = st.file_uploader("Upload an image (required):", type=["jpg", "png"])

# Ensure image upload is mandatory
if not uploaded_image:
    st.warning("You must upload an image to proceed.")

mode = st.radio("Select Mode:", ('Creative', 'Stick to Image'))


# Function to encode image to base64
def encode_image(image):
    return base64.b64encode(image.read()).decode("utf-8")


# Function to upload an image to hosting service
def upload_image_to_hosting_service(image):
    response = requests.post(
        f"{IMG_UPLOAD_URL}?expiration=600&key={IMG_API_KEY}",
        files={"image": (None, str(image))}
    )
    if response.status_code == 200:
        return response.json().get("data", {}).get("url")
    else:
        st.error("Failed to upload image.")
        return None


# Function to generate a Stable Diffusion prompt
def generate_prompt(title, tagline, style, base64_image):
    user_message = (
        f"You are given an image. Use this image as reference. The title of the comic is '{title}', and the tagline is '{tagline}'. "
        f"The style is '{style}'. Generate a Stable Diffusion positive prompt to create a comic book art picture with these details."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": user_message}
        ],
    )

    try:
        return response.choices[0].message.content.strip()
    except (KeyError, IndexError) as e:
        raise ValueError(f"Failed to extract the generated content: {e}")


# Function to generate a comic book cover using Replicate
def generate_comic_cover(prompt, image=None, mode='Creative'):
    replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

    inputs = {
        "model": "dev",
        "width": 1440,
        "height": 1440,
        "prompt": f"{prompt}, SMA, pop art, comic book style, animated, colorful, vibrant, detailed, high quality, high resolution",
        "go_fast": False,
        "lora_scale": 1,
        "megapixels": "1",
        "num_outputs": 1,
        "aspect_ratio": "1:1",
        "output_format": "jpg",
        "guidance_scale": 3.5,
        "output_quality": 90,
        "prompt_strength": 0.7,
        "extra_lora_scale": 1,
        "num_inference_steps": 39
    }

    if mode == 'Stick to Image' and image:
        if not isinstance(image, str):
            raise ValueError("The 'image' parameter must be a base64-encoded string.")
        image_url = upload_image_to_hosting_service(image)
        if not image_url:
            raise ValueError("Failed to generate a valid image URL.")
        inputs["image"] = image_url

    try:
        output = replicate_client.run(REPLICATE_MODEL, input=inputs)
        response = requests.get(output[0])
        if response.status_code == 200:
            return response.content
        else:
            raise Exception("Failed to download the generated image.")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")


# Function to add text to the generated image
def add_text_to_image(image_content, title_text, tagline_text):
    nparr = np.frombuffer(image_content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    title_font = ImageFont.truetype(FONT_PATH, 80)
    tagline_font = ImageFont.truetype(FONT_PATH, 60)

    image_width, image_height = pil_image.size

    def draw_text_with_border(draw, position, text, font, fill, border_fill, border_width):
        x, y = position
        for dx in range(-border_width, border_width + 1):
            for dy in range(-border_width, border_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=border_fill)
        draw.text(position, text, font=font, fill=fill)

    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_x = (image_width - title_bbox[2]) // 2
    title_y = 20

    tagline_bbox = draw.textbbox((0, 0), tagline_text, font=tagline_font)
    tagline_x = (image_width - tagline_bbox[2]) // 2
    tagline_y = title_y + title_bbox[3] + 20

    draw_text_with_border(draw, (title_x, title_y), title_text, title_font, (255, 0, 0), (0, 0, 0), 3)
    draw_text_with_border(draw, (tagline_x, tagline_y), tagline_text, tagline_font, (255, 255, 0), (0, 0, 0), 3)

    final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', final_image)
    return buffer.tobytes()


if st.button("Generate Comic Book Cover"):
    if title and tagline and style and uploaded_image:
        with st.spinner("Generating comic book cover..."):
            base64_image = encode_image(uploaded_image)
            prompt = generate_prompt(title, tagline, style, base64_image)
            image_content = generate_comic_cover(prompt, base64_image, mode)
            if image_content:
                final_image = add_text_to_image(image_content, title, tagline)
                st.download_button(
                    label="Download Comic Book Cover",
                    data=final_image,
                    file_name="comic_book_cover_with_text.jpg",
                    mime="image/jpeg"
                )
    else:
        st.error("Please provide a title, tagline, style, and upload an image.")
