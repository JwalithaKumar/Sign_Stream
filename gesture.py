import streamlit as st
import cv2
import mediapipe as mp
from PIL import Image

# Load MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to process the image and detect hands
def detect_hands(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    return results

# Function to draw landmarks on the image
def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                height, width, _ = image.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

# Streamlit app
def main():
    st.title("Hand Tracking App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to OpenCV format
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detect hands
        results = detect_hands(image_np)

        # Draw landmarks on the image
        draw_landmarks(image_np, results)

        # Display the processed image with landmarks
        st.image(image_np, caption="Processed Image", use_column_width=True)

if __name__ == "__main__":
    main()

