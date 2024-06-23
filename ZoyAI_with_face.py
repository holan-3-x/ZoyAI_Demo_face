import pygame
import sys
import time
import math
import os
import speech_recognition as sr
from gtts import gTTS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import chromadb
import argparse

# Initialize Pygame and mixer
pygame.init()
pygame.mixer.init()

# Set the dimensions of the window
size = width, height = 800, 600
screen = pygame.display.set_mode(size)

# Set the title of the window
pygame.display.set_caption("Sci-Fi Animated Eyes and Talking Lips")

# Define colors
black = (0, 0, 0)
dark_blue = (10, 10, 30)
light_blue = (100, 149, 237)
neon_blue = (0, 255, 255)
neon_purple = (147, 112, 219)

# Define the positions and sizes of facial features
eye_radius = 50
eye_offset_x = 200
eye_offset_y = 150
pupil_radius = 20
wave_amplitude = 20
wave_frequency = 0.1
wave_speed = 5

# Text to speech using gTTS
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("speech.mp3")
    pygame.mixer.music.load("speech.mp3")
    pygame.mixer.music.play()

# Voice recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        print(f"Recognized: {query}")
        return query
    except sr.UnknownValueError:
        print("Could not understand the audio")
        return ""
    except sr.RequestError:
        print("Could not request results; check your network connection")
        return ""

# Main function for the AI interaction
def ai_interaction(query):
    embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
    persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
    target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = Ollama(
        base_url='http://localhost:11434',
        model="tinyllama",
        callbacks=callbacks
    )

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    res = qa(query)
    answer = res['result']
    return answer

# Main loop
def main():
    running = True
    blink = False
    blink_time = 0
    eye_movement_angle = 0
    wave_phase = 0
    talking = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the background
        screen.fill(dark_blue)

        # Calculate eye movement
        eye_movement_angle += 0.05
        eye_movement_x = int(math.cos(eye_movement_angle) * 10)
        eye_movement_y = int(math.sin(eye_movement_angle) * 10)

        # Draw eyes
        eye_left_center = (width // 2 - eye_offset_x, height // 2 - eye_offset_y)
        eye_right_center = (width // 2 + eye_offset_x, height // 2 - eye_offset_y)

        if blink:
            pygame.draw.line(screen, neon_blue, (eye_left_center[0] - eye_radius, eye_left_center[1]), (eye_left_center[0] + eye_radius, eye_left_center[1]), 10)
            pygame.draw.line(screen, neon_blue, (eye_right_center[0] - eye_radius, eye_right_center[1]), (eye_right_center[0] + eye_radius, eye_right_center[1]), 10)
        else:
            pygame.draw.circle(screen, neon_blue, (eye_left_center[0] + eye_movement_x, eye_left_center[1] + eye_movement_y), eye_radius, 2)
            pygame.draw.circle(screen, neon_blue, (eye_right_center[0] + eye_movement_x, eye_right_center[1] + eye_movement_y), eye_radius, 2)
            pygame.draw.circle(screen, neon_purple, (eye_left_center[0] + eye_movement_x, eye_left_center[1] + eye_movement_y), pupil_radius)
            pygame.draw.circle(screen, neon_purple, (eye_right_center[0] + eye_movement_x, eye_right_center[1] + eye_movement_y), pupil_radius)

        # Draw wave-like lips when talking
        if talking:
            wave_phase += wave_speed
            for x in range(-wave_amplitude * 2, wave_amplitude * 2 + 1):
                y_offset = int(math.sin(wave_phase * wave_frequency + x * wave_frequency) * wave_amplitude)
                lip_x = width // 2 + x * 10
                lip_y = height // 2 + eye_offset_y + y_offset
                pygame.draw.circle(screen, neon_blue, (lip_x, lip_y), 5)

        # Update the display
        pygame.display.flip()

        # Blink logic
        if not blink and time.time() - blink_time > 2:  # Blink every 2 seconds
            blink = True
            blink_time = time.time()
        elif blink and time.time() - blink_time > 0.1:
            blink = False
            blink_time = time.time()

        # Check if talking has finished
        if not pygame.mixer.music.get_busy():
            talking = False

        # Check for voice input
        if not talking and not pygame.mixer.music.get_busy():
            query = recognize_speech()
            if query:
                answer = ai_interaction(query)
                speak(answer)
                talking = True

        # Cap the frame rate
        pygame.time.Clock().tick(30)

    # Quit Pygame
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
