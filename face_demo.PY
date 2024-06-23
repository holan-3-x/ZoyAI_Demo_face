import pygame
import sys
import time
import math
from gtts import gTTS
import os
import pygame.mixer

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

# Main loop
running = True
blink = False
blink_time = 0
eye_movement_angle = 0
wave_phase = 0
talking = False

# Start talking
speak("Hello, I am a futuristic robot. How can I assist you today?")
talking = True

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

    # Cap the frame rate
    pygame.time.Clock().tick(30)

# Quit Pygame
pygame.quit()
sys.exit()
