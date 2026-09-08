import json
import os
import time
import ollama
import pyttsx3
import speech_recognition as sr

# 1. Initialize Engines
tts_engine = pyttsx3.init()
HISTORY_FILE = "voice_history.json"

def speak(text):
    """Converts LLM text output into voice."""
    print(f"\n[AI Response]: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

def save_to_history(question, answer):
    """Appends the Q&A pair with a timestamp to a local JSON file."""
    history_data = []
    
    # Read existing history if it exists
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history_data = json.load(f)
        except json.JSONDecodeError:
            pass  # Handle empty or corrupted file

    # Append new entry
    new_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer
    }
    history_data.append(new_entry)
    
    # Save back to file
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_data, indent=4, ensure_ascii=False)
    print(f"[System]: Interaction saved to {HISTORY_FILE}")

def run_voice_system():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        print("\nAdjusting for background noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("🔴 Listening... Speak your question now.")
        audio = recognizer.listen(source)
        
    try:
        # Convert voice input to text
        user_question = recognizer.recognize_google(audio)
        print(f"[You Said]: {user_question}")
        
        # Guard clause to exit the loop
        if user_question.lower() in ["exit", "quit", "stop"]:
            speak("Goodbye!")
            return False

        print("[System]: Thinking...")
        # Send text to local Ollama instance
        response = ollama.chat(
            model='llama3.2:1b', 
            messages=[{'role': 'user', 'content': user_question}]
        )
        
        ai_answer = response['message']['content']
        
        # Output the answer via voice
        speak(ai_answer)
        
        # Save to permanent history
        save_to_history(user_question, ai_answer)
        
    except sr.UnknownValueError:
        speak("I couldn't hear or understand that clearly. Please try again.")
    except sr.RequestError:
        speak("Speech recognition service is currently unavailable.")
    except Exception as e:
        print(f"Error: {e}")
        speak("An internal error occurred.")
        
    return True

if __name__ == "__main__":
    speak("Voice system initialized. I am listening.")
    is_running = True
    while is_running:
        is_running = run_voice_system()