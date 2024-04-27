import pyttsx3

def initialize_speech_engine():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    print("Available Voices:")
    for i, voice in enumerate(voices):
        print(f"{i + 1}. {voice.name}")
    return engine

def speak_text_with_voice(engine, text, voice_index):
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[voice_index].id)
    engine.say(text)
    engine.runAndWait()

def main():
    engine = initialize_speech_engine()
    voice_index = int(input("Select a voice (Enter the voice number): ")) - 1
    text = input("Enter the text you want to hear: ")
    speak_text_with_voice(engine, text, voice_index)
    
main()
