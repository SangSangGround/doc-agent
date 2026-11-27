import pyttsx3

engine = pyttsx3.init()

for voice in engine.getProperty("voices"):
    if "korean" in voice.name.lower() or "heami" in voice.name.lower():
        engine.setProperty("voice", voice.id)
        break

engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)

engine.say("오른쪽으로 3센티미터 이동하세요.")
engine.runAndWait()