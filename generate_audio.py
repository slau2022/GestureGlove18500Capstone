from gtts import gTTS

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

for letter in letters:
    speech = gTTS(text = letter, lang = "en", slow = False)
    speech.save("audio_files/" + letter + ".mp3")