import os
import re

file_path = "c:/Users/mohit/OneDrive/Desktop/MajorProject/Anti_major_1/src/gui.py"

emojis = ['📷', '🔄', '🔊', '⬜', '🔤', '💬', '🎓', 'ℹ', '🎥', '🧠', '✅', '⚠', '⏹', '▶', '●']

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

for e in emojis:
    content = content.replace(e, '')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Removed emojis from src/gui.py")
