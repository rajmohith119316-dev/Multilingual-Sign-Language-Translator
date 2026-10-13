import os

templates_dir = "c:/Users/mohit/OneDrive/Desktop/MajorProject/Anti_major_1/templates"

emojis = ['⬜', '⬛', '⚡', '⚙', '⚙️', '✅', '🎬', '💡', '🎥', '🔄', '🖐', '🖐️', '📖', '📭', '📷', '🧠', '🔍', '🎓', '✋', '📊', '🌍', '🌍', '🌍']

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    for e in emojis:
        content = content.replace(e, '')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for f in os.listdir(templates_dir):
    if f.endswith('.html'):
        process_file(os.path.join(templates_dir, f))

print("Done processing emojis.")
