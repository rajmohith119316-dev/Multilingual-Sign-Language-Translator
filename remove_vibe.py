import os
import re

templates_dir = "c:/Users/mohit/OneDrive/Desktop/MajorProject/Anti_major_1/templates"

def remove_emojis(text):
    # Regex to match emojis
    return re.sub(r'[\U00010000-\U0010ffff]', '', text)

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove emojis
    content = remove_emojis(content)
    
    # 2. Replace em dashes
    content = content.replace('—', '-')
    
    # 3. No purple gradients / No purple
    # We can replace 'linear-gradient(135deg, var(--accent), var(--purple))' with just 'var(--accent)'
    # Or just replace all linear-gradient to flat background if we want to be safe, but they said "no purple gradients".
    content = re.sub(r'linear-gradient\([^)]*var\(--purple\)[^)]*\)', 'var(--blue)', content)
    content = re.sub(r'linear-gradient\([^)]*#6d28d9[^)]*\)', 'var(--blue)', content)
    content = re.sub(r'linear-gradient\([^)]*var\(--pink\)[^)]*\)', 'var(--blue)', content)
    
    # Actually, replacing all purple variables with blue
    content = content.replace('--purple: #7c3aed;', '--purple: #38bdf8;')
    content = content.replace('--purple-dim: #7c3aed20;', '--purple-dim: #38bdf820;')
    content = content.replace('--purple-glow: #7c3aed50;', '--purple-glow: #38bdf850;')
    content = content.replace('rgba(124, 58, 237,', 'rgba(56, 189, 248,')
    content = content.replace('rgba(124,58,237,', 'rgba(56,189,248,')
    
    # 4. Pill-shaped buttons
    content = content.replace('border-radius: 20px;', 'border-radius: 4px;')
    content = content.replace('border-radius: 12px;', 'border-radius: 4px;')
    content = content.replace('border-radius: 24px;', 'border-radius: 4px;')
    # Change root variables
    content = content.replace('--radius-xl: 20px;', '--radius-xl: 4px;')
    content = content.replace('--radius-lg: 16px;', '--radius-lg: 4px;')
    content = content.replace('--radius-md: 12px;', '--radius-md: 4px;')
    content = content.replace('--radius-sm: 8px;', '--radius-sm: 4px;')
    
    # 5. Vague hero texts & Al-slop
    content = content.replace('AI-powered gesture recognition', 'Gesture recognition')
    content = content.replace('AI-powered', '')
    content = content.replace('Al-powered', '')
    
    # 6. Over-the-top animations
    content = content.replace('scroll-behavior: smooth;', '')
    content = re.sub(r'body::before\s*\{[^}]*\}', 'body::before { display: none; }', content)
    content = content.replace('animation: shimmer 4s linear infinite;', '')
    content = content.replace('animation: pulse-dot 2s ease-in-out infinite;', '')
    content = content.replace('animation: pulse-dot 1.5s ease-in-out infinite;', '')
    content = content.replace('animation: pulse-dot 1s ease-in-out infinite;', '')
    content = re.sub(r'@keyframes pulse-dot\s*\{[^}]*\}', '', content)
    content = re.sub(r'@keyframes shimmer\s*\{[^}]*\}', '', content)
    content = re.sub(r'@keyframes fadeIn\s*\{[^}]*\}', '', content)
    content = content.replace('animation: fadeIn 0.4s ease;', '')
    content = content.replace('transition: all var(--transition);', '')
    
    # Also remove "Made with Al" just in case it's slightly different casing
    content = re.sub(r'(?i)made with a[il]', '', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for f in os.listdir(templates_dir):
    if f.endswith('.html'):
        process_file(os.path.join(templates_dir, f))

print("Done processing templates.")
