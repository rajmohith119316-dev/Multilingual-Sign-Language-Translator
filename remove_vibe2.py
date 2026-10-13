import os
import re

templates_dir = "c:/Users/mohit/OneDrive/Desktop/MajorProject/Anti_major_1/templates"

def remove_emojis(text):
    return re.sub(r'[\U00010000-\U0010ffff]', '', text)

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove emojis
    content = remove_emojis(content)
    
    # 2. Replace em dashes
    content = content.replace('—', '-')
    
    # 3. No purple gradients / No purple
    content = re.sub(r'linear-gradient\([^)]*var\(--purple\)[^)]*\)', 'var(--blue)', content)
    content = re.sub(r'linear-gradient\([^)]*#6d28d9[^)]*\)', 'var(--blue)', content)
    content = re.sub(r'linear-gradient\([^)]*var\(--pink\)[^)]*\)', 'var(--blue)', content)
    
    content = content.replace('--purple: #7c3aed;', '--purple: #38bdf8;')
    content = content.replace('--purple-dim: #7c3aed20;', '--purple-dim: #38bdf820;')
    content = content.replace('--purple-glow: #7c3aed50;', '--purple-glow: #38bdf850;')
    content = content.replace('rgba(124, 58, 237,', 'rgba(56, 189, 248,')
    content = content.replace('rgba(124,58,237,', 'rgba(56,189,248,')
    
    # 4. Pill-shaped buttons
    content = content.replace('border-radius: 20px;', 'border-radius: 4px;')
    content = content.replace('border-radius: 12px;', 'border-radius: 4px;')
    content = content.replace('border-radius: 24px;', 'border-radius: 4px;')
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
    
    # Ambient background particles
    content = re.sub(r'body::before\s*\{[^}]*\}', 'body::before { display: none; }', content)
    
    # Remove keyframes and animations by matching blocks carefully
    # Specifically for pulse-dot:
    pulse_dot_regex = r'@keyframes pulse-dot\s*\{.*?\}\s*\}'
    content = re.sub(pulse_dot_regex, '', content, flags=re.DOTALL)
    
    # fadeIn:
    fadein_regex = r'@keyframes fadeIn\s*\{.*?\}\s*\}'
    content = re.sub(fadein_regex, '', content, flags=re.DOTALL)
    
    # shimmer:
    shimmer_regex = r'@keyframes shimmer\s*\{.*?\}\s*\}'
    content = re.sub(shimmer_regex, '', content, flags=re.DOTALL)
    
    # Also remove explicit animation rules
    content = re.sub(r'animation:\s*[^;]+;', '', content)
    
    # Remove hover transitions that could be considered part of the "vibe" or over-the-top
    # But transition: all var(--transition) is probably fine if it's not crazy, but the prompt says "over-the-top scroll animation" and "cursor animation"
    # Actually wait, "animation: shimmer 4s linear infinite;" was captured by the rule above.
    
    # 7. cursor animation
    # "cursor animation" usually implies a custom cursor or something following the mouse.
    # The existing pulse-dot animation is mostly used on the recording dot and status dot.
    # I removed pulse-dot keyframes, so it won't animate anymore.
    
    # 8. fake customer counters
    # The user said "fake reviews or fake metrics" and "fake customer counters"
    # Are there any fake metrics hardcoded in the HTML? I will check manually if this regex doesn't find them.

    # 9. "Made with Al tag"
    content = re.sub(r'(?i)made with a[il]', '', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for f in os.listdir(templates_dir):
    if f.endswith('.html'):
        process_file(os.path.join(templates_dir, f))

print("Done processing templates.")
