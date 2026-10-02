"""
generate_presentation_files.py
Generates updated high-resolution diagrams matching the codebase,
creates a PowerPoint presentation (.pptx), and creates a PDF presentation (.pdf).
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ---------------------------------------------------------
# DIRECTORY SETUP
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "presentation_assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

PPTX_PATH = os.path.join(BASE_DIR, "Multilingual_Sign_Language_Translator_Presentation.pptx")
PDF_PATH  = os.path.join(BASE_DIR, "Multilingual_Sign_Language_Translator_Presentation.pdf")

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# ---------------------------------------------------------
# 1. GENERATE SYSTEM DESIGN DIAGRAM
# ---------------------------------------------------------
def generate_system_design_diagram():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    # Banner
    title_box = patches.FancyBboxPatch((0.5, 8.2), 15.0, 0.65, boxstyle="round,pad=0.1", 
                                        fc='#0A192F', ec='#1E3A8A', lw=2)
    ax.add_patch(title_box)
    ax.text(8.0, 8.52, "SYSTEM DESIGN – MULTILINGUAL REAL-TIME SIGN LANGUAGE TRANSLATOR", 
            color='white', weight='bold', fontsize=13, ha='center', va='center')

    # User Block
    user_box = patches.FancyBboxPatch((0.4, 4.0), 1.6, 2.2, boxstyle="round,pad=0.1", fc='#E2E8F0', ec='#475569', lw=2)
    ax.add_patch(user_box)
    ax.text(1.2, 5.4, "[USER]\nSigner", fontsize=14, weight='bold', color='#1E293B', ha='center', va='center')
    ax.text(1.2, 4.4, "Live Video\nStream", fontsize=9.5, ha='center', va='center', color='#334155', weight='bold')

    # Process blocks
    blocks = [
        ("1.0 Video Capture", "Webcam Frame\nCapture (OpenCV)", 2.5, 6.7, '#DBEAFE', '#1D4ED8'),
        ("2.0 Hand Tracking & Router", "MediaPipe HandLandmarker\nRoute by 0/1/2 Hands", 5.8, 6.7, '#DBEAFE', '#1D4ED8'),
        ("3.0 Feature Extraction", "Single-Hand: 60-D Vector\nDual-Hand: 123-D Vector", 9.1, 6.7, '#DBEAFE', '#1D4ED8'),
        ("4.0 MoE Recognition", "Alphabet Expert (Dense MLP)\nWord Expert (Conv1D + GRU)", 12.4, 6.7, '#FEF08A', '#A16207'),
        ("5.0 Stability & Text Filter", "5-Frame Stability Filter\nPrevent Flicker Output", 12.4, 4.0, '#DCFCE7', '#15803D'),
        ("6.0 Translation & TTS", "deep-translator API\npyttsx3 Speech Engine", 9.1, 4.0, '#E0E7FF', '#4338CA'),
        ("7.0 User Interface (GUI)", "Tkinter Desktop GUI\nOverlay & Audio Output", 5.8, 4.0, '#FCE7F3', '#BE185D'),
    ]

    for title, desc, x, y, bg, border in blocks:
        box = patches.FancyBboxPatch((x, y), 2.7, 1.3, boxstyle="round,pad=0.1", fc=bg, ec=border, lw=2)
        ax.add_patch(box)
        ax.text(x+1.35, y+0.92, title, fontsize=9.5, weight='bold', color=border, ha='center', va='center')
        ax.text(x+1.35, y+0.42, desc, fontsize=8.5, color='#1E293B', ha='center', va='center')

    # Database Box
    db_box = patches.FancyBboxPatch((2.5, 0.8), 6.0, 2.5, boxstyle="round,pad=0.1", fc='#FEF3C7', ec='#D97706', lw=2)
    ax.add_patch(db_box)
    ax.text(5.5, 3.0, "[DATABASE] SQLite Database (sign_language_app.db)", fontsize=10.5, weight='bold', color='#B45309', ha='center', va='center')
    db_text = (
        "• gestures (gesture_id, gesture_name, num_hands)\n"
        "• recognition_history (history_id, text, confidence, timestamp)\n"
        "• translation_cache (cache_id, source, target, translation)\n"
        "• continual_learning_logs (log_id, gesture, samples, timestamp)"
    )
    ax.text(2.7, 1.8, db_text, fontsize=8.0, color='#78350F', ha='left', va='center')

    # Continual Learning Box
    cl_box = patches.FancyBboxPatch((9.1, 0.8), 6.0, 2.5, boxstyle="round,pad=0.1", fc='#F3E8FF', ec='#7E22CE', lw=2)
    ax.add_patch(cl_box)
    ax.text(12.1, 3.0, "[LEARNING] Continual Learning Pipeline", fontsize=10.5, weight='bold', color='#6B21A8', ha='center', va='center')
    cl_text = (
        "1. Record 10 dynamic gesture samples (Teach Tab)\n"
        "2. Save (30, 123) sequence matrices to dataset\n"
        "3. Freeze Conv1D+GRU backbone & replace dense head\n"
        "4. Fine-tune with experience replay & Hot-Reload live"
    )
    ax.text(9.3, 1.8, cl_text, fontsize=8.0, color='#581C87', ha='left', va='center')

    # Connecting Arrows
    arrow = dict(arrowstyle="->", lw=2, color='#334155')
    ax.annotate("", xy=(2.5, 7.35), xytext=(2.0, 5.8), arrowprops=arrow)
    ax.annotate("", xy=(5.8, 7.35), xytext=(5.2, 7.35), arrowprops=arrow)
    ax.annotate("", xy=(9.1, 7.35), xytext=(8.5, 7.35), arrowprops=arrow)
    ax.annotate("", xy=(12.4, 7.35), xytext=(11.8, 7.35), arrowprops=arrow)
    ax.annotate("", xy=(13.75, 5.3), xytext=(13.75, 6.7), arrowprops=arrow)
    ax.annotate("", xy=(11.8, 4.65), xytext=(12.4, 4.65), arrowprops=arrow)
    ax.annotate("", xy=(8.5, 4.65), xytext=(9.1, 4.65), arrowprops=arrow)
    ax.annotate("", xy=(2.0, 4.4), xytext=(5.8, 4.65), arrowprops=dict(arrowstyle="->", lw=2, color='#334155', connectionstyle="arc3,rad=0.35"))

    ax.annotate("", xy=(5.5, 4.0), xytext=(5.5, 3.3), arrowprops=dict(arrowstyle="<->", lw=1.5, color='#D97706', ls='--'))
    ax.annotate("", xy=(10.5, 4.0), xytext=(10.5, 3.3), arrowprops=dict(arrowstyle="<->", lw=1.5, color='#7E22CE', ls='--'))

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    path = os.path.join(ASSETS_DIR, "system_design_diagram.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path

# ---------------------------------------------------------
# 2. GENERATE SYSTEM ARCHITECTURE DIAGRAM
# ---------------------------------------------------------
def generate_system_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Title
    ax.text(8.0, 8.6, "SYSTEM ARCHITECTURE DIAGRAM", fontsize=15, weight='bold', color='#0F172A', ha='center')
    ax.text(8.0, 8.25, "Multilingual Real-Time Sign Language Recognition and Translation System (MoE Edition)", 
            fontsize=10.5, color='#475569', ha='center')

    # 9 Modular Columns / Blocks
    stages = [
        ("1. INPUT LAYER", "Webcam / Camera\nVideo Stream\n(OpenCV)", 0.4, 4.8, 1.4, 3.0, '#EFF6FF', '#3B82F6'),
        ("2. HAND TRACKING & ROUTER", "MediaPipe HandLandmarker\n───────────────────\n0 Hands -> Idle Mode\n1 Hand -> Alphabet Mode\n2 Hands -> Word Mode", 2.0, 4.8, 1.6, 3.0, '#F0FDF4', '#22C55E'),
        ("3. FEATURE EXTRACTION", "Single Hand: (60,)\nWrist-relative + L2-norm\n───────────────────\nDual Hand: (123,)\nLeft60 + Right60 + Δwrist3\nBuffer: Deque(30 frames)", 3.8, 4.8, 1.7, 3.0, '#FEFCE8', '#EAB308'),
        ("4. GESTURE RECOGNITION", "Alphabet Expert:\nDense MLP (A–Z)\n───────────────────\nWord Expert:\nConv1D + GRU (Words)", 5.7, 4.8, 1.6, 3.0, '#FFF7ED', '#F97316'),
        ("5. STABILITY & SENTENCE", "5-Frame Stability Filter\n───────────────────\nAlphabet / Word\nSequence Assembly", 7.5, 4.8, 1.5, 3.0, '#FAF5FF', '#A855F7'),
        ("6. TRANSLATION MODULE", "deep-translator Engine\n───────────────────\nLanguages: Kannada, Hindi,\nTamil, Telugu, English, etc.\nSQLite 0ms Cache Lookup", 9.2, 4.8, 1.6, 3.0, '#EEF2FF', '#6366F1'),
        ("7. USER INTERFACE (GUI)", "Tkinter 3-Tab GUI\n• Real-Time Translator\n• Teach Custom Sign\n• History & Logs\nThreaded pyttsx3 TTS", 11.0, 4.8, 1.5, 3.0, '#FDF2F8', '#EC4899'),
        ("8. DATABASE", "SQLite DB\n(sign_language_app.db)\n• Gestures Table\n• Recognition History\n• Translation Cache", 12.7, 4.8, 1.4, 3.0, '#FEF3C7', '#D97706'),
        ("9. CONTINUAL LEARNING", "Dynamic Teach Session\nRecord 10x30 frames\nFreeze Conv1D+GRU\nFine-Tune & Hot-Reload", 14.3, 4.8, 1.4, 3.0, '#F3E8FF', '#9333EA'),
    ]

    for title, text, x, y, w, h, bg, border in stages:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=bg, ec=border, lw=1.8)
        ax.add_patch(box)
        ax.text(x+w/2, y+h-0.25, title, fontsize=7.5, weight='bold', color=border, ha='center', va='center')
        ax.text(x+w/2, y+h/2-0.2, text, fontsize=7.0, color='#1E293B', ha='center', va='center')

    # Horizontal Flow Arrows
    for i in range(len(stages)-1):
        x1 = stages[i][2] + stages[i][4]
        x2 = stages[i+1][2]
        y_center = 6.3
        ax.annotate("", xy=(x2, y_center), xytext=(x1, y_center),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color='#475569'))

    # Bottom summary box
    summary_box = patches.FancyBboxPatch((0.4, 0.8), 15.3, 3.2, boxstyle="round,pad=0.1", fc='#0F172A', ec='#334155', lw=2)
    ax.add_patch(summary_box)
    ax.text(8.0, 3.6, "SUMMARY OF SYSTEM ARCHITECTURE & CODEBASE MODULES", fontsize=11, weight='bold', color='#38BDF8', ha='center')
    
    summary_text = (
        "• Webcam / OpenCV Captures Frames -> MediaPipe Tasks extracts 21 3D landmarks per hand.\n"
        "• MoE Router dispatches 1-hand input to Alphabet Expert (Dense MLP, 60-D features) and 2-hand input to Word Expert (Conv1D + GRU, 123-D x 30 frames).\n"
        "• 5-Frame Stability Filter ensures smooth, flicker-free gesture recognition before trigger.\n"
        "• Translation Module uses deep-translator with SQLite Cache (0ms latency for repeated phrases) & pyttsx3 for offline audio TTS.\n"
        "• Tkinter Desktop Application handles real-time visual overlay, translation selection, history viewing, and dynamic continual learning."
    )
    ax.text(0.7, 2.2, summary_text, fontsize=9.0, color='#F8FAFC', ha='left', va='center')

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    path = os.path.join(ASSETS_DIR, "system_architecture_diagram.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path

# ---------------------------------------------------------
# 3. GENERATE METHODOLOGY & IMPLEMENTATION DIAGRAM
# ---------------------------------------------------------
def generate_methodology_diagram():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    # Title
    ax.text(8.0, 8.5, "Methodology & Implementation Pipeline", fontsize=16, weight='bold', color='#0A192F', ha='center')
    ax.text(8.0, 8.15, "Multilingual Real-Time Sign Language Translator — Mixture-of-Experts (MoE) Architecture", 
            fontsize=11, color='#475569', ha='center')

    # Top pipeline blocks (1 to 7)
    steps = [
        ("1. WEBCAM CAPTURE", "Capture 30 FPS video\nfeed via OpenCV", 0.5, 5.8, 1.9, 1.8, '#E0F2FE', '#0284C7'),
        ("2. HAND TRACKING", "MediaPipe HandLandmarker\nDetect 0 / 1 / 2 Hands", 2.7, 5.8, 1.9, 1.8, '#DCFCE7', '#16A34A'),
        ("3. MOE ROUTING", "1 Hand -> 60-D (Alphabet)\n2 Hands -> 123-D (Word)", 4.9, 5.8, 1.9, 1.8, '#FEF08A', '#CA8A04'),
        ("4. EXPERT MODELS", "Alphabet: Dense MLP\nWord: Conv1D + GRU", 7.1, 5.8, 1.9, 1.8, '#FFEDD5', '#EA580C'),
        ("5. STABILITY FILTER", "5-Frame consecutive\nstability check", 9.3, 5.8, 1.9, 1.8, '#F3E8FF', '#9333EA'),
        ("6. TRANSLATION & TTS", "deep-translator API\n+ SQLite 0ms Cache & pyttsx3", 11.5, 5.8, 1.9, 1.8, '#E0E7FF', '#4F46E5'),
        ("7. TKINTER GUI & DB", "Real-Time Display & SQLite\nHistory/Logs Persistence", 13.7, 5.8, 1.8, 1.8, '#FCE7F3', '#DB2777'),
    ]

    for title, desc, x, y, w, h, bg, border in steps:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=bg, ec=border, lw=2)
        ax.add_patch(box)
        ax.text(x+w/2, y+h-0.35, title, fontsize=8.5, weight='bold', color=border, ha='center', va='center')
        ax.text(x+w/2, y+h/2-0.2, desc, fontsize=7.5, color='#1E293B', ha='center', va='center')

    # Horizontal arrows
    for i in range(len(steps)-1):
        x1 = steps[i][2] + steps[i][4]
        x2 = steps[i+1][2]
        ax.annotate("", xy=(x2, 6.7), xytext=(x1, 6.7),
                    arrowprops=dict(arrowstyle="->", lw=2, color='#334155'))

    # Bottom 3 key phase boxes
    phases = [
        ("1 - DATA COLLECTION & MOE FEATURE ENGINEERING", 
         "• ASL Alphabet Dataset (Static images) -> 60-D wrist-relative normalized feature vectors.\n"
         "• WLASL Dataset (Dynamic videos) -> 123-D dual-hand feature vectors x 30 sequence frames.\n"
         "• Wrist-relative position normalization ensures scale, distance, and rotation invariance.", 
         0.5, 1.2, 4.8, 3.8, '#F1F5F9', '#334155'),

        ("2 - HYBRID DEEP LEARNING & TRANSLATION PIPELINE", 
         "• Alphabet Expert (Dense MLP) delivers ultra-low latency recognition for static A-Z letters.\n"
         "• Word Expert (Conv1D + GRU) captures spatio-temporal features for dynamic sign words.\n"
         "• 5-Frame Stability Filter eliminates prediction jitter before triggering text & audio output.\n"
         "• Multilingual Translation via deep-translator with SQLite cache lookup (0ms repeat latency).", 
         5.6, 1.2, 4.8, 3.8, '#F1F5F9', '#334155'),

        ("3 - CONTINUAL LEARNING & LOCAL PERSISTENCE", 
         "• Teach Custom Sign: Record 10 dynamic gesture samples (30 frames each) live in the GUI.\n"
         "• Backbone Freezing: Conv1D + GRU feature extractor is frozen; new dense classification head is trained.\n"
         "• Experience Replay: Blends existing classes during fine-tuning to prevent catastrophic forgetting.\n"
         "• Hot-Reloading: Model reloaded into active memory seamlessly without restarting app.", 
         10.7, 1.2, 4.8, 3.8, '#F1F5F9', '#334155'),
    ]

    for title, text, x, y, w, h, bg, border in phases:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=bg, ec=border, lw=2)
        ax.add_patch(box)
        ax.text(x+w/2, y+h-0.35, title, fontsize=8.5, weight='bold', color='#0F172A', ha='center', va='center')
        ax.text(x+0.2, y+h/2-0.25, text, fontsize=7.5, color='#334155', ha='left', va='center')

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    path = os.path.join(ASSETS_DIR, "methodology_implementation_diagram.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path

# ---------------------------------------------------------
# BUILD PPTX PRESENTATION
# ---------------------------------------------------------
def create_pptx_presentation(design_img, arch_img, meth_img):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_bg(slide, color_rgb):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color_rgb

    # ------------------ SLIDE 1: Title Slide ------------------
    slide1 = prs.slides.add_slide(blank_layout)
    set_bg(slide1, RGBColor(10, 25, 47)) # Dark navy

    # Title
    txBox = slide1.shapes.add_textbox(Inches(0.8), Inches(1.2), Inches(11.7), Inches(1.8))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Topic: Multilingual Real-Time Sign Language Translator"
    p.font.bold = True
    p.font.size = Pt(36)
    p.font.color.rgb = RGBColor(255, 255, 255)
    
    p2 = tf.add_paragraph()
    p2.text = "Mixture-of-Experts (MoE) Architecture for Real-Time Gesture Recognition & Translation"
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(56, 189, 248)

    # Guidance & Team info
    info_box = slide1.shapes.add_textbox(Inches(0.8), Inches(3.5), Inches(11.7), Inches(3.2))
    tf_info = info_box.text_frame
    tf_info.word_wrap = True

    p = tf_info.paragraphs[0]
    p.text = "Under the guidance of:  Dr. Prabha Seetaram Naik (Associate Professor & HOD, Dept. of CSE(DS))"
    p.font.size = Pt(16)
    p.font.color.rgb = RGBColor(241, 245, 249)

    p = tf_info.add_paragraph()
    p.text = "Under the Co-guidance of:  Prof. Bhagya M (Assistant Professor)"
    p.font.size = Pt(16)
    p.font.color.rgb = RGBColor(241, 245, 249)

    p = tf_info.add_paragraph()
    p.text = "\nTeam Members (Dept. of Computer Science and Engineering, CITNC):"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = RGBColor(56, 189, 248)

    members = [
        "• Lekhana N (1AJ23CS069)",
        "• Mohith Raj N (1AJ23CS082)",
        "• Pragathi N (1AJ23CS105)",
        "• Shohith Kumar K (1AJ23CS138)"
    ]
    for m in members:
        p = tf_info.add_paragraph()
        p.text = "   " + m
        p.font.size = Pt(14)
        p.font.color.rgb = RGBColor(226, 232, 240)

    # ------------------ SLIDE 2: Contents ------------------
    slide2 = prs.slides.add_slide(blank_layout)
    set_bg(slide2, RGBColor(248, 250, 252))

    txBox = slide2.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.3), Inches(1.0))
    p = txBox.text_frame.paragraphs[0]
    p.text = "Contents"
    p.font.bold = True
    p.font.size = Pt(36)
    p.font.color.rgb = RGBColor(10, 25, 47)

    txBox = slide2.shapes.add_textbox(Inches(1.5), Inches(2.2), Inches(10.0), Inches(4.5))
    tf = txBox.text_frame
    items = [
        "❖  Previous Review Feedback",
        "❖  System Design (Data Flow Diagram & SQLite Data Schema)",
        "❖  System Architecture (Modular Mixture-of-Experts Architecture)",
        "❖  Methodology & Implementation Pipeline (8-Step Execution & Continual Learning)"
    ]
    for item in items:
        p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(22)
        p.font.color.rgb = RGBColor(30, 41, 59)
        p.space_after = Pt(28)

    # ------------------ SLIDE 3: Previous Review Feedback ------------------
    slide3 = prs.slides.add_slide(blank_layout)
    set_bg(slide3, RGBColor(248, 250, 252))

    txBox = slide3.shapes.add_textbox(Inches(1.0), Inches(0.6), Inches(11.3), Inches(0.8))
    p = txBox.text_frame.paragraphs[0]
    p.text = "Previous Review Feedback & Project Solution"
    p.font.bold = True
    p.font.size = Pt(30)
    p.font.color.rgb = RGBColor(10, 25, 47)

    txBox = slide3.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(5.5))
    tf = txBox.text_frame
    tf.word_wrap = True

    sections = [
        ("Title:", "Multilingual Real-Time Sign Language Translator"),
        ("Problem Definition:", "Communication between sign language users and non-signers is difficult due to language barriers and limited real-time translation tools."),
        ("Proposed Solution:", "Develop a real-time system utilizing a Hierarchical Mixture-of-Experts (MoE) deep learning architecture that recognizes static letters (A-Z) and dynamic word gestures, translating them into text and speech in multiple languages."),
        ("Objectives:", "• Recognize sign language gestures accurately in real time with an MoE router (0/1/2 hands).\n• Translate recognized signs into multiple target languages (Kannada, Hindi, Tamil, Telugu, etc.) with 0ms cache lookup.\n• Provide synchronized text overlay and audio speech (TTS) output.\n• Support live continual learning to extend the dictionary with custom gestures without full retraining.")
    ]

    for title, body in sections:
        p = tf.add_paragraph()
        p.text = title
        p.font.bold = True
        p.font.size = Pt(18)
        p.font.color.rgb = RGBColor(29, 78, 216)
        
        p2 = tf.add_paragraph()
        p2.text = body
        p2.font.size = Pt(15)
        p2.font.color.rgb = RGBColor(30, 41, 59)
        p2.space_after = Pt(14)

    # ------------------ SLIDE 4: System Design ------------------
    slide4 = prs.slides.add_slide(blank_layout)
    set_bg(slide4, RGBColor(248, 250, 252))

    txBox = slide4.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.6))
    p = txBox.text_frame.paragraphs[0]
    p.text = "System Design"
    p.font.bold = True
    p.font.size = Pt(28)
    p.font.color.rgb = RGBColor(10, 25, 47)

    slide4.shapes.add_picture(design_img, Inches(0.5), Inches(1.1), width=Inches(12.33))

    # ------------------ SLIDE 5: System Architecture ------------------
    slide5 = prs.slides.add_slide(blank_layout)
    set_bg(slide5, RGBColor(248, 250, 252))

    txBox = slide5.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.6))
    p = txBox.text_frame.paragraphs[0]
    p.text = "System Architecture"
    p.font.bold = True
    p.font.size = Pt(28)
    p.font.color.rgb = RGBColor(10, 25, 47)

    slide5.shapes.add_picture(arch_img, Inches(0.5), Inches(1.1), width=Inches(12.33))

    # ------------------ SLIDE 6: Methodology ------------------
    slide6 = prs.slides.add_slide(blank_layout)
    set_bg(slide6, RGBColor(248, 250, 252))

    txBox = slide6.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.6))
    p = txBox.text_frame.paragraphs[0]
    p.text = "Methodology"
    p.font.bold = True
    p.font.size = Pt(28)
    p.font.color.rgb = RGBColor(10, 25, 47)

    txBox = slide6.shapes.add_textbox(Inches(0.8), Inches(1.1), Inches(11.7), Inches(6.0))
    tf = txBox.text_frame
    tf.word_wrap = True

    m_steps = [
        ("Step 1: Data Collection & Input", "Collect static images (Kaggle ASL A-Z) and dynamic video sequences (WLASL word dataset) via webcam."),
        ("Step 2: Hand Landmark Tracking", "Detect 21 3D hand landmark coordinates per hand using MediaPipe HandLandmarker (supports dual hands)."),
        ("Step 3: Feature Engineering", "Extract 60-D wrist-relative normalized features for single hand, and 123-D features for dual hands across 30 frames."),
        ("Step 4: MoE Routing & Gesture Recognition", "Route single-hand features to Alphabet Expert (Dense MLP) and dual-hand features to Word Expert (Conv1D + GRU)."),
        ("Step 5: Stability Filtering & Sentence Assembly", "Apply a 5-frame consecutive stability filter to prevent flicker and assemble recognized gestures into sentences."),
        ("Step 6: Multilingual Translation & Audio TTS", "Translate recognized text into selected target languages using deep-translator with SQLite cache lookup (0ms repeat latency) & pyttsx3 TTS."),
        ("Step 7: Continual / Transfer Learning", "Record 10 custom gesture samples live in the GUI Teach Tab, freeze Conv1D+GRU backbone, fine-tune dense head, and hot-reload model."),
        ("Step 8: Output & Persistence", "Display real-time visual overlay, play speech output, and persist recognition history and logs in SQLite database.")
    ]

    for title, desc in m_steps:
        p = tf.add_paragraph()
        p.text = title + " — " + desc
        p.font.size = Pt(13.5)
        p.font.color.rgb = RGBColor(30, 41, 59)
        p.space_after = Pt(6)

    # ------------------ SLIDE 7: Methodology & Implementation Diagram ------------------
    slide7 = prs.slides.add_slide(blank_layout)
    set_bg(slide7, RGBColor(248, 250, 252))

    txBox = slide7.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.6))
    p = txBox.text_frame.paragraphs[0]
    p.text = "Methodology & Implementation Pipeline"
    p.font.bold = True
    p.font.size = Pt(28)
    p.font.color.rgb = RGBColor(10, 25, 47)

    slide7.shapes.add_picture(meth_img, Inches(0.5), Inches(1.1), width=Inches(12.33))

    # ------------------ SLIDE 8: Thank You! ------------------
    slide8 = prs.slides.add_slide(blank_layout)
    set_bg(slide8, RGBColor(10, 25, 47))

    txBox = slide8.shapes.add_textbox(Inches(1.0), Inches(2.8), Inches(11.3), Inches(2.0))
    p = txBox.text_frame.paragraphs[0]
    p.text = "Thank You!"
    p.font.bold = True
    p.font.size = Pt(60)
    p.alignment = PP_ALIGN.CENTER
    p.font.color.rgb = RGBColor(255, 255, 255)

    prs.save(PPTX_PATH)
    print(f"PPTX saved to {PPTX_PATH}")

# ---------------------------------------------------------
# BUILD PDF PRESENTATION
# ---------------------------------------------------------
def create_pdf_presentation(design_img, arch_img, meth_img):
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=landscape(letter),
        rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#0A192F'), spaceAfter=12)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#1D4ED8'), spaceAfter=8)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=12, textColor=colors.HexColor('#1E293B'), leading=16, spaceAfter=8)

    story = []

    # Slide 1: Title
    title_dark = ParagraphStyle('TitleDark', parent=styles['Heading1'], fontSize=28, textColor=colors.HexColor('#0A192F'), alignment=1, spaceAfter=15)
    sub_dark = ParagraphStyle('SubDark', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#0284C7'), alignment=1, spaceAfter=25)
    
    story.append(Spacer(1, 40))
    story.append(Paragraph("<b>Topic: Multilingual Real-Time Sign Language Translator</b>", title_dark))
    story.append(Paragraph("Mixture-of-Experts (MoE) Deep Learning System for Sign Recognition & Translation", sub_dark))
    story.append(Spacer(1, 20))

    t_data = [
        [Paragraph("<b>Under the guidance of:</b><br/>Dr. Prabha Seetaram Naik<br/>Associate Professor & HOD, Dept. of CSE(DS)", body_style),
         Paragraph("<b>Under the Co-guidance of:</b><br/>Prof. Bhagya M<br/>Assistant Professor", body_style),
         Paragraph("<b>Team Members (CSE, CITNC):</b><br/>• Lekhana N (1AJ23CS069)<br/>• Mohith Raj N (1AJ23CS082)<br/>• Pragathi N (1AJ23CS105)<br/>• Shohith Kumar K (1AJ23CS138)", body_style)]
    ]
    t = Table(t_data, colWidths=[250, 220, 250])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t)
    story.append(PageBreak())

    # Slide 2: Contents
    story.append(Paragraph("<b>Contents</b>", title_style))
    story.append(Spacer(1, 20))
    contents_p = (
        "<b>❖ Previous Review Feedback</b><br/><br/>"
        "<b>❖ System Design</b> (Data Flow Diagram & SQLite Data Model)<br/><br/>"
        "<b>❖ System Architecture</b> (Modular Mixture-of-Experts Architecture)<br/><br/>"
        "<b>❖ Methodology & Implementation Pipeline</b> (8-Step Execution & Continual Learning)"
    )
    story.append(Paragraph(contents_p, ParagraphStyle('ContStyle', parent=body_style, fontSize=16, leading=28)))
    story.append(PageBreak())

    # Slide 3: Previous Review Feedback
    story.append(Paragraph("<b>Previous Review Feedback & Project Solution</b>", title_style))
    fb_text = (
        "<b>Title:</b> Multilingual Real-Time Sign Language Translator<br/><br/>"
        "<b>Problem Definition:</b> Communication between sign language users and non-signers is difficult due to language barriers and limited real-time translation tools.<br/><br/>"
        "<b>Proposed Solution:</b> Develop a real-time system utilizing a Hierarchical Mixture-of-Experts (MoE) deep learning architecture that recognizes static letters (A-Z) and dynamic word gestures, translating them into text and speech in multiple languages.<br/><br/>"
        "<b>Objectives:</b><br/>"
        "• Recognize sign language gestures accurately in real time with an MoE router (0/1/2 hands).<br/>"
        "• Translate recognized signs into multiple target languages (Kannada, Hindi, Tamil, Telugu, etc.) with 0ms cache lookup.<br/>"
        "• Provide synchronized text overlay and audio speech (TTS) output.<br/>"
        "• Support live continual learning to extend the dictionary with custom gestures without full retraining."
    )
    story.append(Paragraph(fb_text, ParagraphStyle('FBStyle', parent=body_style, fontSize=13, leading=20)))
    story.append(PageBreak())

    # Slide 4: System Design
    story.append(Paragraph("<b>System Design Diagram</b>", title_style))
    story.append(Image(design_img, width=720, height=405))
    story.append(PageBreak())

    # Slide 5: System Architecture
    story.append(Paragraph("<b>System Architecture Diagram</b>", title_style))
    story.append(Image(arch_img, width=720, height=405))
    story.append(PageBreak())

    # Slide 6: Methodology
    story.append(Paragraph("<b>Methodology</b>", title_style))
    m_text = (
        "<b>Step 1: Data Collection & Input</b> — Collect static images (ASL A-Z) and dynamic video sequences (WLASL word dataset).<br/>"
        "<b>Step 2: Hand Landmark Tracking</b> — Detect 21 3D hand landmarks per hand using MediaPipe HandLandmarker.<br/>"
        "<b>Step 3: Feature Engineering</b> — Extract 60-D features for single hand, and 123-D features across 30 frames for dual hands.<br/>"
        "<b>Step 4: MoE Routing & Recognition</b> — Route single hand to Alphabet Expert (MLP) and dual hands to Word Expert (Conv1D+GRU).<br/>"
        "<b>Step 5: Stability Filtering & Sentence Assembly</b> — Apply 5-frame stability filter to eliminate flicker.<br/>"
        "<b>Step 6: Multilingual Translation & TTS</b> — Translate text via deep-translator with SQLite cache (0ms lookup) & pyttsx3 speech.<br/>"
        "<b>Step 7: Continual / Transfer Learning</b> — Record 10 custom gesture samples in GUI, freeze Conv1D+GRU, fine-tune dense head & hot-reload.<br/>"
        "<b>Step 8: Output & Persistence</b> — Display real-time UI overlay, play speech, and save history to SQLite DB."
    )
    story.append(Paragraph(m_text, ParagraphStyle('MStyle', parent=body_style, fontSize=12, leading=18)))
    story.append(PageBreak())

    # Slide 7: Methodology Diagram
    story.append(Paragraph("<b>Methodology & Implementation Pipeline Diagram</b>", title_style))
    story.append(Image(meth_img, width=720, height=405))
    story.append(PageBreak())

    # Slide 8: Thank You
    story.append(Spacer(1, 150))
    story.append(Paragraph("<b>Thank You!</b>", ParagraphStyle('TYStyle', parent=title_dark, fontSize=48, textColor=colors.HexColor('#0A192F'))))

    doc.build(story)
    print(f"PDF saved to {PDF_PATH}")

if __name__ == "__main__":
    print("Generating diagrams...")
    d_path = generate_system_design_diagram()
    a_path = generate_system_architecture_diagram()
    m_path = generate_methodology_diagram()

    print("Generating PPTX presentation...")
    create_pptx_presentation(d_path, a_path, m_path)

    print("Generating PDF presentation...")
    create_pdf_presentation(d_path, a_path, m_path)

    print("All done successfully!")
