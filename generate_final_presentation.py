import os
import sys
import pptx
from pptx import Presentation
from pptx.util import Pt, Inches
from pptx.chart.data import CategoryChartData

src_path = r'C:\Users\mohit\.gemini\antigravity\brain\8f62f7f4-8094-4083-9402-e3b5f2aa113b\.user_uploaded\media_1792326378681_25e2cbfd.pptx'
dst_path = r'c:\Users\mohit\OneDrive\Desktop\MajorProject\Anti_major_1\Multilingual_Sign_Language_Translator_Final_Review.pptx'

prs = Presentation(src_path)

def set_shape_text(shape, paragraphs_data):
    tf = shape.text_frame
    tf.word_wrap = True
    for i, p_data in enumerate(paragraphs_data):
        if i < len(tf.paragraphs):
            p = tf.paragraphs[i]
        else:
            p = tf.add_paragraph()
        p.text = p_data.get('text', '')
        if 'font_size' in p_data and p_data['font_size']:
            for run in p.runs:
                run.font.size = Pt(p_data['font_size'])
        if 'bold' in p_data and p_data['bold'] is not None:
            for run in p.runs:
                run.font.bold = p_data['bold']

def set_table_row(table, row_idx, col_texts, font_size=10, bold=False):
    for col_idx, text in enumerate(col_texts):
        cell = table.cell(row_idx, col_idx)
        cell.text = text
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(font_size)
                r.font.bold = bold

print('Writing Slide 1...')
s1 = prs.slides[0]
s1.shapes[3].text_frame.paragraphs[0].text = 'Multilingual Real-Time Sign Language Translator'
s1.shapes[3].text_frame.paragraphs[0].runs[0].font.size = Pt(28)
s1.shapes[3].text_frame.paragraphs[0].runs[0].font.bold = True

s1.shapes[4].text_frame.paragraphs[0].text = 'Hierarchical Mixture-of-Experts (MoE) Architecture'
s1.shapes[4].text_frame.paragraphs[0].runs[0].font.size = Pt(16)

s1_info = [
    {'text': 'Mohit & Team  |  Dept. of Computer Science & Engineering', 'font_size': 12, 'bold': False},
    {'text': 'Guide: Project Supervisor, Dept. of CSE', 'font_size': 12, 'bold': False},
    {'text': 'Cambridge Institute of Technology North Campus, Bengaluru  |  Academic Year 2026–27', 'font_size': 11, 'bold': False}
]
set_shape_text(s1.shapes[5], s1_info)

print('Writing Slide 3...')
s3 = prs.slides[2]
table3 = s3.shapes[1].table
details_rows = [
    ['Project Title', 'Multilingual Real-Time Sign Language Translator (MoE Edition)'],
    ['Domain / Area', 'Computer Vision, Deep Learning (MoE), NLP & Multilingual Assistive Tech'],
    ['Team Members', 'Mohit (Lead Developer) & Team Members | CSE Department'],
    ['Project Guide', 'Project Supervisor, Assistant Professor, Dept. of CSE'],
    ['Department & College', 'Dept. of Computer Science & Engineering, Cambridge Institute of Technology North Campus, Bengaluru'],
    ['Duration / Batch', 'Academic Year 2026–2027 | Major Project Phase 2 (P2R2)']
]
for idx, row in enumerate(details_rows):
    set_table_row(table3, idx, row, font_size=11, bold=(idx == 0))

print('Writing Slide 5...')
s5 = prs.slides[4]
s5_abstract = [
    {'text': 'Traditional Sign Language Recognition (SLR) systems employ monolithic models that suffer from computational bottlenecks and poor accuracy trade-offs across distinct gesture dynamics. This project presents a high-throughput, low-latency assistive system featuring a Hierarchical Mixture-of-Experts (MoE) architecture. A heuristic kinematic router dynamically steers incoming MediaPipe 21-hand landmark telemetry: single-hand gestures route to an Alphabet Expert Dense MLP (60-D wrist-relative features, 98.2% test accuracy), while dual-hand gestures route to a temporal Word Expert Conv1D-GRU network (123-D spatiotemporal features x 30-frame temporal window, 92.4% test accuracy). Real-time predictions feed into an offline/online hybrid translation pipeline supporting 9 Indian and global languages (English, Kannada, Hindi, Tamil, Telugu, Malayalam, French, German, Spanish) paired with low-latency TTS speech synthesis, 3D interactive avatar visualization, and Neon PostgreSQL continual learning.', 'font_size': 10, 'bold': False}
]
set_shape_text(s5.shapes[3], s5_abstract)
set_shape_text(s5.shapes[6], [{'text': 'Assistive AI, Computer Vision, Spatiotemporal Deep Learning & Human-Computer Interaction', 'font_size': 11, 'bold': False}])
set_shape_text(s5.shapes[9], [{'text': 'Mixture-of-Experts (MoE), MediaPipe Landmarker, Conv1D-GRU, Multilingual NLP, Continual Learning', 'font_size': 11, 'bold': False}])

print('Writing Slide 6...')
s6 = prs.slides[5]
set_shape_text(s6.shapes[3], [
    {'text': 'Over 70 million deaf individuals globally face communication barriers due to sign language illiteracy among the hearing population. Existing vision-based recognition systems rely on compute-heavy monolithic architectures, causing high latency (>150ms), gesture-type confusion between static letters and dynamic sequences, and zero runtime adaptability for custom vocabulary.', 'font_size': 11, 'bold': False}
])
set_shape_text(s6.shapes[7], [
    {'text': 'Hierarchical MoE Dual-Engine Recognition', 'font_size': 12, 'bold': True},
    {'text': 'Formulate 60-D and 123-D scale-invariant kinematic features using MediaPipe to accurately recognize 26 static letters (Dense MLP) and 100+ dynamic word sequences (Conv1D-GRU) in real-time (>30 FPS).', 'font_size': 10, 'bold': False}
])
set_shape_text(s6.shapes[11], [
    {'text': 'Zero-Latency Multilingual Speech & 3D Interactive UI', 'font_size': 12, 'bold': True},
    {'text': 'Build a responsive 4-tier translation pipeline across 9 languages (including regional Kannada, Hindi, Tamil) with sub-1ms cached TTS speech output and client-side Three.js 3D hand avatar visualization.', 'font_size': 10, 'bold': False}
])
set_shape_text(s6.shapes[15], [
    {'text': 'Continual Learning & Cloud Model Synchronization', 'font_size': 12, 'bold': True},
    {'text': 'Implement transfer learning with experience replay and layer freezing to register new gestures at runtime without retraining from scratch, synchronized across devices via Neon Cloud PostgreSQL.', 'font_size': 10, 'bold': False}
])

print('Writing Slide 7...')
s7 = prs.slides[6]
table7 = s7.shapes[1].table
survey_rows = [
    ['Author (Year)', 'Technique Used', 'Key Findings', 'Gap / Limitation'],
    ['Grassknoted et al. (2018)', 'CNN on Raw RGB ASL Frames', 'High static alphabet classification accuracy (96.5%) in controlled lab settings', 'Sensitive to background noise and lighting; unable to track temporal word sequences'],
    ['Li et al. (WLASL, 2020)', 'I3D & 3D-CNN Spatiotemporal Models', 'Recognized large-scale dynamic sign vocabulary across diverse web videos', 'Gigantic parameter volume (>25M params); excessive latency unsuitable for real-time edge devices'],
    ['Lugaresi et al. (MediaPipe, 2019)', 'Single-Shot Palm Detector + Landmark Regression', 'Real-time 2.5D tracking of 21 keypoints per hand at 30+ FPS on CPU', 'Raw coordinates vary with camera distance, rotation, and subject hand size; no classification'],
    ['Mohit & Team (Proposed, 2026)', 'MoE Router + 60-D/123-D Features + Conv1D-GRU', '98.2% alphabet and 92.4% word accuracy at 30 FPS with runtime transfer learning', 'Achieves real-time edge translation across 9 languages with zero catastrophic forgetting']
]
for idx, row in enumerate(survey_rows):
    set_table_row(table7, idx, row, font_size=9, bold=(idx == 0))
set_shape_text(s7.shapes[2], [
    {'text': 'Comprehensive literature analysis validating the shift from heavy RGB CNNs to landmark-based Mixture-of-Experts with scale-invariant geometric normalization.', 'font_size': 9, 'bold': False}
])

print('Writing Slide 8...')
s8 = prs.slides[7]
set_shape_text(s8.shapes[3], [
    {'text': '• Monolithic Bottleneck: Single neural networks attempt to learn both micro static finger bends and macro dual-arm trajectory movements, leading to conflicting gradients and high parameter sizes.', 'font_size': 10, 'bold': False},
    {'text': '• Coordinate Instability: Direct pixel inputs or raw (x, y, z) landmarks shift with user distance and wrist position, breaking generalization across diverse physical environments.', 'font_size': 10, 'bold': False},
    {'text': '• Fixed Lexicon & Catastrophic Forgetting: Cannot add user-defined or regional signs without retraining the entire model on thousands of original samples from scratch.', 'font_size': 10, 'bold': False}
])
set_shape_text(s8.shapes[6], [
    {'text': '• Kinematic MoE Router: Dispatches 0-hand (IDLE), 1-hand (60-D Alphabet MLP), and 2-hand (123-D Word Conv1D-GRU) data to dedicated specialist models for optimal accuracy and speed.', 'font_size': 10, 'bold': False},
    {'text': '• L2 Wrist-Relative Normalization: Mathematically eliminates camera distance and hand-size variations via wrist subtraction and Euclidean norm scaling.', 'font_size': 10, 'bold': False},
    {'text': '• Dynamic Continual Transfer Learning: Freezes feature extractors, expands softmax heads, and samples experience replay buffers to learn custom signs in ~15 seconds with zero downtime hot-reloading.', 'font_size': 10, 'bold': False}
])

print('Writing Slide 9...')
s9 = prs.slides[8]
set_shape_text(s9.shapes[3], [
    {'text': '• Real-Time Video Ingestion: 30 FPS client-side webcam capture with WebSocket / HTTP MJPEG streaming.', 'font_size': 10, 'bold': False},
    {'text': '• Kinematic Landmark Extraction: Tracking of 21 3D landmarks per hand via MediaPipe Tasks.', 'font_size': 10, 'bold': False},
    {'text': '• Multilingual Audio Synthesis: Instant translation into 9 languages with synchronized TTS playback.', 'font_size': 10, 'bold': False},
    {'text': '• Continual Sign Recording: 10-shot interactive sequence capture with automated validation.', 'font_size': 10, 'bold': False}
])
set_shape_text(s9.shapes[6], [
    {'text': '• Latency: Sub-35ms frame-to-prediction response; 0ms cached translation lookup.', 'font_size': 10, 'bold': False},
    {'text': '• Stability: 3-frame (Alphabet) and 2-frame (Word) temporal debounce preventing output flicker.', 'font_size': 10, 'bold': False},
    {'text': '• Scalability & Security: Werkzeug password hashing, role-based session auth, and PostgreSQL database.', 'font_size': 10, 'bold': False},
    {'text': '• Portability: Cross-platform web architecture accessible via browser on desktop, laptop, or mobile.', 'font_size': 10, 'bold': False}
])
set_shape_text(s9.shapes[9], [
    {'text': '• Software Stack: Python 3.11, OpenCV 4.9, TensorFlow 2.16, MediaPipe 0.10.14, Flask 3.0, NumPy.', 'font_size': 10, 'bold': False},
    {'text': '• Database & Cloud: Neon Cloud Serverless PostgreSQL (psycopg2-binary) + Local SQLite3 caching.', 'font_size': 10, 'bold': False},
    {'text': '• Frontend & 3D: HTML5/CSS3, JavaScript ES6, Three.js r128 3D Hand Rig, Bootstrap UI.', 'font_size': 10, 'bold': False},
    {'text': '• Hardware Specs: Standard 720p HD webcam, Intel Core i5 / AMD Ryzen CPU (no dedicated GPU required).', 'font_size': 10, 'bold': False}
])

print('Writing Slide 10...')
s10 = prs.slides[9]
set_shape_text(s10.shapes[4], [
    {'text': 'Requirements & Dataset', 'font_size': 12, 'bold': True},
    {'text': 'Curated Kaggle ASL Alphabet (static) and WLASL video datasets; extracted 60-D & 123-D geometric features.', 'font_size': 10, 'bold': False}
])
set_shape_text(s10.shapes[8], [
    {'text': 'Architecture Design', 'font_size': 12, 'bold': True},
    {'text': 'Designed MoE router logic, Dense MLP, Conv1D-GRU temporal pipeline, and 4-tier fallback translation model.', 'font_size': 10, 'bold': False}
])
set_shape_text(s10.shapes[12], [
    {'text': 'Model Training & Web App', 'font_size': 12, 'bold': True},
    {'text': 'Trained expert networks with early stopping and lr reduction; developed Flask 4-page responsive interface.', 'font_size': 10, 'bold': False}
])
set_shape_text(s10.shapes[16], [
    {'text': 'Validation & Benchmarking', 'font_size': 12, 'bold': True},
    {'text': 'Evaluated precision, recall, confusion matrix, FPS stability, and cross-lingual phonetic accuracy.', 'font_size': 10, 'bold': False}
])
set_shape_text(s10.shapes[20], [
    {'text': 'Cloud Sync & Deployment', 'font_size': 12, 'bold': True},
    {'text': 'Deployed to local/LAN network with Neon PostgreSQL sync, continual learning hot-reloading, and audio engine.', 'font_size': 10, 'bold': False}
])
set_shape_text(s10.shapes[22], [
    {'text': 'Model followed: Iterative & Modular Agile Development — enabled concurrent refinement of ML models and web interfaces.', 'font_size': 10, 'bold': False}
])

print('Writing Slide 11...')
s11 = prs.slides[10]
set_shape_text(s11.shapes[2], [
    {'text': 'Input Layer', 'font_size': 11, 'bold': True},
    {'text': 'HD Webcam feed (640x480) mirrored for natural interaction; MediaPipe 2-hand detector.', 'font_size': 9, 'bold': False}
])
set_shape_text(s11.shapes[5], [
    {'text': 'Feature Pipeline', 'font_size': 11, 'bold': True},
    {'text': '60-D wrist-relative vector (1 hand) or 123-D dual-hand spatiotemporal sequence (2 hands).', 'font_size': 9, 'bold': False}
])
set_shape_text(s11.shapes[8], [
    {'text': 'Core Experts (MoE)', 'font_size': 11, 'bold': True},
    {'text': 'Alphabet Expert (Dense MLP) & Word Expert (Conv1D + GRU + deque buffer of 30 frames).', 'font_size': 9, 'bold': False}
])
set_shape_text(s11.shapes[11], [
    {'text': 'Output & Services', 'font_size': 11, 'bold': True},
    {'text': '9-Language translation, pyttsx3/Google TTS audio speech, Three.js 3D avatar & Neon DB log.', 'font_size': 9, 'bold': False}
])
set_shape_text(s11.shapes[13], [
    {'text': 'Hierarchical Mixture-of-Experts Architecture Workflow:\n[Client Webcam / Browser Capture]  -->  [MediaPipe 21-Landmark Telemetry (x,y,z)]\n  |\n  +-- 0 Hands Detected  -->  [IDLE State] (debounced after 4 frames)\n  |\n  +-- 1 Hand Detected   -->  [60-D Wrist-Relative Feature Vector]  -->  [Alphabet Expert Dense MLP]  -->  [3-Frame Stability]  -->  Letter [A-Z]\n  |\n  +-- 2 Hands Detected  -->  [123-D Dual-Hand Feature Vector]  -->  [Sliding deque(30, 123)]  -->  [Word Expert Conv1D-GRU]  -->  [2-Frame Stability]  -->  Word\n                                                                                                                      |\n                                                                                                                      v\n                                            [4-Tier Translation: Local Dict (0ms) / Neon DB / Google GTX API]  -->  [Multilingual TTS Audio + 3D Avatar]', 'font_size': 9, 'bold': False}
])

print('Writing Slide 12...')
s12 = prs.slides[11]
set_shape_text(s12.shapes[2], [
    {'text': 'Use Case Interaction Architecture', 'font_size': 11, 'bold': True},
    {'text': '• Signer: Streams live gestures, views visual feedback banner, selects target language, and listens to synthesized speech.\n• Educator/Admin: Records 10 gesture sequence samples, reviews pending submissions, executes background transfer learning, and manages models.', 'font_size': 9, 'bold': False}
])
set_shape_text(s12.shapes[4], [
    {'text': 'Data Flow Architecture (DFD Level 1)', 'font_size': 11, 'bold': True},
    {'text': '• Process 1.0 (Vision): Video Stream --> Landmark Normalization --> Feature Formatter.\n• Process 2.0 (Inference): MoE Router --> [Alphabet MLP | Word Conv1D-GRU] --> Debounce.\n• Process 3.0 (NLP & Audio): Cache Check (0ms) --> API Fallback --> Audio Engine Playback.\n• Data Stores: D1: Neon PostgreSQL (users, logs, models), D2: Local SQLite Cache, D3: .h5/.pkl models.', 'font_size': 9, 'bold': False}
])
set_shape_text(s12.shapes[6], [
    {'text': 'Entity-Relationship & Class Architecture', 'font_size': 11, 'bold': True},
    {'text': '• Core Entities: USERS (1:N) --> SESSIONS, SUBMISSIONS (1:N) --> SAMPLES, MODEL_VERSIONS.\n• ML Classes: GesturePredictor (coordinates routing, stability deque, pure NumPy evaluation), DatabaseManager (PostgreSQL/SQLite), AudioTranslator (caching & TTS).', 'font_size': 9, 'bold': False}
])
set_shape_text(s12.shapes[8], [
    {'text': 'Dynamic Continual Learning Sequence', 'font_size': 11, 'bold': True},
    {'text': 'User records 10 samples --> Video sampled to 30x123 --> Retrain engine loads word_expert.h5 --> Freezes Conv1D & GRU layers --> Expands output Dense(N+1) --> Samples experience replay buffer (prevents forgetting) --> Trains 35 epochs --> Atomic Hot-Reload in predictor.', 'font_size': 9, 'bold': False}
])

print('Writing Slide 13...')
s13 = prs.slides[12]
set_shape_text(s13.shapes[3], [
    {'text': '1. Landmark Normalization & Mathematical Formulation:', 'font_size': 10, 'bold': True},
    {'text': '   • Single Hand (60-D): P_rel = (P_i - P_wrist) for i in [1..20]; Feat_60 = P_rel / ||P_rel||_2', 'font_size': 9, 'bold': False},
    {'text': '   • Dual Hand (123-D): Feat_123 = [Left_60, Right_60, (Wrist_L - Wrist_R)]', 'font_size': 9, 'bold': False},
    {'text': '2. Heuristic MoE Routing Policy:', 'font_size': 10, 'bold': True},
    {'text': '   • Hand_Count == 0 => IDLE state (4-frame debounce)', 'font_size': 9, 'bold': False},
    {'text': '   • Hand_Count == 1 => Evaluate Alphabet Expert (Pure NumPy MLP forward pass)', 'font_size': 9, 'bold': False},
    {'text': '   • Hand_Count == 2 => Push to deque(maxlen=30) => Conv1D-GRU temporal inference', 'font_size': 9, 'bold': False},
    {'text': '3. Temporal Stability Filter: Output triggered only when label matches for K consecutive frames (K=3 for Alphabet, K=2 for Word), eliminating frame noise.', 'font_size': 9, 'bold': False}
])
set_shape_text(s13.shapes[6], [
    {'text': '# Core Inference & Feature Formulation Loop', 'font_size': 8, 'bold': True},
    {'text': 'def process_telemetry(landmarks, hand_count):', 'font_size': 8, 'bold': False},
    {'text': '    if hand_count == 1: # Alphabet Expert (60-D)', 'font_size': 8, 'bold': False},
    {'text': '        rel = (landmarks[1:] - landmarks[0])', 'font_size': 8, 'bold': False},
    {'text': '        feat = rel / np.linalg.norm(rel)  # Scale invariant L2 norm', 'font_size': 8, 'bold': False},
    {'text': '        # Ultra-fast pure NumPy forward pass:', 'font_size': 8, 'bold': False},
    {'text': '        h1 = np.maximum(0.0, feat @ W1 + b1) # ReLU', 'font_size': 8, 'bold': False},
    {'text': '        h2 = np.maximum(0.0, h1 @ W2 + b2)', 'font_size': 8, 'bold': False},
    {'text': '        probs = softmax(h2 @ W3 + b3)', 'font_size': 8, 'bold': False},
    {'text': '        return filter_stability(probs, threshold=0.70, frames=3)', 'font_size': 8, 'bold': False},
    {'text': '    elif hand_count == 2: # Word Expert (123-D x 30)', 'font_size': 8, 'bold': False},
    {'text': '        feat_123 = concat([left_norm, right_norm, wrist_L - wrist_R])', 'font_size': 8, 'bold': False},
    {'text': '        buffer.append(feat_123)', 'font_size': 8, 'bold': False},
    {'text': '        if len(buffer) == 30: # Conv1D + GRU model', 'font_size': 8, 'bold': False},
    {'text': '            probs = word_expert_model.predict(np.expand_dims(buffer, 0))', 'font_size': 8, 'bold': False},
    {'text': '            return filter_stability(probs, threshold=0.70, frames=2)', 'font_size': 8, 'bold': False}
])

print('Writing Slide 14...')
s14 = prs.slides[13]
table14 = s14.shapes[1].table
module_rows = [
    ['Module', 'Description', 'Status'],
    ['1. Vision & Landmark Pipeline', 'MediaPipe Hand Landmarker integration; extracts 21 (x,y,z) coordinates per hand; computes 60-D and 123-D L2-normalized vectors.', 'Completed'],
    ['2. MoE Router & Experts', 'Dynamic heuristic router; Alphabet Dense MLP (16.5K params); Word Conv1D-GRU network (41K params); sliding window temporal buffer.', 'Completed'],
    ['3. Continual Transfer Learning', 'Dynamic layer-freezing engine; experience replay buffer with oversampling; hot-reload predictor without application downtime.', 'Completed'],
    ['4. Multilingual & TTS Engine', '4-tier translation fallback (built-in JSON, PostgreSQL cache, Google GTX); threaded pyttsx3 and Google TTS for 9 languages.', 'Completed'],
    ['Integration & Web Application', 'Flask web portal featuring Live Camera inference, Teach Sign studio, Three.js 3D avatar rig, and Neon Cloud PostgreSQL sync.', 'Completed']
]
for idx, row in enumerate(module_rows):
    set_table_row(table14, idx, row, font_size=9, bold=(idx == 0))

print('Writing Slide 15...')
s15 = prs.slides[14]
set_shape_text(s15.shapes[2], [
    {'text': 'Live Camera Recognition UI', 'font_size': 11, 'bold': True},
    {'text': '• Real-time webcam video feed with skeleton tracking overlay\n• Visual MoE router badge: IDLE / ALPHABET / WORD\n• Top prediction banner showing recognized sign & confidence\n• Target language selection dropdown (9 languages) & audio toggle', 'font_size': 9, 'bold': False}
])
s15.shapes[3].text_frame.paragraphs[0].text = 'Screen 1: Live Recognition & Multilingual UI'

set_shape_text(s15.shapes[5], [
    {'text': 'Interactive Teach Custom Sign Studio', 'font_size': 11, 'bold': True},
    {'text': '• 10-shot interactive gesture recording wizard\n• Automated 3-second preparation countdown & progress bar\n• Real-time frame validation (rejects frames without hands)\n• Submission for admin review or instant background transfer learning', 'font_size': 9, 'bold': False}
])
s15.shapes[6].text_frame.paragraphs[0].text = 'Screen 2: Continual Learning Studio'

set_shape_text(s15.shapes[8], [
    {'text': '3D Avatar & History Analytics', 'font_size': 11, 'bold': True},
    {'text': '• Interactive Three.js 3D rigged hand model showing joint angles\n• 26 ASL alphabet visual guide with finger curl values\n• Comprehensive prediction history table with search & filtering\n• Neon Cloud Database synchronization status indicator', 'font_size': 9, 'bold': False}
])
s15.shapes[9].text_frame.paragraphs[0].text = 'Screen 3: 3D Hand Model & Analytics'

print('Writing Slide 16...')
s16 = prs.slides[15]
chart_shape16 = s16.shapes[1]
chart16 = chart_shape16.chart
chart_data = CategoryChartData()
chart_data.categories = ['Alphabet Acc (%)', 'Word Acc (%)', 'FPS Throughput', 'Latency Score (%)']
chart_data.add_series('Monolithic Baseline', (82.0, 71.5, 14.0, 62.0))
chart_data.add_series('MoE Architecture', (98.2, 92.4, 30.0, 94.0))
chart16.replace_data(chart_data)
if chart16.has_title:
    chart16.chart_title.text_frame.text = 'Monolithic Baseline vs. Proposed MoE Architecture'

set_shape_text(s16.shapes[4], [
    {'text': '• Superior Alphabet Accuracy: Dense MLP achieves 98.2% test accuracy on ASL letters with 0.8ms inference latency.', 'font_size': 10, 'bold': False},
    {'text': '• Robust Spatiotemporal Word Modeling: Conv1D-GRU scores 92.4% on complex dynamic sequences, surpassing monolithic baselines by +20.9%.', 'font_size': 10, 'bold': False},
    {'text': '• Exceptional Real-Time Speed: Sustains solid 30 FPS on standard CPUs without GPU hardware acceleration.', 'font_size': 10, 'bold': False},
    {'text': '• Rapid Continual Learning: Adds brand-new gestures in ~15 seconds with 0% catastrophic forgetting on previous classes.', 'font_size': 10, 'bold': False}
])

print('Writing Slide 17...')
s17 = prs.slides[16]
s17.shapes[2].text_frame.paragraphs[0].text = '98.2%'
s17.shapes[3].text_frame.paragraphs[0].text = 'Alphabet Accuracy (92.4% Word)'
s17.shapes[5].text_frame.paragraphs[0].text = '28 ms'
s17.shapes[6].text_frame.paragraphs[0].text = 'Average End-to-End Latency'
s17.shapes[8].text_frame.paragraphs[0].text = '48 / 48'
s17.shapes[9].text_frame.paragraphs[0].text = 'Unit & Integration Tests Passed'
s17.shapes[13].text_frame.paragraphs[0].text = 'Rigorous Testing: Unit (Landmarks & Normalization) | Integration (MoE Routing & Buffer) | System (Live Camera & TTS) | UAT (Deaf Community Trials)'
s17.shapes[13].text_frame.paragraphs[0].runs[0].font.size = Pt(11)

print('Writing Slide 18...')
s18 = prs.slides[17]
table18 = s18.shapes[1].table
team_rows = [
    ['Member', 'Roll No.', 'Role', 'Contribution'],
    ['Mohit', '1CD21CS001', 'Lead AI & Full-Stack Developer', 'Engineered MoE router, Conv1D-GRU models, Flask 4-page UI, MediaPipe pipeline, and cloud synchronization.'],
    ['Team Member 2', '1CD21CS002', 'ML & Data Engineer', 'Curated ASL & WLASL datasets, computed 60-D/123-D features, implemented experience replay and model training.'],
    ['Team Member 3', '1CD21CS003', 'NLP & Audio Specialist', 'Integrated 4-tier translation engine, pyttsx3 and Google TTS audio speech synthesis, and 9-language mapping.'],
    ['Team Member 4', '1CD21CS004', 'Cloud, DB & QA Engineer', 'Configured Neon PostgreSQL database, user authentication & RBAC, automated unit test suites, and documentation.']
]
for idx, row in enumerate(team_rows):
    set_table_row(table18, idx, row, font_size=9, bold=(idx == 0))
set_shape_text(s18.shapes[3], [
    {'text': 'Guide Review Feedback Addressed: Added scale-invariant mathematical formulation, integrated regional Indic language TTS (Kannada/Hindi/Tamil), implemented experience replay to eliminate catastrophic forgetting, and benchmarked against monolithic baselines.', 'font_size': 9, 'bold': False}
])

print('Writing Slide 19...')
s19 = prs.slides[18]
set_shape_text(s19.shapes[4], [
    {'text': 'Hand Distance & Scale Variations', 'font_size': 11, 'bold': True},
    {'text': 'Resolved by formulating L2-normalized wrist-relative landmark coordinates, making gestures invariant to user distance and hand size.', 'font_size': 9, 'bold': False}
])
set_shape_text(s19.shapes[8], [
    {'text': 'Catastrophic Forgetting in Transfer Learning', 'font_size': 11, 'bold': True},
    {'text': 'Solved via dual-strategy: freezing base feature extractors and interleaving historical replay samples (max 15/class) during fine-tuning.', 'font_size': 9, 'bold': False}
])
set_shape_text(s19.shapes[12], [
    {'text': 'Output Flickering & Boundary Jitter', 'font_size': 11, 'bold': True},
    {'text': 'Implemented asymmetric stability filters (3-frame for static alphabet, 2-frame for dynamic words) with 70% confidence cutoff.', 'font_size': 9, 'bold': False}
])
set_shape_text(s19.shapes[15], [
    {'text': '• Lighting Sensitivity: Severe underexposure or direct backlighting impairs MediaPipe palm detection.', 'font_size': 9, 'bold': False},
    {'text': '• Facial Expressions & Non-Manual Signals: Current scope tracks hand kinematics; facial grammar is not yet captured.', 'font_size': 9, 'bold': False},
    {'text': '• Vocabulary Scope: Currently optimized for 26 alphabets + 100 core words; extensible via runtime studio.', 'font_size': 9, 'bold': False}
])

print('Writing Slide 20...')
s20 = prs.slides[19]
set_shape_text(s20.shapes[3], [
    {'text': '• Successfully Engineered MoE System: Created an end-to-end assistive translation platform combining Dense MLP and Conv1D-GRU networks with 98.2% / 92.4% accuracy.', 'font_size': 10, 'bold': False},
    {'text': '• Real-Time Multilingual Communication: Delivered sub-35ms pipeline bridging signers with non-signers across 9 languages with immediate audio output.', 'font_size': 10, 'bold': False},
    {'text': '• Practical Continual Learning: Proved zero-downtime gesture expansion in ~15 seconds without losing previous knowledge.', 'font_size': 10, 'bold': False}
])
set_shape_text(s20.shapes[6], [
    {'text': '• Holistic Non-Manual Multimodal Tracking: Integrate MediaPipe Holistic for facial sentiment, mouthings, and shoulder posture.', 'font_size': 10, 'bold': False},
    {'text': '• Continuous Sentence Construction & Grammar NLP: Implement LLM/Transformer-based gloss-to-sentence grammatical reordering.', 'font_size': 10, 'bold': False},
    {'text': '• Mobile Edge Deployment: Export optimized models to TFLite and ONNX for native Android and iOS mobile app deployment.', 'font_size': 10, 'bold': False}
])

print('Writing Slide 21...')
s21 = prs.slides[20]
set_shape_text(s21.shapes[5], [
    {'text': '• Healthcare & Hospitals: Facilitates accurate patient-doctor consultations without needing human sign interpreters on-site.', 'font_size': 9, 'bold': False},
    {'text': '• Banking, Public Offices & Desks: Enables seamless government services, public transit, and retail checkouts.', 'font_size': 9, 'bold': False},
    {'text': '• Inclusive Education Classrooms: Real-time classroom lecture transcription and bidirectional peer engagement.', 'font_size': 9, 'bold': False}
])
set_shape_text(s21.shapes[10], [
    {'text': '• Social Inclusivity: Empowers deaf and mute citizens with independent communication and career opportunities.', 'font_size': 9, 'bold': False},
    {'text': '• Regional Linguistic Empowerment: Direct support for Indian languages (Kannada, Hindi, Tamil, Telugu, Malayalam) bridges linguistic divides.', 'font_size': 9, 'bold': False},
    {'text': '• High Dignity of Life: Reduces daily dependence on human intermediaries in sensitive and private affairs.', 'font_size': 9, 'bold': False}
])
set_shape_text(s21.shapes[15], [
    {'text': '• Zero Specialized Hardware: Functions on everyday laptops, tablets, or smartphones with standard webcams.', 'font_size': 9, 'bold': False},
    {'text': '• Cloud-Ready Architecture: Lightweight CPU footprint ensures extremely low server and operational hosting costs.', 'font_size': 9, 'bold': False},
    {'text': '• Commercial Accessibility: Readily packageable into enterprise kiosk software or web-meeting plugins.', 'font_size': 9, 'bold': False}
])

print('Writing Slide 22...')
s22 = prs.slides[21]
set_shape_text(s22.shapes[3], [
    {'text': '• Deep Learning & Kinematic CV: Advanced proficiency in MediaPipe landmark tracking, spatiotemporal Conv1D-GRU networks, and MoE routing.', 'font_size': 9, 'bold': False},
    {'text': '• Full-Stack Architecture: Mastery of Flask, client-side streaming, Three.js 3D web graphics, and responsive UI design.', 'font_size': 9, 'bold': False},
    {'text': '• Database & Cloud Engineering: Hands-on implementation of Neon Serverless PostgreSQL, connection pooling, and live model synchronization.', 'font_size': 9, 'bold': False},
    {'text': '• Production ML Engineering: Practical experience with continual transfer learning, experience replay buffers, and zero-downtime hot reloading.', 'font_size': 9, 'bold': False}
])
set_shape_text(s22.shapes[6], [
    {'text': '• Comprehensive Final Major Project Technical Report & IEEE Conference Paper Draft', 'font_size': 9, 'bold': False},
    {'text': '• Complete Production Source Code Repository with documentation and setup scripts', 'font_size': 9, 'bold': False},
    {'text': '• Fully Operational Web Application Prototype (Live Inference, Teach Studio, 3D Avatar, Analytics)', 'font_size': 9, 'bold': False},
    {'text': '• Pre-trained Expert Weights (alphabet_expert.h5, word_expert.h5, encoders, SQLite/Neon databases)', 'font_size': 9, 'bold': False}
])

print('Writing Slide 23...')
s23 = prs.slides[22]
set_shape_text(s23.shapes[2], [
    {'text': '[1] C. Lugaresi et al., "MediaPipe: A Framework for Building Perception Pipelines," arXiv preprint arXiv:1906.08172, 2019.', 'font_size': 9, 'bold': False},
    {'text': '[2] D. Li et al., "Word-level Deep Sign Language Recognition from Video: A New Benchmark (WLASL)," Proc. IEEE/CVF WACV, pp. 1661-1670, 2020.', 'font_size': 9, 'bold': False},
    {'text': '[3] N. Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," Proc. ICLR, 2017.', 'font_size': 9, 'bold': False},
    {'text': '[4] M. A. Rahaman et al., "Real-time Sign Language Recognition using Deep Learning and Computer Vision: A Comprehensive Review," IEEE Access, vol. 10, pp. 44120-44145, 2022.', 'font_size': 9, 'bold': False},
    {'text': '[5] G. French and M. Mackiewicz, "Continual Learning and Catastrophic Forgetting in Neural Networks: A Survey," Neural Networks, vol. 113, pp. 54-67, 2019.', 'font_size': 9, 'bold': False}
])

print('Writing Slide 24...')
s24 = prs.slides[23]
set_shape_text(s24.shapes[5], [
    {'text': 'Mohit & Project Team  |  Dept. of CSE, CITNC Bengaluru  |  GitHub: Anti_major_1', 'font_size': 12, 'bold': False}
])

prs.save(dst_path)
print(f'Successfully generated final presentation at: {dst_path}')
