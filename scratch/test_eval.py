import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path

def main():
    model_path = r"c:\Users\mohit\OneDrive\Desktop\MajorProject\Anti_major_1\models\word_expert.h5"
    enc_path = r"c:\Users\mohit\OneDrive\Desktop\MajorProject\Anti_major_1\models\word_encoder.pkl"
    hi_dir = Path(r"c:\Users\mohit\OneDrive\Desktop\MajorProject\Anti_major_1\data\processed\words\HI")
    
    if not hi_dir.exists():
        print("HI directory not found.")
        return
        
    model = tf.keras.models.load_model(model_path, compile=False)
    le = joblib.load(enc_path)
    print("Classes in LE:", le.classes_)
    
    print("Evaluating samples...")
    for npy_file in hi_dir.glob("*.npy"):
        arr = np.load(str(npy_file))
        inp = arr[np.newaxis, ...]
        probs = model.predict(inp, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = le.inverse_transform([idx])[0]
        
        print(f"{npy_file.name}: Predicted {label} (conf={conf:.3f})")

if __name__ == "__main__":
    main()
