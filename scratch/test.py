import tensorflow as tf
import numpy as np

model_path = r"c:\Users\mohit\OneDrive\Desktop\MajorProject\Anti_major_1\models\word_expert.h5"
print(f"Loading {model_path}...")
old_model = tf.keras.models.load_model(model_path, compile=False)

try:
    print("Trying build...")
    old_model.build(input_shape=(None, 30, 123))
    print("Build successful.")
    
    print("Trying dummy forward pass...")
    dummy = np.zeros((1, 30, 123), dtype=np.float32)
    _ = old_model(dummy)
    print("Dummy forward pass successful.")
    
    print("Getting layer output...")
    penultimate_output = old_model.layers[-2].output
    print("Success!", penultimate_output)
except Exception as e:
    import traceback
    traceback.print_exc()
