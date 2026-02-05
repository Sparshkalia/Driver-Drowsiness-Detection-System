import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# Load TinyML model
interpreter = tf.lite.Interpreter(model_path="tflite_learn_897057_10.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero_point = input_details[0]["quantization"]
output_scale, output_zero_point = output_details[0]["quantization"]

IMG_SIZE = 96
# 5-10 frames smoothing as requested
pred_buffer = deque(maxlen=8) 
status_buffer = deque(maxlen=10) # For additional stability on the label

# Initialize Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def predict_eye(crop):
    """Predicts if an eye crop is Open (Awake) or Closed (Drowsy)."""
    if crop.size == 0: return 0.0
    
    img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img / input_scale + input_zero_point
    img = np.clip(img, -128, 127).astype(np.int8)
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    output = (output.astype(np.float32) - output_zero_point) * output_scale
    
    # Return "AWAKE" probability (Index 0)
    # The user reported "Drowsy" (Low Awake Prob) when eyes are Open.
    # This implies the model outputs High Prob at Index 0 for Open Eyes.
    return output[0]

# State Retention variables
last_awake_prob = 0.5
last_label = "WAITING"
last_color = (100, 100, 100)

cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    current_frame_prob = None
    
    if len(faces) > 0:
        # Find largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # --- ROI Constraint: Top 60% of face ---
        face_roi_h = int(h * 0.6)
        roi_gray = gray[y : y + face_roi_h, x : x + w]
        roi_color = frame[y : y + face_roi_h, x : x + w]
        
        # Show ROI for debugging (optional)
        # cv2.rectangle(frame, (x, y), (x+w, y+face_roi_h), (0, 255, 255), 1)

        # Detect eyes in ROI
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10, # Strict neighbor check
            minSize=(int(w/10), int(w/10)) # Size filtering relative to face width
        )
        
        eye_probs = []
        for (ex, ey, ew, eh) in eyes:
            # Draw eye rectangle relative to full frame
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 1)
            
            # Extract eye crop from color ROI
            eye_crop = roi_color[ey : ey + eh, ex : ex + ew]
            
            # Predict
            prob = predict_eye(eye_crop)
            eye_probs.append(prob)
            
        if len(eye_probs) > 0:
            # Average probability of all detected eyes
            current_frame_prob = sum(eye_probs) / len(eye_probs)
        else:
            # Face found but no eyes detected?
            # Haar Cascade often fails to detect closed eyes.
            # So, if face is present but eyes are missing, assume they are CLOSED (Drowsy).
            current_frame_prob = 0.0

    else:
        # No face found? Retain last known state but maybe decay towards "Unknown" if long time?
        # For now, per requirement: "retain the last known state"
        current_frame_prob = last_awake_prob

    # --- Signal Smoothing ---
    if current_frame_prob is not None:
        pred_buffer.append(current_frame_prob)
        last_awake_prob = current_frame_prob # Update last known
    
    # Calculate smooth probability
    if len(pred_buffer) > 0:
        smooth_prob = sum(pred_buffer) / len(pred_buffer)
    else:
        smooth_prob = 0.5 # Default

    # Hysteresis / Buffer for label stability
    if smooth_prob > 0.5:
        status_buffer.append(1) # Awake
    else:
        status_buffer.append(0) # Drowsy
        
    # Majority vote / Average from status buffer
    if sum(status_buffer) / len(status_buffer) > 0.5 if len(status_buffer) > 0 else 0.5:
        label = "AWAKE"
        color = (0, 255, 0)
    else:
        label = "DROWSY"
        color = (0, 0, 255)
    
    confidence = smooth_prob if label == "AWAKE" else (1.0 - smooth_prob)

    # Display Text
    text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    cv2.imshow("Drowsiness Detection (Haar)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
