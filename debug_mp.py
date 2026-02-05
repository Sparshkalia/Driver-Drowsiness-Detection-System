import mediapipe as mp
import sys

print(f"Python Executable: {sys.executable}")
print(f"MediaPipe File: {mp.__file__}")
print(f"Dir MP: {dir(mp)}")

try:
    import mediapipe.python.solutions as solutions
    print("Successfully imported mediapipe.python.solutions")
except ImportError as e:
    print(f"Failed internal import: {e}")

try:
    from mediapipe import solutions
    print("Successfully imported from mediapipe import solutions")
except ImportError as e:
    print(f"Failed from import: {e}")
