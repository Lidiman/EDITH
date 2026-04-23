"""
AI Harmonizer — Finger-Count Hand Gesture Musical Synthesizer
=============================================================
Uses MediaPipe hand-tracking to detect gestures via webcam.

Left hand (root note):
  0 fingers = A | 1 = C | 2 = D | 3 = E | 4 = F | 5 = G

Right hand (chord type):
  0 fingers = Major | 1 = Minor | 2 = Maj7 | 3 = 7 | 4 = Aug | 5 = Aug7

Keyboard
--------
  q / ESC  — quit
  r        — toggle reverb
  +/-      — octave up/down
"""

import sys, os, time, threading, math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import sounddevice as sd

# ── AUDIO CONFIG ──
SAMPLE_RATE = 44100
BLOCK_SIZE  = 1024
CHANNELS    = 1

# ── MUSICAL DATA ──
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# Left hand: finger count → (name, semitone offset from C)
ROOT_BY_FINGERS = {
    0: ('A',  9),
    1: ('C',  0),
    2: ('D',  2),
    3: ('E',  4),
    4: ('F',  5),
    5: ('G',  7),
}

# Right hand: finger count → (name, intervals)
CHORD_BY_FINGERS = {
    0: ('Major', [0, 4, 7]),
    1: ('Minor', [0, 3, 7]),
    2: ('Maj7',  [0, 4, 7, 11]),
    3: ('7',     [0, 4, 7, 10]),
    4: ('Aug',   [0, 4, 8]),
    5: ('Aug7',  [0, 4, 8, 10]),
}

BASE_FREQ = 261.63  # C4

# ── COLORS (BGR) ──
COL_BG      = (20, 20, 20)
COL_ACCENT  = (255, 160, 50)
COL_ACCENT2 = (80, 220, 255)
COL_MUTE    = (80, 80, 80)
COL_TEXT    = (240, 240, 240)
COL_GREEN   = (100, 220, 100)
COL_PURPLE  = (200, 100, 255)
COL_GLOW_L  = (50, 200, 255)   # left circle glow
COL_GLOW_R  = (255, 120, 200)  # right circle glow

VIS_COLORS = [
    (255,100,50),(255,180,50),(50,220,255),(100,255,180),(200,100,255),
]

# ── AUDIO ENGINE ──
class HarmonizerEngine:
    def __init__(self):
        self.target_freqs = []
        self.target_amp = 0.0
        self.phases = np.zeros(16, dtype=np.float64)
        self.reverb_on = False
        self.reverb_buf = np.zeros(SAMPLE_RATE, dtype=np.float64)
        self.reverb_idx = 0
        self.lock = threading.Lock()
        self._stream = None
        self._amp_smooth = 0.0
        self.last_block = np.zeros(BLOCK_SIZE, dtype=np.float32)

    def start(self):
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
            channels=CHANNELS, dtype='float32', callback=self._callback)
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop(); self._stream.close()

    def set_notes(self, freqs, amp):
        with self.lock:
            self.target_freqs = freqs[:16]
            self.target_amp = np.clip(amp, 0.0, 1.0)

    def _callback(self, outdata, frames, time_info, status):
        with self.lock:
            freqs = list(self.target_freqs)
            target = self.target_amp
        t = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
        signal = np.zeros(frames, dtype=np.float64)
        ramp = np.linspace(self._amp_smooth, target, frames)
        self._amp_smooth = target
        if freqs:
            for i, f in enumerate(freqs):
                if f <= 0: continue
                detune = 1.0 + (i * 0.001)
                phase_inc = 2.0 * np.pi * f * detune * t + self.phases[i]
                wave = np.sin(phase_inc) + 0.15 * np.sin(2 * phase_inc)
                signal += wave
                self.phases[i] += 2.0*np.pi*f*detune*frames/SAMPLE_RATE
                self.phases[i] %= (2.0 * np.pi)
            signal /= max(len(freqs), 1)
        signal *= ramp
        if self.reverb_on:
            ds = int(0.08 * SAMPLE_RATE)
            for i in range(frames):
                idx = (self.reverb_idx + i) % len(self.reverb_buf)
                di = (idx - ds) % len(self.reverb_buf)
                wet = self.reverb_buf[di] * 0.4
                out_s = signal[i] + wet
                self.reverb_buf[idx] = out_s
                signal[i] = out_s * 0.8 + signal[i] * 0.2
            self.reverb_idx = (self.reverb_idx + frames) % len(self.reverb_buf)
        signal = np.tanh(signal * 0.8)
        block = signal.astype(np.float32)
        self.last_block = block.copy()
        outdata[:, 0] = block

# ── GESTURE DETECTOR ──
class GestureDetector:
    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),(0,17),
    ]

    def __init__(self, model_path):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=2,
            min_hand_detection_confidence=0.7, min_tracking_confidence=0.6,
            running_mode=vision.RunningMode.IMAGE)
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self.landmarker.detect(mp_image)

    @staticmethod
    def count_fingers(landmarks, handedness_label):
        lm = landmarks; count = 0
        if handedness_label == 'Right':
            if lm[4].x < lm[3].x: count += 1
        else:
            if lm[4].x > lm[3].x: count += 1
        for tip, pip in zip(GestureDetector.FINGER_TIPS, GestureDetector.FINGER_PIPS):
            if lm[tip].y < lm[pip].y: count += 1
        return count

    @staticmethod
    def get_fingertip(landmarks):
        return landmarks[8].x, landmarks[8].y

    @staticmethod
    def get_palm_center(landmarks):
        lm = landmarks
        return (lm[0].x+lm[5].x+lm[17].x)/3, (lm[0].y+lm[5].y+lm[17].y)/3

    @staticmethod
    def is_pinch(landmarks):
        lm = landmarks
        return math.hypot(lm[4].x-lm[8].x, lm[4].y-lm[8].y) < 0.05

    @staticmethod
    def draw_landmarks(frame, landmarks, w, h):
        pts = []
        for lm in landmarks:
            px, py = int(lm.x*w), int(lm.y*h)
            pts.append((px, py))
            cv2.circle(frame, (px,py), 3, COL_ACCENT, -1, cv2.LINE_AA)
        for a, b in GestureDetector.HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], COL_ACCENT2, 1, cv2.LINE_AA)

# ── FINGER COUNT DISPLAY ──
def draw_finger_indicators(img, count, x, y, label, color):
    """Draw finger count dots with a label."""
    cv2.putText(img, label, (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    for i in range(6):  # 0-5 fingers
        cx = x + i * 28
        c = color if i <= count else COL_MUTE
        cv2.circle(img, (cx, y), 9, c, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, y), 9, COL_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, str(i), (cx - 4, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COL_BG, 1, cv2.LINE_AA)


def draw_selection_display(img, cx, cy, root_name, chord_name, glow_root, glow_chord):
    """Draw a large centered display of the current root+chord."""
    overlay = img.copy()
    # Background pill
    pw, ph = 300, 80
    draw_rounded_rect(img, (cx - pw//2, cy - ph//2),
                      (cx + pw//2, cy + ph//2), COL_BG, radius=20, alpha=0.8)
    # Root note
    if root_name:
        ts = cv2.getTextSize(root_name, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)[0]
        cv2.putText(img, root_name, (cx - ts[0]//2 - 60, cy + ts[1]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, glow_root, 3, cv2.LINE_AA)
    # Chord type
    if chord_name:
        ts = cv2.getTextSize(chord_name, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        cv2.putText(img, chord_name, (cx - ts[0]//2 + 50, cy + ts[1]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, glow_chord, 2, cv2.LINE_AA)

# ── HUD HELPERS ──
def draw_rounded_rect(img, pt1, pt2, color, radius=15, thickness=-1, alpha=0.6):
    overlay = img.copy()
    x1,y1 = pt1; x2,y2 = pt2
    cv2.rectangle(overlay,(x1+radius,y1),(x2-radius,y2),color,thickness)
    cv2.rectangle(overlay,(x1,y1+radius),(x2,y2-radius),color,thickness)
    cv2.circle(overlay,(x1+radius,y1+radius),radius,color,thickness)
    cv2.circle(overlay,(x2-radius,y1+radius),radius,color,thickness)
    cv2.circle(overlay,(x1+radius,y2-radius),radius,color,thickness)
    cv2.circle(overlay,(x2-radius,y2-radius),radius,color,thickness)
    cv2.addWeighted(overlay,alpha,img,1-alpha,0,img)

def draw_waveform(img, waveform, x, y, w, h, color=COL_ACCENT2):
    if len(waveform) == 0: return
    step = max(1, len(waveform)//w)
    samples = waveform[::step][:w]
    mid_y = y + h//2
    pts = []
    for i, s in enumerate(samples):
        px = x + i
        py = int(mid_y - s*(h//2)*0.9)
        py = max(y, min(y+h, py))
        pts.append((px, py))
    if len(pts) > 1:
        for thick, am in [(5,0.2),(3,0.4),(1,1.0)]:
            c = tuple(int(ch*am) for ch in color)
            cv2.polylines(img,[np.array(pts)],False,c,thick,cv2.LINE_AA)

def draw_status_panel(img, info, x, y):
    pw, ph = 260, 170
    draw_rounded_rect(img,(x,y),(x+pw,y+ph),COL_BG,alpha=0.75)
    ty = y+28; lh = 26
    items = [
        ('Root',   info.get('root','-'),   COL_GLOW_L),
        ('Chord',  info.get('chord','-'),  COL_GLOW_R),
        ('Playing',info.get('playing','-'),COL_ACCENT),
        ('Octave', str(info.get('octave',4)), COL_PURPLE),
        ('Reverb', 'ON' if info.get('reverb') else 'OFF',
         COL_GREEN if info.get('reverb') else COL_MUTE),
        ('Vol',    f"{info.get('volume',0):.0%}", COL_TEXT),
    ]
    for label, value, color in items:
        cv2.putText(img,f'{label}:',(x+12,ty),
                    cv2.FONT_HERSHEY_SIMPLEX,0.48,COL_MUTE,1,cv2.LINE_AA)
        cv2.putText(img,str(value),(x+100,ty),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,color,2,cv2.LINE_AA)
        ty += lh

def draw_title(img, w):
    cv2.putText(img,"AI  HARMONIZER",(w//2-140,35),
                cv2.FONT_HERSHEY_SIMPLEX,1.0,COL_ACCENT,2,cv2.LINE_AA)
    cv2.putText(img,"Finger Gesture Control",(w//2-110,58),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,COL_ACCENT2,1,cv2.LINE_AA)

def draw_help_panel(img, x, y):
    helps = [
        "L: 0=A 1=C 2=D 3=E 4=F 5=G  |  R: 0=Maj 1=Min 2=Maj7 3=7 4=Aug 5=Aug7",
        "Pinch: Reverb | [R] Reverb | [+/-] Octave | [Q] Quit",
    ]
    for i, text in enumerate(helps):
        cv2.putText(img,text,(x,y+i*20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,COL_MUTE,1,cv2.LINE_AA)

# ── MAIN ──
def main():
    print("="*60)
    print("  AI HARMONIZER — Finger Gesture Controller")
    print("="*60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'hand_landmarker.task')
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("  wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera."); sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    engine = HarmonizerEngine()
    detector = GestureDetector(model_path)
    engine.start()

    smooth_volume = 0.5
    octave = 4
    pinch_cooldown = 0.0
    left_fingers = 0
    right_fingers = 0

    print(f"[INFO] Camera: {fw}x{fh}")
    print("[INFO] Audio engine started.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = (frame * 0.7).astype(np.uint8)

            results = detector.process(rgb)

            pinch_detected = False

            if results.hand_landmarks and results.handedness:
                for hand_lm, hand_info in zip(results.hand_landmarks, results.handedness):
                    label = hand_info[0].category_name
                    detector.draw_landmarks(frame, hand_lm, fw, fh)
                    fc = detector.count_fingers(hand_lm, label)

                    if detector.is_pinch(hand_lm):
                        pinch_detected = True

                    if label == 'Left':
                        left_fingers = fc
                    elif label == 'Right':
                        right_fingers = fc

                    # Hand label on screen
                    cx, cy = detector.get_palm_center(hand_lm)
                    px, py = int(cx*fw), int(cy*fh)
                    cv2.putText(frame, f'{label} [{fc}]', (px-30, py-40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_ACCENT, 2, cv2.LINE_AA)

            # Pinch reverb toggle
            now = time.time()
            if pinch_detected and now - pinch_cooldown > 1.0:
                engine.reverb_on = not engine.reverb_on
                pinch_cooldown = now

            # Map finger counts to root + chord
            root_name, root_semi = ROOT_BY_FINGERS.get(left_fingers, ('A', 9))
            chord_name_s, chord_intervals = CHORD_BY_FINGERS.get(right_fingers, ('Major', [0,4,7]))

            root_freq = BASE_FREQ * (2 ** ((root_semi + (octave-4)*12) / 12.0))
            active_freqs = [root_freq * (2**(iv/12.0)) for iv in chord_intervals]
            root_label = f"{root_name}{octave}"
            playing_label = f"{root_name} {chord_name_s}"

            engine.set_notes(active_freqs, smooth_volume * 0.5)

            # ── Draw HUD ──
            draw_title(frame, fw)

            # Big center display of current chord
            draw_selection_display(frame, fw//2, fh//2 - 20,
                                   root_name, chord_name_s, COL_GLOW_L, COL_GLOW_R)

            # Finger indicators
            draw_finger_indicators(frame, left_fingers, 30, 90, "LEFT (Root)", COL_GLOW_L)
            draw_finger_indicators(frame, right_fingers, fw-200, 90, "RIGHT (Chord)", COL_GLOW_R)

            info = {
                'root': root_label, 'chord': chord_name_s,
                'playing': playing_label, 'octave': octave,
                'reverb': engine.reverb_on, 'volume': smooth_volume,
            }
            draw_status_panel(frame, info, fw//2-130, 75)

            draw_waveform(frame, engine.last_block,
                          fw//2-200, fh-100, 400, 70, COL_ACCENT2)
            draw_help_panel(frame, 15, fh-45)

            # Volume bar
            bx, by, bh = fw//2-140, fh-190, 80
            cv2.putText(frame,"VOL",(bx,by-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.35,COL_MUTE,1,cv2.LINE_AA)
            cv2.rectangle(frame,(bx,by),(bx+16,by+bh),COL_MUTE,1)
            filled = int(bh*smooth_volume)
            cv2.rectangle(frame,(bx,by+bh-filled),(bx+16,by+bh),COL_GREEN,-1)

            cv2.imshow('AI Harmonizer', frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): break
            elif key == ord('r'):
                engine.reverb_on = not engine.reverb_on
                print(f"[INFO] Reverb -> {'ON' if engine.reverb_on else 'OFF'}")
            elif key == ord('+'): octave = min(octave+1, 7)
            elif key == ord('-'): octave = max(octave-1, 2)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        engine.set_notes([], 0.0)
        time.sleep(0.1)
        engine.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Harmonizer stopped. Goodbye!")

if __name__ == '__main__':
    main()
