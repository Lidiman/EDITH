"""
AI Harmonizer — Hand Gesture Controlled Musical Synthesizer
============================================================
Uses MediaPipe hand-tracking to detect gestures via webcam and maps them
to real-time musical harmonies generated with numpy + sounddevice.

Controls
--------
• Number of raised fingers  → selects the chord / harmony
• Right hand Y-position     → controls pitch (octave shift)
• Left  hand Y-position     → controls volume
• Pinch gesture (thumb+index close) → toggles reverb effect
• Fist (0 fingers)           → mute

Keyboard
--------
  q / ESC  — quit
  m        — toggle scale (Major / Minor / Pentatonic)
  r        — toggle reverb
"""

import sys
import os
import time
import threading
import math

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import sounddevice as sd

# ──────────────────────────────────────────────────────────────────────
# AUDIO CONFIG
# ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
CHANNELS = 1

# ──────────────────────────────────────────────────────────────────────
# MUSICAL DATA
# ──────────────────────────────────────────────────────────────────────
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

SCALES = {
    'Major':       [0, 2, 4, 5, 7, 9, 11],
    'Minor':       [0, 2, 3, 5, 7, 8, 10],
    'Pentatonic':  [0, 2, 4, 7, 9],
}

CHORD_MAP = {
    0: [],
    1: [0],
    2: [0, 7],
    3: [0, 4, 7],
    4: [0, 4, 7, 11],
    5: [0, 4, 7, 11, 14],
}

CHORD_NAMES = {
    0: 'Mute',
    1: 'Unison',
    2: 'Power',
    3: 'Triad',
    4: 'Maj7',
    5: 'Maj9',
}

BASE_FREQ = 261.63

# ──────────────────────────────────────────────────────────────────────
# COLORS (BGR)
# ──────────────────────────────────────────────────────────────────────
COL_BG       = (20, 20, 20)
COL_ACCENT   = (255, 160, 50)
COL_ACCENT2  = (80, 220, 255)
COL_MUTE     = (80, 80, 80)
COL_TEXT     = (240, 240, 240)
COL_GREEN    = (100, 220, 100)
COL_PURPLE   = (200, 100, 255)

VIS_COLORS = [
    (255, 100, 50),
    (255, 180, 50),
    (50, 220, 255),
    (100, 255, 180),
    (200, 100, 255),
]

# ──────────────────────────────────────────────────────────────────────
# AUDIO ENGINE
# ──────────────────────────────────────────────────────────────────────
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
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype='float32',
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()

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
                if f <= 0:
                    continue
                detune = 1.0 + (i * 0.001)
                phase_inc = 2.0 * np.pi * f * detune * t + self.phases[i]
                wave = np.sin(phase_inc)
                wave += 0.15 * np.sin(2 * phase_inc)
                signal += wave
                self.phases[i] += 2.0 * np.pi * f * detune * frames / SAMPLE_RATE
                self.phases[i] %= (2.0 * np.pi)
            if len(freqs) > 0:
                signal /= max(len(freqs), 1)

        signal *= ramp

        if self.reverb_on:
            delay_samples = int(0.08 * SAMPLE_RATE)
            for i in range(frames):
                idx = (self.reverb_idx + i) % len(self.reverb_buf)
                delayed_idx = (idx - delay_samples) % len(self.reverb_buf)
                wet = self.reverb_buf[delayed_idx] * 0.4
                out_sample = signal[i] + wet
                self.reverb_buf[idx] = out_sample
                signal[i] = out_sample * 0.8 + signal[i] * 0.2
            self.reverb_idx = (self.reverb_idx + frames) % len(self.reverb_buf)

        signal = np.tanh(signal * 0.8)
        block = signal.astype(np.float32)
        self.last_block = block.copy()
        outdata[:, 0] = block


# ──────────────────────────────────────────────────────────────────────
# HAND GESTURE DETECTOR  (MediaPipe Tasks API)
# ──────────────────────────────────────────────────────────────────────
class GestureDetector:
    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]

    # Hand connections for drawing
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17),
    ]

    def __init__(self, model_path):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.6,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_rgb):
        """Process an RGB numpy frame and return the result."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self.landmarker.detect(mp_image)

    @staticmethod
    def count_fingers(landmarks, handedness_label):
        lm = landmarks
        count = 0
        if handedness_label == 'Right':
            if lm[4].x < lm[3].x:
                count += 1
        else:
            if lm[4].x > lm[3].x:
                count += 1
        for tip, pip in zip(GestureDetector.FINGER_TIPS, GestureDetector.FINGER_PIPS):
            if lm[tip].y < lm[pip].y:
                count += 1
        return count

    @staticmethod
    def get_palm_center(landmarks):
        lm = landmarks
        cx = (lm[0].x + lm[5].x + lm[17].x) / 3
        cy = (lm[0].y + lm[5].y + lm[17].y) / 3
        return cx, cy

    @staticmethod
    def is_pinch(landmarks):
        lm = landmarks
        dist = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        return dist < 0.05

    @staticmethod
    def draw_landmarks(frame, landmarks, w, h):
        pts = []
        for lm in landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            pts.append((px, py))
            cv2.circle(frame, (px, py), 4, COL_ACCENT, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 4, COL_TEXT, 1, cv2.LINE_AA)
        for a, b in GestureDetector.HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], COL_ACCENT2, 2, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────
# HUD DRAWING
# ──────────────────────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, radius=15, thickness=-1, alpha=0.6):
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_waveform(img, waveform, x, y, w, h, color=COL_ACCENT2):
    if len(waveform) == 0:
        return
    step = max(1, len(waveform) // w)
    samples = waveform[::step][:w]
    mid_y = y + h // 2
    pts = []
    for i, s in enumerate(samples):
        px = x + i
        py = int(mid_y - s * (h // 2) * 0.9)
        py = max(y, min(y + h, py))
        pts.append((px, py))
    if len(pts) > 1:
        for thickness, alpha_mult in [(5, 0.2), (3, 0.4), (1, 1.0)]:
            c = tuple(int(ch * alpha_mult) for ch in color)
            cv2.polylines(img, [np.array(pts)], False, c, thickness, cv2.LINE_AA)


def draw_frequency_bars(img, freqs, x, y, w, h):
    if not freqs:
        return
    bar_w = max(4, w // (len(freqs) * 2))
    gap = max(2, (w - bar_w * len(freqs)) // (len(freqs) + 1))
    cx = x + gap
    for i, f in enumerate(freqs):
        if f <= 0:
            continue
        bar_h = int(np.interp(f, [60, 2000], [h * 0.2, h * 0.9]))
        bar_h = min(bar_h, h)
        color = VIS_COLORS[i % len(VIS_COLORS)]
        for row in range(bar_h):
            ratio = row / max(bar_h, 1)
            c = tuple(int(ch * (0.3 + 0.7 * ratio)) for ch in color)
            py = y + h - row
            cv2.line(img, (cx, py), (cx + bar_w, py), c, 1)
        note_idx = int(round(12 * math.log2(f / 261.63))) % 12
        note_name = NOTE_NAMES[note_idx]
        cv2.putText(img, note_name, (cx, y + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TEXT, 1, cv2.LINE_AA)
        cx += bar_w + gap


def draw_status_panel(img, info, x, y):
    panel_w, panel_h = 280, 200
    draw_rounded_rect(img, (x, y), (x + panel_w, y + panel_h), COL_BG, alpha=0.75)
    ty = y + 30
    line_h = 28
    items = [
        ('Scale',  info.get('scale', '-'),      COL_ACCENT),
        ('Chord',  info.get('chord', '-'),       COL_ACCENT2),
        ('Root',   info.get('root', '-'),         COL_GREEN),
        ('Octave', str(info.get('octave', '-')),  COL_PURPLE),
        ('Reverb', 'ON' if info.get('reverb') else 'OFF',
         COL_GREEN if info.get('reverb') else COL_MUTE),
        ('Vol',    f"{info.get('volume', 0):.0%}", COL_TEXT),
    ]
    for label, value, color in items:
        cv2.putText(img, f'{label}:', (x + 15, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_MUTE, 1, cv2.LINE_AA)
        cv2.putText(img, str(value), (x + 110, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        ty += line_h


def draw_help_panel(img, x, y):
    helps = [
        "Fingers: 0=Mute  1=Note  2=Power  3=Triad  4=Maj7  5=Maj9",
        "Right hand Y: Pitch  |  Left hand Y: Volume",
        "Pinch: Toggle Reverb  |  [M] Scale  [R] Reverb  [Q] Quit",
    ]
    for i, text in enumerate(helps):
        cv2.putText(img, text, (x, y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_MUTE, 1, cv2.LINE_AA)


def draw_title(img, w):
    cv2.putText(img, "AI  HARMONIZER", (w // 2 - 140, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL_ACCENT, 2, cv2.LINE_AA)
    cv2.putText(img, "Hand Gesture Control", (w // 2 - 108, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_ACCENT2, 1, cv2.LINE_AA)


def draw_finger_indicators(img, count, x, y):
    for i in range(5):
        cx = x + i * 30
        color = COL_GREEN if i < count else COL_MUTE
        cv2.circle(img, (cx, y), 10, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, y), 10, COL_TEXT, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AI HARMONIZER — Hand Gesture Musical Controller")
    print("=" * 60)
    print()
    print("  Controls:")
    print("  • Show fingers to select chord type")
    print("  • Move right hand up/down to change pitch")
    print("  • Move left  hand up/down to change volume")
    print("  • Pinch (thumb+index) to toggle reverb")
    print("  • Press 'm' to switch scale, 'r' for reverb, 'q' to quit")
    print()

    # Resolve model path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'hand_landmarker.task')
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print("  Download it with:")
        print("  wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    engine = HarmonizerEngine()
    detector = GestureDetector(model_path)
    engine.start()

    scale_names = list(SCALES.keys())
    current_scale_idx = 0
    pinch_toggle_cooldown = 0.0
    smooth_volume = 0.5
    smooth_octave_shift = 0.0

    print(f"[INFO] Camera resolution: {frame_w}x{frame_h}")
    print("[INFO] Audio engine started. Listening for gestures...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = (frame * 0.75).astype(np.uint8)

            results = detector.process(rgb)

            finger_count = 0
            volume = smooth_volume
            octave_shift = smooth_octave_shift
            active_freqs = []
            pinch_detected = False
            current_scale = scale_names[current_scale_idx]
            chord_name = CHORD_NAMES.get(0, 'Mute')
            root_note = '-'

            if results.hand_landmarks and results.handedness:
                for hand_lm, hand_info in zip(results.hand_landmarks, results.handedness):
                    # MediaPipe Tasks returns handedness as list of Category
                    label = hand_info[0].category_name
                    landmarks = hand_lm

                    detector.draw_landmarks(frame, landmarks, frame_w, frame_h)

                    fc = detector.count_fingers(landmarks, label)
                    cx, cy = detector.get_palm_center(landmarks)

                    if detector.is_pinch(landmarks):
                        pinch_detected = True

                    if label == 'Right':
                        finger_count = fc
                        octave_shift = np.interp(cy, [0.1, 0.9], [2.0, -1.0])
                    elif label == 'Left':
                        volume = np.interp(cy, [0.1, 0.9], [1.0, 0.0])

                    px, py = int(cx * frame_w), int(cy * frame_h)
                    cv2.putText(frame, f'{label} [{fc}]', (px - 30, py - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_ACCENT, 2, cv2.LINE_AA)

            smooth_volume += (volume - smooth_volume) * 0.15
            smooth_octave_shift += (octave_shift - smooth_octave_shift) * 0.1

            now = time.time()
            if pinch_detected and now - pinch_toggle_cooldown > 1.0:
                engine.reverb_on = not engine.reverb_on
                pinch_toggle_cooldown = now

            chord_intervals = CHORD_MAP.get(finger_count, [])
            chord_name = CHORD_NAMES.get(finger_count, 'Mute')
            scale_intervals = SCALES[current_scale]

            if chord_intervals:
                root_semitone = scale_intervals[
                    int(np.interp(smooth_octave_shift, [-1, 2],
                                  [0, len(scale_intervals) - 1]))
                ]
                octave = 4 + int(smooth_octave_shift)
                root_freq = BASE_FREQ * (2 ** ((root_semitone + (octave - 4) * 12) / 12.0))
                root_note = NOTE_NAMES[root_semitone % 12] + str(octave)
                active_freqs = [
                    root_freq * (2 ** (interval / 12.0))
                    for interval in chord_intervals
                ]
            else:
                root_note = '-'

            engine.set_notes(active_freqs, smooth_volume * 0.5)

            # ── HUD ──
            draw_title(frame, frame_w)
            info = {
                'scale': current_scale,
                'chord': chord_name,
                'root': root_note,
                'octave': 4 + int(smooth_octave_shift),
                'reverb': engine.reverb_on,
                'volume': smooth_volume,
            }
            draw_status_panel(frame, info, 15, 75)
            draw_finger_indicators(frame, finger_count, frame_w - 180, 90)
            draw_waveform(frame, engine.last_block,
                          frame_w // 2 - 200, frame_h - 120, 400, 80, COL_ACCENT2)
            draw_frequency_bars(frame, active_freqs,
                                frame_w - 240, frame_h - 220, 220, 140)

            bar_x, bar_y, bar_h = 25, frame_h - 160, 120
            cv2.putText(frame, "VOL", (bar_x, bar_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_MUTE, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 20, bar_y + bar_h), COL_MUTE, 1)
            filled = int(bar_h * smooth_volume)
            cv2.rectangle(frame, (bar_x, bar_y + bar_h - filled),
                          (bar_x + 20, bar_y + bar_h), COL_GREEN, -1)

            draw_help_panel(frame, 15, frame_h - 25)

            cv2.imshow('AI Harmonizer', frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('m'):
                current_scale_idx = (current_scale_idx + 1) % len(scale_names)
                print(f"[INFO] Scale -> {scale_names[current_scale_idx]}")
            elif key == ord('r'):
                engine.reverb_on = not engine.reverb_on
                print(f"[INFO] Reverb -> {'ON' if engine.reverb_on else 'OFF'}")

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
