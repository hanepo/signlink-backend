"""
BIM (Malaysian Sign Language) Gesture Detector
Implements rule-based detection for 15 BIM signs using MediaPipe hand landmarks
"""

import math
import numpy as np

class BIMGestureDetector:
    """Rule-based detector for 15 BIM gestures"""

    # Landmark indices (MediaPipe Hand)
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(self):
        self.gesture_names = {
            'SALAM': 'Salam (Hello)',
            'TERIMA KASIH': 'Terima Kasih (Thank You)',
            'TOLONG': 'Tolong (Help)',
            'SAYA': 'Saya (I/Me)',
            'OK': 'OK (Good)',
            'YA': 'Ya (Yes)',
            'TIDAK': 'Tidak (No)',
            'LIHAT': 'Lihat (Look/See)',
            'TUNGGU': 'Tunggu (Wait)',
            'PERGI': 'Pergi (Go)',
            'DATANG': 'Datang (Come)',
            'AWAK': 'Awak (You)',
            'DIA': 'Dia (He/She)',
            'KITA': 'Kita (We)',
            'MAAF': 'Maaf (Sorry)'
        }

    def distance(self, p1, p2):
        """Calculate Euclidean distance between two 3D points"""
        return math.sqrt((p1['x'] - p2['x'])**2 +
                        (p1['y'] - p2['y'])**2 +
                        (p1['z'] - p2['z'])**2)

    def is_finger_extended(self, landmarks, finger_base, finger_tip):
        """Check if a finger is extended (tip further from wrist than base)"""
        wrist = landmarks[self.WRIST]
        base = landmarks[finger_base]
        tip = landmarks[finger_tip]

        dist_base = self.distance(wrist, base)
        dist_tip = self.distance(wrist, tip)

        return dist_tip > dist_base * 1.2  # 20% threshold

    def is_finger_curled(self, landmarks, finger_mcp, finger_tip):
        """Check if a finger is curled (tip close to MCP)"""
        mcp = landmarks[finger_mcp]
        tip = landmarks[finger_tip]
        wrist = landmarks[self.WRIST]

        dist_tip_mcp = self.distance(tip, mcp)
        dist_wrist_mcp = self.distance(wrist, mcp)

        return dist_tip_mcp < dist_wrist_mcp * 0.5

    def finger_tip_distance(self, landmarks, tip1_idx, tip2_idx):
        """Distance between two fingertips"""
        return self.distance(landmarks[tip1_idx], landmarks[tip2_idx])

    def get_finger_angle(self, landmarks, finger_tip):
        """Get the angle of finger pointing direction (in degrees)"""
        wrist = landmarks[self.WRIST]
        tip = landmarks[finger_tip]

        dx = tip['x'] - wrist['x']
        dy = tip['y'] - wrist['y']

        angle = math.degrees(math.atan2(dy, dx))
        return angle

    def detect_gesture(self, landmarks):
        """
        Detect BIM gesture from hand landmarks

        Args:
            landmarks: List of 21 hand landmarks with x, y, z coordinates

        Returns:
            Detected gesture name or None
        """
        if not landmarks or len(landmarks) != 21:
            return None

        # Check each gesture in priority order
        # Some gestures need to be checked first to avoid confusion

        # 1. OK - specific thumb-index circle with middle extended
        if self._is_ok_gesture(landmarks):
            return 'OK'

        # 2. KITA - Y-shape (thumb + pinky extended)
        if self._is_kita_gesture(landmarks):
            return 'KITA'

        # 3. YA - Thumbs up
        if self._is_ya_gesture(landmarks):
            return 'YA'

        # 4. MAAF - Fist with thumb crossing
        if self._is_maaf_gesture(landmarks):
            return 'MAAF'

        # 5. TOLONG - Closed fist
        if self._is_tolong_gesture(landmarks):
            return 'TOLONG'

        # 6. TIDAK - Index + middle together/crossing
        if self._is_tidak_gesture(landmarks):
            return 'TIDAK'

        # 7. LIHAT - V sign (index + middle spread)
        if self._is_lihat_gesture(landmarks):
            return 'LIHAT'

        # 8. TUNGGU - Index + middle together (parallel)
        if self._is_tunggu_gesture(landmarks):
            return 'TUNGGU'

        # 9. Pointing gestures (check angle)
        pointing = self._detect_pointing_direction(landmarks)
        if pointing:
            return pointing

        # 10. TERIMA KASIH - Flat palm, fingers together
        if self._is_terima_kasih_gesture(landmarks):
            return 'TERIMA KASIH'

        # 11. SALAM - Open palm, fingers spread
        if self._is_salam_gesture(landmarks):
            return 'SALAM'

        return None

    def _is_salam_gesture(self, landmarks):
        """Open palm facing camera, fingers extended with natural spread"""
        # All fingers extended
        index_ext = self.is_finger_extended(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_ext = self.is_finger_extended(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_ext = self.is_finger_extended(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_ext = self.is_finger_extended(landmarks, self.PINKY_MCP, self.PINKY_TIP)

        if not (index_ext and middle_ext and ring_ext and pinky_ext):
            return False

        # Fingers should have some spread (not too close together)
        index_middle_dist = self.finger_tip_distance(landmarks, self.INDEX_TIP, self.MIDDLE_TIP)
        middle_ring_dist = self.finger_tip_distance(landmarks, self.MIDDLE_TIP, self.RING_TIP)

        # Check for natural spread between fingers
        avg_spread = (index_middle_dist + middle_ring_dist) / 2

        return avg_spread > 0.03  # Threshold for spread

    def _is_terima_kasih_gesture(self, landmarks):
        """Flat palm with fingers close together"""
        # All fingers extended
        index_ext = self.is_finger_extended(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_ext = self.is_finger_extended(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_ext = self.is_finger_extended(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_ext = self.is_finger_extended(landmarks, self.PINKY_MCP, self.PINKY_TIP)

        if not (index_ext and middle_ext and ring_ext and pinky_ext):
            return False

        # Fingers should be close together
        index_middle_dist = self.finger_tip_distance(landmarks, self.INDEX_TIP, self.MIDDLE_TIP)
        middle_ring_dist = self.finger_tip_distance(landmarks, self.MIDDLE_TIP, self.RING_TIP)
        ring_pinky_dist = self.finger_tip_distance(landmarks, self.RING_TIP, self.PINKY_TIP)

        # All adjacent fingers close together
        return (index_middle_dist < 0.03 and
                middle_ring_dist < 0.03 and
                ring_pinky_dist < 0.03)

    def _is_tolong_gesture(self, landmarks):
        """Closed fist - all fingers curled"""
        index_curled = self.is_finger_curled(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_curled = self.is_finger_curled(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_curled = self.is_finger_curled(landmarks, self.PINKY_MCP, self.PINKY_TIP)

        # Thumb should not be extended upward (would be YA)
        thumb_up = self.is_finger_extended(landmarks, self.THUMB_CMC, self.THUMB_TIP)
        wrist = landmarks[self.WRIST]
        thumb_tip = landmarks[self.THUMB_TIP]

        # Check if thumb is pointing upward
        thumb_vertical = abs(thumb_tip['y'] - wrist['y']) > abs(thumb_tip['x'] - wrist['x'])

        return (index_curled and middle_curled and ring_curled and pinky_curled and
                not (thumb_up and thumb_vertical))

    def _is_saya_gesture(self, landmarks):
        """Index finger extended, others curled, pointing toward chest"""
        index_ext = self.is_finger_extended(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_curled = self.is_finger_curled(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_curled = self.is_finger_curled(landmarks, self.PINKY_MCP, self.PINKY_TIP)

        return index_ext and middle_curled and ring_curled and pinky_curled

    def _is_ok_gesture(self, landmarks):
        """Thumb tip touches index tip (circle), middle extended"""
        # Check if thumb and index tips are close (forming circle)
        thumb_index_dist = self.finger_tip_distance(landmarks, self.THUMB_TIP, self.INDEX_TIP)

        # Middle finger should be extended
        middle_ext = self.is_finger_extended(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)

        # Ring and pinky should be somewhat curled
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)

        return thumb_index_dist < 0.05 and middle_ext

    def _is_ya_gesture(self, landmarks):
        """Thumbs up - thumb extended upward"""
        # Thumb should be extended
        thumb_ext = self.is_finger_extended(landmarks, self.THUMB_CMC, self.THUMB_TIP)

        # Other fingers should be curled
        index_curled = self.is_finger_curled(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_curled = self.is_finger_curled(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_curled = self.is_finger_curled(landmarks, self.PINKY_MCP, self.PINKY_TIP)

        # Check if thumb is pointing upward (vertical)
        wrist = landmarks[self.WRIST]
        thumb_tip = landmarks[self.THUMB_TIP]

        # Thumb should be more vertical than horizontal
        dy = abs(thumb_tip['y'] - wrist['y'])
        dx = abs(thumb_tip['x'] - wrist['x'])

        thumb_vertical = dy > dx * 0.7  # Mostly vertical

        return (thumb_ext and thumb_vertical and
                index_curled and middle_curled and ring_curled and pinky_curled)

    def _is_tidak_gesture(self, landmarks):
        """Index and middle together/crossing, ring & pinky curled"""
        index_ext = self.is_finger_extended(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_ext = self.is_finger_extended(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_curled = self.is_finger_curled(landmarks, self.PINKY_MCP, self.PINKY_TIP)

        if not (index_ext and middle_ext and ring_curled and pinky_curled):
            return False

        # Index and middle should be very close (crossing or touching)
        index_middle_dist = self.finger_tip_distance(landmarks, self.INDEX_TIP, self.MIDDLE_TIP)

        return index_middle_dist < 0.04  # Very close together

    def _is_lihat_gesture(self, landmarks):
        """V sign - Index and middle extended and spread apart"""
        index_ext = self.is_finger_extended(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_ext = self.is_finger_extended(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_curled = self.is_finger_curled(landmarks, self.PINKY_MCP, self.PINKY_TIP)

        if not (index_ext and middle_ext and ring_curled and pinky_curled):
            return False

        # Index and middle should be spread apart (V shape)
        index_middle_dist = self.finger_tip_distance(landmarks, self.INDEX_TIP, self.MIDDLE_TIP)

        return index_middle_dist > 0.06  # Clear V spread

    def _is_tunggu_gesture(self, landmarks):
        """Index and middle extended together (parallel), not spread"""
        index_ext = self.is_finger_extended(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_ext = self.is_finger_extended(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_curled = self.is_finger_curled(landmarks, self.PINKY_MCP, self.PINKY_TIP)

        if not (index_ext and middle_ext and ring_curled and pinky_curled):
            return False

        # Index and middle should be close but not too close (parallel)
        index_middle_dist = self.finger_tip_distance(landmarks, self.INDEX_TIP, self.MIDDLE_TIP)

        return 0.04 <= index_middle_dist <= 0.06  # Parallel, not spread

    def _detect_pointing_direction(self, landmarks):
        """Detect pointing direction for PERGI, DATANG, AWAK, DIA"""
        # Index extended, others curled
        index_ext = self.is_finger_extended(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_curled = self.is_finger_curled(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_curled = self.is_finger_curled(landmarks, self.PINKY_MCP, self.PINKY_TIP)

        if not (index_ext and middle_curled and ring_curled and pinky_curled):
            return None

        # Get pointing angle
        angle = self.get_finger_angle(landmarks, self.INDEX_TIP)

        # Normalize angle to 0-360
        if angle < 0:
            angle += 360

        # Determine direction based on angle
        # Right: -45 to 45 degrees (or 315-360 and 0-45)
        # Left: 135 to 225 degrees
        # Up: 225 to 315 degrees (pointing up = negative y)
        # Down: 45 to 135 degrees (pointing down = positive y)
        # Diagonal: other ranges

        # Adjust for coordinate system (y increases downward in image)
        wrist = landmarks[self.WRIST]
        index_tip = landmarks[self.INDEX_TIP]

        dx = index_tip['x'] - wrist['x']
        dy = index_tip['y'] - wrist['y']

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        # Determine if mostly horizontal, vertical, or diagonal
        if abs_dx > abs_dy * 1.5:  # Mostly horizontal
            if dx > 0:
                return 'PERGI'  # Pointing right
            else:
                return 'DATANG'  # Pointing left
        elif abs_dy > abs_dx * 1.5:  # Mostly vertical
            return 'AWAK'  # Pointing up or down
        else:  # Diagonal
            return 'DIA'  # Pointing diagonally

        return None

    def _is_kita_gesture(self, landmarks):
        """Y-shape: Thumb and pinky extended, others curled"""
        thumb_ext = self.is_finger_extended(landmarks, self.THUMB_CMC, self.THUMB_TIP)
        pinky_ext = self.is_finger_extended(landmarks, self.PINKY_MCP, self.PINKY_TIP)

        index_curled = self.is_finger_curled(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_curled = self.is_finger_curled(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)

        return (thumb_ext and pinky_ext and
                index_curled and middle_curled and ring_curled)

    def _is_maaf_gesture(self, landmarks):
        """Fist with thumb crossing over index MCP"""
        # All fingers curled
        index_curled = self.is_finger_curled(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_curled = self.is_finger_curled(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_curled = self.is_finger_curled(landmarks, self.PINKY_MCP, self.PINKY_TIP)

        if not (index_curled and middle_curled and ring_curled and pinky_curled):
            return False

        # Thumb should be close to index MCP (crossing over)
        thumb_tip = landmarks[self.THUMB_TIP]
        index_mcp = landmarks[self.INDEX_MCP]

        thumb_index_dist = self.distance(thumb_tip, index_mcp)

        # Also check thumb is not pointing upward (would be TOLONG)
        wrist = landmarks[self.WRIST]
        thumb_up_dist = abs(thumb_tip['y'] - wrist['y'])
        thumb_side_dist = abs(thumb_tip['x'] - wrist['x'])

        thumb_horizontal = thumb_side_dist > thumb_up_dist

        return thumb_index_dist < 0.08 and thumb_horizontal
