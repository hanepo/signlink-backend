"""
Universal Hand Gesture Detector
Rule-based detection for 17 common hand gestures (emojis)
"""

import math

class UniversalGestureDetector:
    """Rule-based detector for 17 universal hand gestures"""

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
            'THUMBS_UP': 'Thumbs Up',
            'THUMBS_DOWN': 'Thumbs Down',
            'STOP': 'Stop',
            'PEACE': 'Peace / Victory',
            'PINCH_SMALL': 'Pinch Small',
            'PINCH_BIG': 'Pinch Big',
            'POINTING': 'Pointing',
            'OK': 'OK Sign',
            'FIST': 'Fist',
            'ILY': 'I Love You',
            'ONE': 'One',
            'TWO': 'Two Fingers',
            'ROCK': 'Rock Sign',
            'PALM_UP': 'Palm Up',
            'HAND_DOWN': 'Hand Down',
            'HAND_FORWARD': 'Hand Forward',
            'CALL_ME': 'Call Me'
        }

    def distance(self, p1, p2):
        """Calculate Euclidean distance between two 3D points"""
        return math.sqrt((p1['x'] - p2['x'])**2 +
                        (p1['y'] - p2['y'])**2 +
                        (p1['z'] - p2['z'])**2)

    def is_finger_extended(self, landmarks, finger_base, finger_tip):
        """Check if a finger is extended"""
        wrist = landmarks[self.WRIST]
        base = landmarks[finger_base]
        tip = landmarks[finger_tip]
        
        dist_base = self.distance(wrist, base)
        dist_tip = self.distance(wrist, tip)
        
        return dist_tip > dist_base * 1.2

    def is_finger_curled(self, landmarks, finger_mcp, finger_tip):
        """Check if a finger is curled"""
        mcp = landmarks[finger_mcp]
        tip = landmarks[finger_tip]
        wrist = landmarks[self.WRIST]
        
        dist_tip_mcp = self.distance(tip, mcp)
        dist_wrist_mcp = self.distance(wrist, mcp)
        
        return dist_tip_mcp < dist_wrist_mcp * 0.5

    def detect_gesture(self, landmarks):
        """
        Detect gesture from hand landmarks
        Returns gesture name or None
        """
        if not landmarks or len(landmarks) != 21:
            return None

        # Check in priority order (most specific first)
        if self._is_peace_sign(landmarks):
            return 'PEACE'
        if self._is_two(landmarks):
            return 'TWO'
        if self._is_one(landmarks):
            return 'ONE'
        if self._is_call_me(landmarks):
            return 'CALL_ME'
        if self._is_rock(landmarks):
            return 'ROCK'
        if self._is_ily(landmarks):
            return 'ILY'
        if self._is_pointing(landmarks):
            return 'POINTING'
        if self._is_fist(landmarks):
            return 'FIST'
        if self._is_palm_up(landmarks):
            return 'PALM_UP'
        if self._is_hand_down(landmarks):
            return 'HAND_DOWN'
        if self._is_hand_forward(landmarks):
            return 'HAND_FORWARD'
        if self._is_stop(landmarks):
            return 'STOP'
        if self._is_pinch_small(landmarks):
            return 'PINCH_SMALL'
        if self._is_pinch_big(landmarks):
            return 'PINCH_BIG'
        if self._is_ok_sign(landmarks):
            return 'OK'
        if self._is_thumbs_up(landmarks):
            return 'THUMBS_UP'
        if self._is_thumbs_down(landmarks):
            return 'THUMBS_DOWN'
        
        return None

    def _is_thumbs_up(self, landmarks):
        """Thumb pointing up, other fingers curled"""
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        wrist = landmarks[self.WRIST]
        
        # Thumb extended upward
        thumb_up = thumb_tip['y'] < thumb_ip['y'] - 0.05
        thumb_high = thumb_tip['y'] < wrist['y'] - 0.1
        
        # Other fingers curled
        index_curled = landmarks[self.INDEX_TIP]['y'] > landmarks[self.INDEX_MCP]['y'] - 0.05
        middle_curled = landmarks[self.MIDDLE_TIP]['y'] > landmarks[self.MIDDLE_MCP]['y'] - 0.05
        
        return thumb_up and thumb_high and index_curled and middle_curled

    def _is_thumbs_down(self, landmarks):
        """Thumb pointing down, other fingers curled"""
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        wrist = landmarks[self.WRIST]
        
        # Thumb pointing down
        thumb_down = thumb_tip['y'] > thumb_ip['y'] + 0.05
        thumb_low = thumb_tip['y'] > wrist['y'] + 0.05
        
        # Other fingers curled
        index_curled = landmarks[self.INDEX_TIP]['y'] > landmarks[self.INDEX_MCP]['y'] - 0.05
        
        return thumb_down and thumb_low and index_curled

    def _is_peace_sign(self, landmarks):
        """Index and middle extended in V, others curled"""
        index_ext = self.is_finger_extended(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_ext = self.is_finger_extended(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_curled = self.is_finger_curled(landmarks, self.PINKY_MCP, self.PINKY_TIP)
        
        if not (index_ext and middle_ext and ring_curled and pinky_curled):
            return False
        
        # Fingers spread in V shape
        fingers_spread = abs(landmarks[self.INDEX_TIP]['x'] - landmarks[self.MIDDLE_TIP]['x']) > 0.03
        return fingers_spread

    def _is_ok_sign(self, landmarks):
        """Thumb and index touching in circle, others extended"""
        thumb_index_dist = self.distance(landmarks[self.THUMB_TIP], landmarks[self.INDEX_TIP])
        circle_formed = thumb_index_dist < 0.05
        
        # Other fingers extended
        middle_ext = landmarks[self.MIDDLE_TIP]['y'] < landmarks[self.MIDDLE_MCP]['y'] - 0.04
        ring_ext = landmarks[self.RING_TIP]['y'] < landmarks[self.RING_MCP]['y'] - 0.04
        
        return circle_formed and middle_ext and ring_ext

    def _is_pinch_small(self, landmarks):
        """Thumb and index almost touching, small gap"""
        thumb_index_dist = self.distance(landmarks[self.THUMB_TIP], landmarks[self.INDEX_TIP])
        
        # Very small distance
        pinch_tight = 0.02 < thumb_index_dist < 0.04
        
        # Other fingers curled or neutral
        middle_not_ext = landmarks[self.MIDDLE_TIP]['y'] > landmarks[self.MIDDLE_MCP]['y'] - 0.08
        
        return pinch_tight and middle_not_ext

    def _is_pinch_big(self, landmarks):
        """All fingertips gathered together"""
        # Calculate center of all fingertips
        center_x = (landmarks[self.THUMB_TIP]['x'] + landmarks[self.INDEX_TIP]['x'] + 
                   landmarks[self.MIDDLE_TIP]['x'] + landmarks[self.RING_TIP]['x'] + 
                   landmarks[self.PINKY_TIP]['x']) / 5
        center_y = (landmarks[self.THUMB_TIP]['y'] + landmarks[self.INDEX_TIP]['y'] + 
                   landmarks[self.MIDDLE_TIP]['y'] + landmarks[self.RING_TIP]['y'] + 
                   landmarks[self.PINKY_TIP]['y']) / 5
        
        center = {'x': center_x, 'y': center_y, 'z': 0}
        
        # All tips close to center
        all_close = all(
            self.distance(landmarks[tip], center) < 0.08
            for tip in [self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        )
        
        return all_close

    def _is_pointing(self, landmarks):
        """Index extended, others curled"""
        index_ext = self.is_finger_extended(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_curled = self.is_finger_curled(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_curled = self.is_finger_curled(landmarks, self.PINKY_MCP, self.PINKY_TIP)
        
        return index_ext and middle_curled and ring_curled and pinky_curled

    def _is_one(self, landmarks):
        """Index pointing up, others curled"""
        index_ext = self.is_finger_extended(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_curled = self.is_finger_curled(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_curled = self.is_finger_curled(landmarks, self.PINKY_MCP, self.PINKY_TIP)
        
        if not (index_ext and middle_curled and ring_curled and pinky_curled):
            return False
        
        # Index should be more vertical
        index_vertical = abs(landmarks[self.INDEX_TIP]['y'] - landmarks[self.WRIST]['y']) > 0.15
        return index_vertical

    def _is_two(self, landmarks):
        """Index and middle up, others curled"""
        index_ext = self.is_finger_extended(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_ext = self.is_finger_extended(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_curled = self.is_finger_curled(landmarks, self.PINKY_MCP, self.PINKY_TIP)
        
        if not (index_ext and middle_ext and ring_curled and pinky_curled):
            return False
        
        # Fingers should be close together (not peace sign)
        fingers_close = abs(landmarks[self.INDEX_TIP]['x'] - landmarks[self.MIDDLE_TIP]['x']) < 0.03
        return fingers_close

    def _is_call_me(self, landmarks):
        """Thumb and pinky extended (phone shape)"""
        thumb_ext = self.is_finger_extended(landmarks, self.THUMB_CMC, self.THUMB_TIP)
        pinky_ext = self.is_finger_extended(landmarks, self.PINKY_MCP, self.PINKY_TIP)
        index_curled = self.is_finger_curled(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_curled = self.is_finger_curled(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)
        
        return thumb_ext and pinky_ext and index_curled and middle_curled and ring_curled

    def _is_palm_up(self, landmarks):
        """Open hand palm facing up"""
        # All fingers extended
        index_ext = self.is_finger_extended(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_ext = self.is_finger_extended(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_ext = self.is_finger_extended(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_ext = self.is_finger_extended(landmarks, self.PINKY_MCP, self.PINKY_TIP)
        
        if not (index_ext and middle_ext and ring_ext and pinky_ext):
            return False
        
        # Palm orientation (z values indicate depth)
        palm_facing_up = landmarks[self.MIDDLE_MCP]['z'] < landmarks[self.MIDDLE_TIP]['z']
        return palm_facing_up

    def _is_hand_down(self, landmarks):
        """Open hand palm facing down"""
        index_ext = self.is_finger_extended(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_ext = self.is_finger_extended(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_ext = self.is_finger_extended(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_ext = self.is_finger_extended(landmarks, self.PINKY_MCP, self.PINKY_TIP)
        
        if not (index_ext and middle_ext and ring_ext and pinky_ext):
            return False
        
        # Palm facing down
        palm_down = landmarks[self.MIDDLE_MCP]['z'] > landmarks[self.MIDDLE_TIP]['z']
        return palm_down

    def _is_hand_forward(self, landmarks):
        """Open palm pushing forward"""
        index_ext = self.is_finger_extended(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_ext = self.is_finger_extended(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_ext = self.is_finger_extended(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_ext = self.is_finger_extended(landmarks, self.PINKY_MCP, self.PINKY_TIP)
        
        if not (index_ext and middle_ext and ring_ext and pinky_ext):
            return False
        
        # Fingers more vertical
        vertical = landmarks[self.MIDDLE_TIP]['y'] < landmarks[self.WRIST]['y']
        return vertical

    def _is_stop(self, landmarks):
        """Open palm facing forward, vertical"""
        return self._is_hand_forward(landmarks)

    def _is_fist(self, landmarks):
        """All fingers curled"""
        index_curled = self.is_finger_curled(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        middle_curled = self.is_finger_curled(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)
        pinky_curled = self.is_finger_curled(landmarks, self.PINKY_MCP, self.PINKY_TIP)
        
        return index_curled and middle_curled and ring_curled and pinky_curled

    def _is_ily(self, landmarks):
        """Thumb, index, and pinky extended"""
        thumb_ext = self.is_finger_extended(landmarks, self.THUMB_CMC, self.THUMB_TIP)
        index_ext = self.is_finger_extended(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        pinky_ext = self.is_finger_extended(landmarks, self.PINKY_MCP, self.PINKY_TIP)
        middle_curled = self.is_finger_curled(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)
        
        return thumb_ext and index_ext and pinky_ext and middle_curled and ring_curled

    def _is_rock(self, landmarks):
        """Index and pinky extended (horns)"""
        index_ext = self.is_finger_extended(landmarks, self.INDEX_MCP, self.INDEX_TIP)
        pinky_ext = self.is_finger_extended(landmarks, self.PINKY_MCP, self.PINKY_TIP)
        middle_curled = self.is_finger_curled(landmarks, self.MIDDLE_MCP, self.MIDDLE_TIP)
        ring_curled = self.is_finger_curled(landmarks, self.RING_MCP, self.RING_TIP)
        
        # Thumb can be out or in
        return index_ext and pinky_ext and middle_curled and ring_curled
