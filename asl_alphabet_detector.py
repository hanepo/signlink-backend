"""
ASL Alphabet Detector
Rule-based detection for ASL fingerspelling alphabet (A-Z)
Based on MediaPipe hand landmark positions
"""

import math

class ASLAlphabetDetector:
    """Rule-based detector for ASL alphabet (A-Z)"""
    
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
        self.letter_names = {
            'A': 'Letter A',
            'B': 'Letter B',
            'C': 'Letter C',
            'D': 'Letter D',
            'E': 'Letter E',
            'F': 'Letter F',
            'G': 'Letter G',
            'H': 'Letter H',
            'I': 'Letter I',
            'J': 'Letter J',
            'K': 'Letter K',
            'L': 'Letter L',
            'M': 'Letter M',
            'N': 'Letter N',
            'O': 'Letter O',
            'P': 'Letter P',
            'Q': 'Letter Q',
            'R': 'Letter R',
            'S': 'Letter S',
            'T': 'Letter T',
            'U': 'Letter U',
            'V': 'Letter V',
            'W': 'Letter W',
            'X': 'Letter X',
            'Y': 'Letter Y',
            'Z': 'Letter Z',
        }
    
    def distance(self, p1, p2):
        """Calculate Euclidean distance between two 3D points"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def is_finger_extended(self, landmarks, finger_tip_idx, finger_pip_idx, finger_mcp_idx):
        """Check if finger is extended (tip is far from palm)"""
        # Distance from tip to MCP should be larger than PIP to MCP
        tip_to_mcp = self.distance(landmarks[finger_tip_idx], landmarks[finger_mcp_idx])
        pip_to_mcp = self.distance(landmarks[finger_pip_idx], landmarks[finger_mcp_idx])
        return tip_to_mcp > pip_to_mcp * 1.3
    
    def is_finger_curled(self, landmarks, finger_tip_idx, finger_mcp_idx):
        """Check if finger is curled (tip close to MCP)"""
        dist = self.distance(landmarks[finger_tip_idx], landmarks[finger_mcp_idx])
        wrist_to_mcp = self.distance(landmarks[self.WRIST], landmarks[finger_mcp_idx])
        return dist < wrist_to_mcp * 0.6
    
    def detect_letter(self, landmarks):
        """
        Detect ASL alphabet letter from hand landmarks
        Returns: (letter, confidence) or (None, 0.0)
        """
        if not landmarks or len(landmarks) < 21:
            return None, 0.0
        
        # Check each letter in priority order
        detectors = [
            self._is_letter_a,
            self._is_letter_b,
            self._is_letter_c,
            self._is_letter_d,
            self._is_letter_e,
            self._is_letter_f,
            self._is_letter_g,
            self._is_letter_h,
            self._is_letter_i,
            self._is_letter_j,
            self._is_letter_k,
            self._is_letter_l,
            self._is_letter_m,
            self._is_letter_n,
            self._is_letter_o,
            self._is_letter_p,
            self._is_letter_q,
            self._is_letter_r,
            self._is_letter_s,
            self._is_letter_t,
            self._is_letter_u,
            self._is_letter_v,
            self._is_letter_w,
            self._is_letter_x,
            self._is_letter_y,
            self._is_letter_z,
        ]
        
        for detector in detectors:
            try:
                result = detector(landmarks)
                if result:
                    return result
            except Exception as e:
                # Skip this detector if it fails
                print(f"Error in {detector.__name__}: {e}")
                continue
        
        return None, 0.0
    
    # Letter A: Closed fist with thumb on side
    def _is_letter_a(self, lm):
        # All fingers curled into fist
        fingers_curled = all([
            self.is_finger_curled(lm, self.INDEX_TIP, self.INDEX_MCP),
            self.is_finger_curled(lm, self.MIDDLE_TIP, self.MIDDLE_MCP),
            self.is_finger_curled(lm, self.RING_TIP, self.RING_MCP),
            self.is_finger_curled(lm, self.PINKY_TIP, self.PINKY_MCP),
        ])
        
        # Thumb is sticking out to the side (not extended up)
        thumb_side = lm[self.THUMB_TIP].y > lm[self.THUMB_MCP].y - 0.05
        
        # Thumb not covering fingers (Letter S would have thumb over)
        thumb_not_over = lm[self.THUMB_TIP].y > lm[self.INDEX_PIP].y - 0.03
        
        if fingers_curled and thumb_side and thumb_not_over:
            return 'A', 0.90
        return None
    
    # Letter B: Four fingers extended, thumb folded across palm
    def _is_letter_b(self, lm):
        # Four fingers extended upward
        fingers_extended = all([
            self.is_finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP),
            self.is_finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP),
            self.is_finger_extended(lm, self.RING_TIP, self.RING_PIP, self.RING_MCP),
            self.is_finger_extended(lm, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP),
        ])
        
        # Thumb folded in (thumb tip below index MCP)
        thumb_folded = lm[self.THUMB_TIP].y > lm[self.INDEX_MCP].y
        
        # Fingers close together (not spread)
        fingers_together = self.distance(lm[self.INDEX_TIP], lm[self.MIDDLE_TIP]) < 0.05
        
        if fingers_extended and thumb_folded and fingers_together:
            return 'B', 0.90
        return None
    
    # Letter C: Hand curved like a 'C'
    def _is_letter_c(self, lm):
        # All fingers partially curled forming C shape
        thumb_index_gap = self.distance(lm[self.THUMB_TIP], lm[self.INDEX_TIP])
        thumb_middle_gap = self.distance(lm[self.THUMB_TIP], lm[self.MIDDLE_TIP])
        
        # Wide gap for C shape
        c_shape = thumb_index_gap > 0.15 and thumb_middle_gap > 0.15
        
        # Fingers slightly curled
        fingers_curled = all([
            lm[self.INDEX_TIP].y < lm[self.INDEX_MCP].y,
            lm[self.MIDDLE_TIP].y < lm[self.MIDDLE_MCP].y,
        ])
        
        if c_shape and fingers_curled:
            return 'C', 0.80
        return None
    
    # Letter D: Index finger up, other fingers and thumb form O
    def _is_letter_d(self, lm):
        # Only index extended
        index_extended = self.is_finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP)
        
        # Middle, ring, pinky curled
        others_curled = all([
            self.is_finger_curled(lm, self.MIDDLE_TIP, self.MIDDLE_MCP),
            self.is_finger_curled(lm, self.RING_TIP, self.RING_MCP),
            self.is_finger_curled(lm, self.PINKY_TIP, self.PINKY_MCP),
        ])
        
        # Thumb touches middle finger (forming O)
        thumb_middle_close = self.distance(lm[self.THUMB_TIP], lm[self.MIDDLE_TIP]) < 0.06
        
        if index_extended and others_curled and thumb_middle_close:
            return 'D', 0.85
        return None
    
    # Letter E: All fingers curled, thumb pressed
    def _is_letter_e(self, lm):
        # All fingers tightly curled
        all_curled = all([
            self.is_finger_curled(lm, self.INDEX_TIP, self.INDEX_MCP),
            self.is_finger_curled(lm, self.MIDDLE_TIP, self.MIDDLE_MCP),
            self.is_finger_curled(lm, self.RING_TIP, self.RING_MCP),
            self.is_finger_curled(lm, self.PINKY_TIP, self.PINKY_MCP),
        ])
        
        # Thumb pressed against fingers
        thumb_pressed = lm[self.THUMB_TIP].y > lm[self.INDEX_MCP].y
        
        if all_curled and thumb_pressed:
            return 'E', 0.80
        return None
    
    # Letter F: Index and thumb form circle, other three extended
    def _is_letter_f(self, lm):
        # OK sign with three fingers up
        thumb_index_touch = self.distance(lm[self.THUMB_TIP], lm[self.INDEX_TIP]) < 0.06
        
        # Middle, ring, pinky extended
        three_extended = all([
            self.is_finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP),
            self.is_finger_extended(lm, self.RING_TIP, self.RING_PIP),
            self.is_finger_extended(lm, self.PINKY_TIP, self.PINKY_PIP),
        ])
        
        if thumb_index_touch and three_extended:
            return 'F', 0.85
        return None
    
    # Letter G: Index and thumb point horizontally
    def _is_letter_g(self, lm):
        # Index and thumb extended horizontally
        index_extended = self.is_finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP)
        
        # Horizontal orientation
        horizontal = abs(lm[self.INDEX_TIP].y - lm[self.WRIST].y) < 0.15
        
        # Other fingers curled
        others_curled = all([
            not self.is_finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP),
            not self.is_finger_extended(lm, self.RING_TIP, self.RING_PIP),
        ])
        
        if index_extended and horizontal and others_curled:
            return 'G', 0.75
        return None
    
    # Letter H: Index and middle extended horizontally
    def _is_letter_h(self, lm):
        # Index and middle extended
        two_extended = all([
            self.is_finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP),
            self.is_finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP),
        ])
        
        # Horizontal orientation
        horizontal = abs(lm[self.INDEX_TIP].y - lm[self.MIDDLE_TIP].y) < 0.08
        
        # Others curled
        others_curled = all([
            not self.is_finger_extended(lm, self.RING_TIP, self.RING_PIP),
            not self.is_finger_extended(lm, self.PINKY_TIP, self.PINKY_PIP),
        ])
        
        if two_extended and horizontal and others_curled:
            return 'H', 0.80
        return None
    
    # Letter I: Pinky extended, others curled
    def _is_letter_i(self, lm):
        # Only pinky extended
        pinky_extended = self.is_finger_extended(lm, self.PINKY_TIP, self.PINKY_PIP)
        
        # Others curled
        others_curled = all([
            not self.is_finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP),
            not self.is_finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP),
            not self.is_finger_extended(lm, self.RING_TIP, self.RING_PIP),
        ])
        
        if pinky_extended and others_curled:
            return 'I', 0.85
        return None
    
    # Letter J: Pinky extended with J motion (static: just pinky up)
    def _is_letter_j(self, lm):
        # Same as I but requires motion - for static images, similar to I
        pinky_extended = self.is_finger_extended(lm, self.PINKY_TIP, self.PINKY_PIP)
        
        others_curled = all([
            not self.is_finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP),
            not self.is_finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP),
            not self.is_finger_extended(lm, self.RING_TIP, self.RING_PIP),
        ])
        
        # Slight downward angle for J
        j_angle = lm[self.PINKY_TIP].y > lm[self.PINKY_MCP].y - 0.15
        
        if pinky_extended and others_curled and j_angle:
            return 'J', 0.70
        return None
    
    # Letter K: Index and middle extended in V, thumb between
    def _is_letter_k(self, lm):
        # Index and middle extended
        index_middle_extended = all([
            self.is_finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP),
            self.is_finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP),
        ])
        
        # V shape
        v_shape = self.distance(lm[self.INDEX_TIP], lm[self.MIDDLE_TIP]) > 0.08
        
        # Thumb touches middle finger base
        thumb_position = lm[self.THUMB_TIP].y < lm[self.MIDDLE_MCP].y
        
        if index_middle_extended and v_shape and thumb_position:
            return 'K', 0.75
        return None
    
    # Letter L: Index and thumb extended, forming L
    def _is_letter_l(self, lm):
        # Index extended upward
        index_extended = self.is_finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP)
        
        # Thumb extended outward (perpendicular to index)
        thumb_extended = self.distance(lm[self.THUMB_TIP], lm[self.THUMB_MCP]) > 0.08
        
        # Others curled
        others_curled = all([
            self.is_finger_curled(lm, self.MIDDLE_TIP, self.MIDDLE_MCP),
            self.is_finger_curled(lm, self.RING_TIP, self.RING_MCP),
            self.is_finger_curled(lm, self.PINKY_TIP, self.PINKY_MCP),
        ])
        
        # L shape: thumb and index are far apart
        l_shape = self.distance(lm[self.INDEX_TIP], lm[self.THUMB_TIP]) > 0.15
        
        if index_extended and thumb_extended and others_curled and l_shape:
            return 'L', 0.85
        return None
    
    # Letter M: Three fingers curled over thumb
    def _is_letter_m(self, lm):
        # Index, middle, ring curled
        three_curled = all([
            self.is_finger_curled(lm, self.INDEX_TIP, self.INDEX_MCP),
            self.is_finger_curled(lm, self.MIDDLE_TIP, self.MIDDLE_MCP),
            self.is_finger_curled(lm, self.RING_TIP, self.RING_MCP),
        ])
        
        # Thumb tucked
        thumb_tucked = lm[self.THUMB_TIP].y > lm[self.INDEX_MCP].y
        
        # Pinky curled too
        pinky_curled = not self.is_finger_extended(lm, self.PINKY_TIP, self.PINKY_PIP)
        
        if three_curled and thumb_tucked and pinky_curled:
            return 'M', 0.75
        return None
    
    # Letter N: Two fingers curled over thumb
    def _is_letter_n(self, lm):
        # Index and middle curled
        two_curled = all([
            self.is_finger_curled(lm, self.INDEX_TIP, self.INDEX_MCP),
            self.is_finger_curled(lm, self.MIDDLE_TIP, self.MIDDLE_MCP),
        ])
        
        # Thumb tucked
        thumb_tucked = lm[self.THUMB_TIP].y > lm[self.INDEX_MCP].y
        
        # Ring and pinky not extended
        others_down = all([
            not self.is_finger_extended(lm, self.RING_TIP, self.RING_PIP),
            not self.is_finger_extended(lm, self.PINKY_TIP, self.PINKY_PIP),
        ])
        
        if two_curled and thumb_tucked and others_down:
            return 'N', 0.75
        return None
    
    # Letter O: All fingertips touch forming circle
    def _is_letter_o(self, lm):
        # All fingertips close to thumb
        all_touching = all([
            self.distance(lm[self.THUMB_TIP], lm[self.INDEX_TIP]) < 0.08,
            self.distance(lm[self.THUMB_TIP], lm[self.MIDDLE_TIP]) < 0.10,
        ])
        
        # Circle shape
        circle_shape = self.distance(lm[self.INDEX_TIP], lm[self.MIDDLE_TIP]) < 0.08
        
        if all_touching and circle_shape:
            return 'O', 0.80
        return None
    
    # Letter P: Like K but pointing down
    def _is_letter_p(self, lm):
        # Index and middle fingers pointing DOWN (hand inverted)
        index_down = lm[self.INDEX_TIP].y > lm[self.INDEX_MCP].y
        middle_down = lm[self.MIDDLE_TIP].y > lm[self.MIDDLE_MCP].y
        
        # Must be pointing down, not up
        fingers_pointing_down = index_down and middle_down
        
        # V shape between index and middle
        v_shape = self.distance(lm[self.INDEX_TIP], lm[self.MIDDLE_TIP]) > 0.06
        
        # Other fingers curled
        others_curled = all([
            self.is_finger_curled(lm, self.RING_TIP, self.RING_MCP),
            self.is_finger_curled(lm, self.PINKY_TIP, self.PINKY_MCP),
        ])
        
        if fingers_pointing_down and v_shape and others_curled:
            return 'P', 0.80
        return None
    
    # Letter Q: Like G but pointing down
    def _is_letter_q(self, lm):
        # Index and thumb pointing down
        pointing_down = lm[self.INDEX_TIP].y > lm[self.WRIST].y
        
        # Others curled
        others_curled = all([
            not self.is_finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP),
            not self.is_finger_extended(lm, self.RING_TIP, self.RING_PIP),
        ])
        
        if pointing_down and others_curled:
            return 'Q', 0.70
        return None
    
    # Letter R: Index and middle crossed
    def _is_letter_r(self, lm):
        # Index and middle extended and crossed
        two_extended = all([
            self.is_finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP),
            self.is_finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP),
        ])
        
        # Crossed (middle over index)
        crossed = lm[self.MIDDLE_TIP].x < lm[self.INDEX_TIP].x
        
        # Others curled
        others_curled = all([
            not self.is_finger_extended(lm, self.RING_TIP, self.RING_PIP),
            not self.is_finger_extended(lm, self.PINKY_TIP, self.PINKY_PIP),
        ])
        
        if two_extended and crossed and others_curled:
            return 'R', 0.75
        return None
    
    # Letter S: Fist with thumb over fingers (covering them)
    def _is_letter_s(self, lm):
        # All fingers curled into fist
        all_curled = all([
            self.is_finger_curled(lm, self.INDEX_TIP, self.INDEX_MCP),
            self.is_finger_curled(lm, self.MIDDLE_TIP, self.MIDDLE_MCP),
            self.is_finger_curled(lm, self.RING_TIP, self.RING_MCP),
            self.is_finger_curled(lm, self.PINKY_TIP, self.PINKY_MCP),
        ])
        
        # Thumb is OVER the fingers (on top, covering them)
        # In letter A, thumb is to the side. In S, thumb is on top
        thumb_over = lm[self.THUMB_TIP].y < lm[self.INDEX_PIP].y
        thumb_covering = lm[self.THUMB_TIP].z < lm[self.INDEX_PIP].z
        
        if all_curled and thumb_over and thumb_covering:
            return 'S', 0.90
        return None
    
    # Letter T: Thumb between index and middle
    def _is_letter_t(self, lm):
        # Fist with thumb poking through
        fingers_curled = all([
            self.is_finger_curled(lm, self.INDEX_TIP, self.INDEX_MCP),
            self.is_finger_curled(lm, self.MIDDLE_TIP, self.MIDDLE_MCP),
        ])
        
        # Thumb between index and middle
        thumb_between = (lm[self.INDEX_MCP].x < lm[self.THUMB_TIP].x < lm[self.MIDDLE_MCP].x or
                        lm[self.MIDDLE_MCP].x < lm[self.THUMB_TIP].x < lm[self.INDEX_MCP].x)
        
        if fingers_curled and thumb_between:
            return 'T', 0.75
        return None
    
    # Letter U: Index and middle extended together, pointing up
    def _is_letter_u(self, lm):
        # Index and middle extended
        two_extended = all([
            self.is_finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP),
            self.is_finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP),
        ])
        
        # Together (not spread apart like V)
        together = self.distance(lm[self.INDEX_TIP], lm[self.MIDDLE_TIP]) < 0.05
        
        # Others curled
        others_curled = all([
            self.is_finger_curled(lm, self.RING_TIP, self.RING_MCP),
            self.is_finger_curled(lm, self.PINKY_TIP, self.PINKY_MCP),
        ])
        
        if two_extended and together and others_curled:
            return 'U', 0.90
        return None
    
    # Letter V: Index and middle extended in V shape (peace sign)
    def _is_letter_v(self, lm):
        # Index and middle extended upward
        two_extended = all([
            self.is_finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP),
            self.is_finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP),
        ])
        
        # V shape (spread apart, not together like U)
        v_shape = self.distance(lm[self.INDEX_TIP], lm[self.MIDDLE_TIP]) > 0.07
        
        # Pointing UP not down (unlike P)
        pointing_up = lm[self.INDEX_TIP].y < lm[self.INDEX_MCP].y and lm[self.MIDDLE_TIP].y < lm[self.MIDDLE_MCP].y
        
        # Others curled
        others_curled = all([
            self.is_finger_curled(lm, self.RING_TIP, self.RING_MCP),
            self.is_finger_curled(lm, self.PINKY_TIP, self.PINKY_MCP),
        ])
        
        if two_extended and v_shape and pointing_up and others_curled:
            return 'V', 0.90
        return None
    
    # Letter W: Three fingers extended (index, middle, ring)
    def _is_letter_w(self, lm):
        # Index, middle, ring extended
        three_extended = all([
            self.is_finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP),
            self.is_finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP),
            self.is_finger_extended(lm, self.RING_TIP, self.RING_PIP, self.RING_MCP),
        ])
        
        # Spread in W shape
        spread = self.distance(lm[self.INDEX_TIP], lm[self.RING_TIP]) > 0.12
        
        # Pinky curled
        pinky_curled = self.is_finger_curled(lm, self.PINKY_TIP, self.PINKY_MCP)
        
        if three_extended and spread and pinky_curled:
            return 'W', 0.85
        return None
    
    # Letter X: Index finger bent, forming hook
    def _is_letter_x(self, lm):
        # Index bent at middle joint (hook shape)
        index_hooked = (lm[self.INDEX_TIP].y > lm[self.INDEX_DIP].y and
                       lm[self.INDEX_DIP].y < lm[self.INDEX_PIP].y)
        
        # Others curled
        others_curled = all([
            not self.is_finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP),
            not self.is_finger_extended(lm, self.RING_TIP, self.RING_PIP),
            not self.is_finger_extended(lm, self.PINKY_TIP, self.PINKY_PIP),
        ])
        
        if index_hooked and others_curled:
            return 'X', 0.75
        return None
    
    # Letter Y: Thumb and pinky extended (hang loose)
    def _is_letter_y(self, lm):
        # Pinky and thumb extended
        pinky_extended = self.is_finger_extended(lm, self.PINKY_TIP, self.PINKY_PIP)
        thumb_extended = lm[self.THUMB_TIP].x < lm[self.THUMB_MCP].x or lm[self.THUMB_TIP].x > lm[self.THUMB_MCP].x
        
        # Others curled
        others_curled = all([
            not self.is_finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP),
            not self.is_finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP),
            not self.is_finger_extended(lm, self.RING_TIP, self.RING_PIP),
        ])
        
        if pinky_extended and thumb_extended and others_curled:
            return 'Y', 0.85
        return None
    
    # Letter Z: Index draws Z motion (static: index pointing)
    def _is_letter_z(self, lm):
        # Index extended and pointing
        index_extended = self.is_finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP)
        
        # Others curled
        others_curled = all([
            not self.is_finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP),
            not self.is_finger_extended(lm, self.RING_TIP, self.RING_PIP),
            not self.is_finger_extended(lm, self.PINKY_TIP, self.PINKY_PIP),
        ])
        
        # Slight zigzag angle (hard to detect in static image)
        pointing = lm[self.INDEX_TIP].y < lm[self.INDEX_MCP].y
        
        if index_extended and others_curled and pointing:
            return 'Z', 0.65
        return None
