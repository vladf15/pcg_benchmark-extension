import numpy as np
from .utils import get_racing_line_targets

class SimpleAgent:
    def __init__(self, curve_points, track_width=50, line_strategy=None):
        self.curve_points = curve_points
        self.track_width = track_width
        self.current_segment_idx = 0
        self.current_phase = 'entry'  # 'entry', 'apex', 'exit'
        # line_strategy: dict with keys 'entry', 'apex', 'exit', values 'left'/'center'/'right'
        if line_strategy is None:
            self.line_strategy = {'entry': 'center', 'apex': 'center', 'exit': 'center'}
        else:
            self.line_strategy = line_strategy
        self.targets = get_racing_line_targets(curve_points, track_width)

    def reset(self):
        self.current_segment_idx = 0
        self.current_phase = 'entry'

    def act(self, car_state):
        x, y, angle, velocity, steering_angle = car_state
        seg_idx = self.current_segment_idx
        phase = self.current_phase
        if seg_idx >= len(self.targets):
            return {'steering': 0.0, 'throttle': 0.0}
        target = self.targets[seg_idx][phase][self.line_strategy[phase]]
        dx = target[0] - x
        dy = target[1] - y
        distance = np.hypot(dx, dy)
        # Advance phase/segment when close to target
        threshold = 12.0
        if distance < threshold:
            if phase == 'entry':
                self.current_phase = 'apex'
            elif phase == 'apex':
                self.current_phase = 'exit'
            elif phase == 'exit':
                self.current_phase = 'entry'
                self.current_segment_idx += 1
        angle_to_target = np.arctan2(dy, dx)
        steering = angle_to_target - angle
        steering = (steering + np.pi) % (2 * np.pi) - np.pi
        # Throttle logic: slow down for sharp turns
        # Estimate turn angle at apex (between previous and next segment)
        turn_factor = 1.0
        if phase == 'apex' and 0 < seg_idx < len(self.targets)-1:
            prev_apex = self.targets[seg_idx-1]['apex']['center']
            curr_apex = self.targets[seg_idx]['apex']['center']
            next_apex = self.targets[seg_idx+1]['apex']['center']
            v1 = curr_apex - prev_apex
            v2 = next_apex - curr_apex
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-3 and norm2 > 1e-3:
                v1 /= norm1
                v2 /= norm2
                turn_angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                # Scale throttle: sharper turn = lower throttle
                turn_factor = max(0.3, 1.0 - turn_angle/np.pi)
        throttle = turn_factor if phase == 'apex' else 1.0
        return {'steering': steering, 'throttle': throttle}