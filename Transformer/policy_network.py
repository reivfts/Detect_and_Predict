"""
Policy Anticipation Network for High-Level Maneuver Prediction.

Implements LSTM-based policy classification inspired by the two-level
trajectory prediction framework. Predicts discrete driving policies
(forward, yield, turn_left, turn_right, lane_change_left, lane_change_right)
from sequential track observations.

This addresses the multimodal nature of future trajectories by anticipating
high-level intentions before low-level optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


class PolicyAnticipationLSTM(nn.Module):
    """
    LSTM-based policy anticipation network.
    
    Takes sequential observations [x, y, theta, v] over Tobs window
    and predicts future policy over Tpred window.
    
    Policy classes:
        0: forward
        1: yield (stop/slow)
        2: turn_left
        3: turn_right
        4: lane_change_left
        5: lane_change_right
    """
    
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, num_policies=6, dropout=0.2):
        super(PolicyAnticipationLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_policies = num_policies
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification head
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_policies)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_size) - sequential observations
        
        Returns:
            policy_probs: (batch, num_policies) - policy probabilities
        """
        # LSTM encoding
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Take last hidden state
        last_hidden = hn[-1]  # (batch, hidden_size)
        
        # Classification
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        policy_probs = self.softmax(out)
        
        return policy_probs


class PolicyInterpreter:
    """
    Interprets policy predictions and generates trajectory initial guesses
    for optimization-based reasoning.
    
    Bridges high-level policy anticipation with low-level trajectory optimization.
    """
    
    POLICY_NAMES = {
        0: "forward",
        1: "yield",
        2: "turn_left",
        3: "turn_right",
        4: "lane_change_left",
        5: "lane_change_right"
    }
    
    def __init__(self):
        self.policy_references = {}  # Cache for policy -> reference trajectory mappings
        
    def interpret(self, policy_probs, current_state, num_steps=10):
        """
        Interpret policy probabilities and generate trajectory initial guess.
        
        Args:
            policy_probs: (num_policies,) - softmax policy probabilities
            current_state: dict with keys 'box', 'velocity', etc.
            num_steps: number of predicted trajectory points
        
        Returns:
            initial_trajectory: (num_steps, 2) - [x, y] coordinates
            selected_policy: int - index of selected policy
            policy_name: str - human-readable policy name
        """
        # Select policy with maximum likelihood
        selected_policy = np.argmax(policy_probs)
        policy_name = self.POLICY_NAMES[selected_policy]
        
        # Extract current position and velocity
        box = current_state['box']
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        velocity = current_state.get('velocity', [0, 0])
        
        # Generate simple initial trajectory based on policy
        # (in production, this would use map references)
        initial_trajectory = self._generate_reference_trajectory(
            cx, cy, velocity, selected_policy, num_steps
        )
        
        return initial_trajectory, selected_policy, policy_name
    
    def _generate_reference_trajectory(self, cx, cy, velocity, policy, num_steps):
        """
        Generate simple reference trajectory for given policy.
        
        In production, this would extract map-based reference lines.
        For now, generates geometric patterns.
        """
        dt = 0.1  # time step (100ms)
        trajectory = np.zeros((num_steps, 2))
        trajectory[0] = [cx, cy]
        
        # Estimate velocity magnitude
        v_mag = np.linalg.norm(velocity) if isinstance(velocity, (list, np.ndarray)) else 5.0
        if v_mag < 0.1:
            v_mag = 5.0  # default 5 m/s
        
        # Estimate heading from velocity or assume forward
        if isinstance(velocity, (list, np.ndarray)) and len(velocity) >= 2:
            heading = np.arctan2(velocity[1], velocity[0])
        else:
            heading = 0.0  # assume forward
        
        for i in range(1, num_steps):
            if policy == 0:  # forward
                dx = v_mag * dt * np.cos(heading)
                dy = v_mag * dt * np.sin(heading)
            elif policy == 1:  # yield (decelerate)
                v_current = max(0, v_mag - 2.0 * i * dt)
                dx = v_current * dt * np.cos(heading)
                dy = v_current * dt * np.sin(heading)
            elif policy == 2:  # turn_left
                turn_rate = 0.3  # rad/s
                heading += turn_rate * dt
                dx = v_mag * dt * np.cos(heading)
                dy = v_mag * dt * np.sin(heading)
            elif policy == 3:  # turn_right
                turn_rate = -0.3
                heading += turn_rate * dt
                dx = v_mag * dt * np.cos(heading)
                dy = v_mag * dt * np.sin(heading)
            elif policy == 4:  # lane_change_left
                lateral_shift = 0.3  # m per step
                dx = v_mag * dt * np.cos(heading)
                dy = v_mag * dt * np.sin(heading) + lateral_shift
            elif policy == 5:  # lane_change_right
                lateral_shift = -0.3
                dx = v_mag * dt * np.cos(heading)
                dy = v_mag * dt * np.sin(heading) + lateral_shift
            else:
                dx = dy = 0
            
            trajectory[i] = trajectory[i-1] + [dx, dy]
        
        return trajectory


class OnlinePolicyPredictor:
    """
    Online policy prediction wrapper for real-time trajectory anticipation.
    
    Maintains sliding windows of observations per track and provides
    policy predictions for active tracks.
    """
    
    def __init__(self, model_path=None, obs_window=40, device='cuda'):
        self.obs_window = obs_window
        self.device = device
        
        # Initialize network
        self.network = PolicyAnticipationLSTM().to(device)
        if model_path:
            self.network.load_state_dict(torch.load(model_path))
        self.network.eval()
        
        # Interpreter
        self.interpreter = PolicyInterpreter()
        
        # Sliding observation windows per track
        self.track_observations = defaultdict(list)  # track_id -> [(x,y,theta,v), ...]
        
    def update_observation(self, track_id, box, velocity=None):
        """
        Update observation history for a track.
        
        Args:
            track_id: unique track identifier
            box: [x1, y1, x2, y2]
            velocity: [vx, vy] or scalar speed
        """
        # Compute center and estimate theta from velocity
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        
        if velocity is not None:
            if isinstance(velocity, (list, tuple, np.ndarray)) and len(velocity) >= 2:
                vx, vy = velocity[0], velocity[1]
                v_mag = np.sqrt(vx**2 + vy**2)
                theta = np.arctan2(vy, vx)
            else:
                v_mag = float(velocity)
                theta = 0.0  # unknown heading
        else:
            v_mag = 0.0
            theta = 0.0
        
        # Append to sliding window
        obs = np.array([cx, cy, theta, v_mag], dtype=np.float32)
        self.track_observations[track_id].append(obs)
        
        # Keep only last obs_window observations
        if len(self.track_observations[track_id]) > self.obs_window:
            self.track_observations[track_id] = self.track_observations[track_id][-self.obs_window:]
    
    def predict_policy(self, track_id, current_state):
        """
        Predict policy for a track.
        
        Args:
            track_id: track identifier
            current_state: dict with 'box', 'velocity', etc.
        
        Returns:
            dict with keys:
                - policy_probs: (num_policies,) probabilities
                - selected_policy: int
                - policy_name: str
                - initial_trajectory: (num_steps, 2) if interpretation requested
        """
        # Check if enough observations
        if track_id not in self.track_observations:
            return None
        
        obs_history = self.track_observations[track_id]
        if len(obs_history) < 5:  # need minimum observations
            return None
        
        # Pad to obs_window if needed
        if len(obs_history) < self.obs_window:
            padding = [obs_history[0]] * (self.obs_window - len(obs_history))
            obs_sequence = padding + obs_history
        else:
            obs_sequence = obs_history[-self.obs_window:]
        
        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs_sequence).unsqueeze(0).to(self.device)  # (1, obs_window, 4)
        
        # Predict
        with torch.no_grad():
            policy_probs = self.network(obs_tensor)
        
        policy_probs_np = policy_probs.cpu().numpy()[0]  # (num_policies,)
        
        # Interpret policy
        initial_trajectory, selected_policy, policy_name = self.interpreter.interpret(
            policy_probs_np, current_state
        )
        
        return {
            'policy_probs': policy_probs_np,
            'selected_policy': selected_policy,
            'policy_name': policy_name,
            'initial_trajectory': initial_trajectory
        }
    
    def cleanup_track(self, track_id):
        """Remove observations for lost track."""
        if track_id in self.track_observations:
            del self.track_observations[track_id]
