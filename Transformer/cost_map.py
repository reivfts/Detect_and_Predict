"""
Multi-Layer Cost Map Structure for Context-Aware Trajectory Optimization.

Implements the optimization-based low-level reasoning from the two-level
trajectory prediction framework. Uses multiple cost layers:
    - Static: lane geometry, road boundaries
    - Moving Obstacles (MO): collision avoidance with dynamic objects
    - Context: traffic rules, speed limits, red lights
    - Nonholonomic: kinematic constraints (curvature, acceleration limits)

Optimizes trajectory over predicted horizon to minimize total cost.
"""

import numpy as np
from scipy.optimize import minimize
from collections import defaultdict


class CostMapLayer:
    """Base class for cost map layers."""
    
    def __init__(self, weight=1.0):
        self.weight = weight
    
    def compute_cost(self, trajectory, context=None):
        """
        Compute cost for given trajectory.
        
        Args:
            trajectory: (num_steps, 2) - [x, y] coordinates
            context: dict with contextual information
        
        Returns:
            cost: float - layer cost
        """
        raise NotImplementedError


class StaticCostLayer(CostMapLayer):
    """
    Static environment cost: lane geometry, road boundaries.
    
    Penalizes deviation from lane centerline and proximity to boundaries.
    """
    
    def __init__(self, weight=1.0, lane_width=3.5):
        super().__init__(weight)
        self.lane_width = lane_width
    
    def compute_cost(self, trajectory, context=None):
        """
        Compute static cost.
        
        For now, uses simple distance-to-reference metric.
        In production, would integrate HD map queries.
        """
        if context is None or 'lane_centerline' not in context:
            # No map available, return zero cost
            return 0.0
        
        lane_centerline = context['lane_centerline']  # (N, 2)
        total_cost = 0.0
        
        for point in trajectory:
            # Find nearest centerline point
            distances = np.linalg.norm(lane_centerline - point, axis=1)
            min_dist = np.min(distances)
            
            # Quadratic penalty for deviation
            total_cost += min_dist ** 2
        
        return self.weight * total_cost


class MovingObstacleCostLayer(CostMapLayer):
    """
    Moving obstacle cost: collision avoidance with dynamic objects.
    
    Penalizes proximity to predicted future positions of other agents.
    """
    
    def __init__(self, weight=10.0, safety_radius=2.0):
        super().__init__(weight)
        self.safety_radius = safety_radius
    
    def compute_cost(self, trajectory, context=None):
        """
        Compute moving obstacle cost.
        
        Context should contain 'obstacles': list of dicts with
            - predicted_trajectory: (num_steps, 2)
            - radius: float
        """
        if context is None or 'obstacles' not in context:
            return 0.0
        
        obstacles = context['obstacles']
        total_cost = 0.0
        
        for i, ego_point in enumerate(trajectory):
            for obs in obstacles:
                obs_traj = obs.get('predicted_trajectory', None)
                if obs_traj is None or i >= len(obs_traj):
                    continue
                
                obs_point = obs_traj[i]
                obs_radius = obs.get('radius', 1.0)
                
                # Distance to obstacle
                dist = np.linalg.norm(ego_point - obs_point)
                
                # Exponential penalty for proximity
                margin = self.safety_radius + obs_radius
                if dist < margin:
                    cost = np.exp(-(dist - margin))
                    total_cost += cost
        
        return self.weight * total_cost


class ContextCostLayer(CostMapLayer):
    """
    Context cost: traffic rules, speed limits, red lights.
    
    Penalizes violations of traffic rules and unsafe behaviors.
    """
    
    def __init__(self, weight=5.0):
        super().__init__(weight)
    
    def compute_cost(self, trajectory, context=None):
        """
        Compute context cost.
        
        Context should contain:
            - speed_limit: float (m/s)
            - red_light_zones: list of (x, y, radius)
            - no_go_zones: list of (x, y, radius)
        """
        if context is None:
            return 0.0
        
        total_cost = 0.0
        
        # Speed limit violation
        speed_limit = context.get('speed_limit', 20.0)  # default 20 m/s
        for i in range(1, len(trajectory)):
            displacement = np.linalg.norm(trajectory[i] - trajectory[i-1])
            speed = displacement / 0.1  # assume dt=0.1s
            if speed > speed_limit:
                total_cost += (speed - speed_limit) ** 2
        
        # Red light zones
        red_light_zones = context.get('red_light_zones', [])
        for zone in red_light_zones:
            zone_center = np.array(zone[:2])
            zone_radius = zone[2]
            for point in trajectory:
                dist = np.linalg.norm(point - zone_center)
                if dist < zone_radius:
                    total_cost += 100.0  # large penalty for entering red light zone
        
        # No-go zones (e.g., opposite lanes, sidewalks)
        no_go_zones = context.get('no_go_zones', [])
        for zone in no_go_zones:
            zone_center = np.array(zone[:2])
            zone_radius = zone[2]
            for point in trajectory:
                dist = np.linalg.norm(point - zone_center)
                if dist < zone_radius:
                    total_cost += 50.0
        
        return self.weight * total_cost


class NonholonomicCostLayer(CostMapLayer):
    """
    Nonholonomic cost: kinematic constraints.
    
    Penalizes high curvature, acceleration, jerk for realistic vehicle motion.
    """
    
    def __init__(self, weight=2.0, max_curvature=0.5, max_accel=3.0):
        super().__init__(weight)
        self.max_curvature = max_curvature
        self.max_accel = max_accel
    
    def compute_cost(self, trajectory, context=None):
        """
        Compute nonholonomic cost.
        """
        if len(trajectory) < 3:
            return 0.0
        
        total_cost = 0.0
        dt = 0.1  # assume 100ms steps
        
        # Curvature penalty
        for i in range(1, len(trajectory) - 1):
            p1, p2, p3 = trajectory[i-1], trajectory[i], trajectory[i+1]
            
            # Estimate curvature from three points
            # Simplified: angle change per unit distance
            v1 = p2 - p1
            v2 = p3 - p2
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.arccos(cos_theta)
                curvature = theta / (norm1 + 1e-6)
                
                if curvature > self.max_curvature:
                    total_cost += (curvature - self.max_curvature) ** 2
        
        # Acceleration penalty
        for i in range(2, len(trajectory)):
            v_prev = (trajectory[i-1] - trajectory[i-2]) / dt
            v_curr = (trajectory[i] - trajectory[i-1]) / dt
            accel = (v_curr - v_prev) / dt
            accel_mag = np.linalg.norm(accel)
            
            if accel_mag > self.max_accel:
                total_cost += (accel_mag - self.max_accel) ** 2
        
        return self.weight * total_cost


class MultiLayerCostMap:
    """
    Multi-layer cost map combining all cost sources.
    
    Aggregates costs from static, moving obstacles, context, and
    nonholonomic layers to produce total trajectory cost.
    """
    
    def __init__(self):
        self.layers = {
            'static': StaticCostLayer(weight=1.0),
            'moving_obstacles': MovingObstacleCostLayer(weight=10.0),
            'context': ContextCostLayer(weight=5.0),
            'nonholonomic': NonholonomicCostLayer(weight=2.0)
        }
    
    def compute_total_cost(self, trajectory, context=None):
        """
        Compute total cost across all layers.
        
        Args:
            trajectory: (num_steps, 2) - [x, y] coordinates
            context: dict with contextual information for all layers
        
        Returns:
            total_cost: float
            cost_breakdown: dict - per-layer costs
        """
        cost_breakdown = {}
        total_cost = 0.0
        
        for layer_name, layer in self.layers.items():
            layer_cost = layer.compute_cost(trajectory, context)
            cost_breakdown[layer_name] = layer_cost
            total_cost += layer_cost
        
        return total_cost, cost_breakdown
    
    def set_layer_weight(self, layer_name, weight):
        """Update weight for a specific layer."""
        if layer_name in self.layers:
            self.layers[layer_name].weight = weight


class TrajectoryOptimizer:
    """
    Optimization-based trajectory prediction.
    
    Takes policy-based initial guess and optimizes trajectory using
    multi-layer cost map to find context-aware predictions.
    """
    
    def __init__(self, num_steps=10, dt=0.1):
        self.num_steps = num_steps
        self.dt = dt
        self.cost_map = MultiLayerCostMap()
    
    def optimize(self, initial_trajectory, context=None, max_iter=50):
        """
        Optimize trajectory to minimize cost.
        
        Args:
            initial_trajectory: (num_steps, 2) - initial guess (e.g., from policy)
            context: dict with all contextual information
            max_iter: maximum optimization iterations
        
        Returns:
            optimized_trajectory: (num_steps, 2)
            final_cost: float
            cost_breakdown: dict
        """
        # Flatten initial trajectory
        x0 = initial_trajectory.flatten()
        
        # Define cost function for optimizer
        def cost_fn(x_flat):
            traj = x_flat.reshape(self.num_steps, 2)
            cost, _ = self.cost_map.compute_total_cost(traj, context)
            return cost
        
        # Optimize using scipy
        result = minimize(
            cost_fn,
            x0,
            method='BFGS',
            options={'maxiter': max_iter, 'disp': False}
        )
        
        # Reshape optimized trajectory
        optimized_trajectory = result.x.reshape(self.num_steps, 2)
        
        # Compute final cost breakdown
        final_cost, cost_breakdown = self.cost_map.compute_total_cost(
            optimized_trajectory, context
        )
        
        return optimized_trajectory, final_cost, cost_breakdown


class TwoLevelTrajectoryPredictor:
    """
    Two-level trajectory predictor combining policy anticipation and optimization.
    
    High-level: LSTM policy anticipation
    Low-level: Cost map optimization with context reasoning
    """
    
    def __init__(self, policy_model_path=None, obs_window=40, pred_steps=10):
        from Transformer.policy_network import OnlinePolicyPredictor
        
        self.policy_predictor = OnlinePolicyPredictor(
            model_path=policy_model_path,
            obs_window=obs_window
        )
        self.optimizer = TrajectoryOptimizer(num_steps=pred_steps)
        
        # Cache for tracking
        self.track_contexts = defaultdict(dict)
    
    def update_observation(self, track_id, box, velocity=None):
        """Update observation for policy prediction."""
        self.policy_predictor.update_observation(track_id, box, velocity)
    
    def predict(self, track_id, current_state, context=None):
        """
        Full two-level prediction.
        
        Args:
            track_id: track identifier
            current_state: dict with 'box', 'velocity', etc.
            context: dict with map/traffic/obstacle information
        
        Returns:
            dict with:
                - trajectory: (num_steps, 2) - optimized prediction
                - policy_name: str - anticipated policy
                - policy_probs: array - policy probabilities
                - cost: float - final optimized cost
                - cost_breakdown: dict - per-layer costs
        """
        # High-level: Policy anticipation
        policy_result = self.policy_predictor.predict_policy(track_id, current_state)
        
        if policy_result is None:
            # Not enough observations, return simple linear extrapolation
            from Transformer.trajectory_predictor import linear_extrapolate
            box = current_state['box']
            velocity = current_state.get('velocity', [0, 0])
            traj = linear_extrapolate(box, velocity, self.optimizer.num_steps)
            return {
                'trajectory': traj,
                'policy_name': 'linear_fallback',
                'policy_probs': None,
                'cost': 0.0,
                'cost_breakdown': {}
            }
        
        # Low-level: Optimization with cost map
        initial_trajectory = policy_result['initial_trajectory']
        optimized_trajectory, final_cost, cost_breakdown = self.optimizer.optimize(
            initial_trajectory, context
        )
        
        return {
            'trajectory': optimized_trajectory,
            'policy_name': policy_result['policy_name'],
            'policy_probs': policy_result['policy_probs'],
            'cost': final_cost,
            'cost_breakdown': cost_breakdown
        }
    
    def cleanup_track(self, track_id):
        """Cleanup track data."""
        self.policy_predictor.cleanup_track(track_id)
        if track_id in self.track_contexts:
            del self.track_contexts[track_id]
