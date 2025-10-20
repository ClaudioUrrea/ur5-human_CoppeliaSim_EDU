"""
Generate synthetic training data for CBF
"""

import numpy as np
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))


def compute_minimum_distance(q_r, p_h):
    """
    Compute minimum distance between robot and human
    
    Simplified: distance between end-effector and closest human point
    """
    # Forward kinematics (simplified)
    # In real implementation, use robot model
    p_r_ee = q_r[0:3]  # Use first 3 joints as proxy for position
    
    # Reshape human positions
    p_h_reshaped = p_h.reshape(-1, 3)  # [6 body parts x 3D]
    
    # Compute distances to all body parts
    distances = np.linalg.norm(p_r_ee - p_h_reshaped, axis=1)
    return np.min(distances)


def generate_samples(n_samples: int = 50000,
                     safe_ratio: float = 0.4,
                     unsafe_ratio: float = 0.4) -> dict:
    """
    Generate training samples
    
    Args:
        n_samples: Total number of samples
        safe_ratio: Ratio of safe samples
        unsafe_ratio: Ratio of unsafe samples
        (remaining are boundary samples)
    
    Returns:
        Dictionary with states, labels, masks
    """
    print(f"Generating {n_samples} training samples...")
    
    # State dimensions
    n_safe = int(n_samples * safe_ratio)
    n_unsafe = int(n_samples * unsafe_ratio)
    n_boundary = n_samples - n_safe - n_unsafe
    
    states = []
    labels = []
    safe_mask = []
    boundary_mask = []
    
    # Safety threshold
    d_min_base = 0.15  # meters
    
    # Generate safe samples
    print("Generating safe samples...")
    for i in range(n_safe):
        # Random robot configuration
        q_r = np.random.uniform(-np.pi, np.pi, 6)
        q_dot_r = np.random.uniform(-1.0, 1.0, 6)
        
        # Random human position (far from robot)
        p_h = np.random.uniform(-2.0, 2.0, 18)
        v_h = np.random.uniform(-0.5, 0.5, 18)
        
        # Ensure safe distance
        d = compute_minimum_distance(q_r, p_h)
        while d < d_min_base + 0.05:
            p_h = np.random.uniform(-2.0, 2.0, 18)
            d = compute_minimum_distance(q_r, p_h)
        
        # Construct state
        s_obj = np.random.rand(12)
        s_conv = np.random.rand(1)
        s_task = np.random.rand(5)
        
        state = np.concatenate([q_r, q_dot_r, p_h, v_h, s_obj, s_conv, s_task])
        
        states.append(state)
        labels.append(1.0)  # Safe
        safe_mask.append(True)
        boundary_mask.append(False)
    
    # Generate unsafe samples
    print("Generating unsafe samples...")
    for i in range(n_unsafe):
        q_r = np.random.uniform(-np.pi, np.pi, 6)
        q_dot_r = np.random.uniform(-1.0, 1.0, 6)
        
        # Human position (close to robot)
        p_h = np.random.uniform(-1.0, 1.0, 18)
        v_h = np.random.uniform(-0.5, 0.5, 18)
        
        # Ensure unsafe distance
        d = compute_minimum_distance(q_r, p_h)
        while d > d_min_base - 0.05:
            p_h = np.random.uniform(-0.5, 0.5, 18)
            d = compute_minimum_distance(q_r, p_h)
        
        s_obj = np.random.rand(12)
        s_conv = np.random.rand(1)
        s_task = np.random.rand(5)
        
        state = np.concatenate([q_r, q_dot_r, p_h, v_h, s_obj, s_conv, s_task])
        
        states.append(state)
        labels.append(0.0)  # Unsafe
        safe_mask.append(False)
        boundary_mask.append(False)
    
    # Generate boundary samples
    print("Generating boundary samples...")
    for i in range(n_boundary):
        q_r = np.random.uniform(-np.pi, np.pi, 6)
        q_dot_r = np.random.uniform(-1.0, 1.0, 6)
        
        # Human position (near boundary)
        p_h = np.random.uniform(-1.5, 1.5, 18)
        v_h = np.random.uniform(-0.5, 0.5, 18)
        
        # Adjust to be near boundary
        d = compute_minimum_distance(q_r, p_h)
        target_d = d_min_base + np.random.uniform(-0.05, 0.05)
        p_h *= (target_d / d)
        
        s_obj = np.random.rand(12)
        s_conv = np.random.rand(1)
        s_task = np.random.rand(5)
        
        state = np.concatenate([q_r, q_dot_r, p_h, v_h, s_obj, s_conv, s_task])
        
        # Label based on actual distance
        d_actual = compute_minimum_distance(q_r, p_h)
        label = 1.0 if d_actual >= d_min_base else 0.0
        
        states.append(state)
        labels.append(label)
        safe_mask.append(label == 1.0)
        boundary_mask.append(True)
    
    # Convert to arrays
    states = np.array(states)
    labels = np.array(labels)
    safe_mask = np.array(safe_mask)
    boundary_mask = np.array(boundary_mask)
    
    print(f"Generated {len(states)} samples")
    print(f"  Safe: {np.sum(safe_mask)} ({100*np.mean(safe_mask):.1f}%)")
    print(f"  Unsafe: {np.sum(~safe_mask & ~boundary_mask)} ({100*np.mean(~safe_mask & ~boundary_mask):.1f}%)")
    print(f"  Boundary: {np.sum(boundary_mask)} ({100*np.mean(boundary_mask):.1f}%)")
    
    return {
        'states': states,
        'labels': labels,
        'safe_mask': safe_mask,
        'boundary_mask': boundary_mask,
        'metadata': {
            'n_samples': n_samples,
            'd_min_base': d_min_base,
            'state_dim': 38
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--output', type=str, default='data/raw/training_dataset_50k.npz')
    args = parser.parse_args()
    
    # Generate data
    data = generate_samples(n_samples=args.samples)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        **data
    )
    
    print(f"\nSaved to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()