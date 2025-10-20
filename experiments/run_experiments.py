"""
Run HRC experiments with different methods
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.cbf.lipschitz_network import LipschitzCBFNetwork
from src.control.mpc_solver import LBCFMPC
from src.prediction.gp_predictor import GPHumanPredictor
from src.simulation.coppeliasim_interface import CoppeliaSimEnv
from src.utils.metrics import SafetyMetrics, EfficiencyMetrics


class ExperimentRunner:
    """Run experiments across scenarios and methods"""
    
    def __init__(self, scenario: int, method: str, visualize: bool = False):
        self.scenario = scenario
        self.method = method
        self.visualize = visualize
        
        # Load models
        self.load_models()
        
        # Initialize environment
        self.env = CoppeliaSimEnv(
            scene_file='coppeliasim/Scene_in_CoppeliaSim_for_Mathematics.ttt',
            scenario=scenario,
            headless=not visualize
        )
        
        # Metrics
        self.safety_metrics = SafetyMetrics()
        self.efficiency_metrics = EfficiencyMetrics()
    
    def load_models(self):
        """Load pre-trained models"""
        # CBF model
        self.cbf = LipschitzCBFNetwork()
        checkpoint = torch.load('models/checkpoints/cbf_lipschitz_epoch500.pth')
        self.cbf.load_state_dict(checkpoint['model_state_dict'])
        self.cbf.eval()
        
        # GP predictor
        self.gp = GPHumanPredictor()
        
        # Controller
        if self.method == 'LBCF-MPC':
            self.controller = LBCFMPC(
                cbf_model=self.cbf,
                gp_predictor=self.gp,
                horizon=20
            )
        # Add other methods...
    
    def run_single_episode(self, episode_id: int, duration: float = 120.0) -> dict:
        """Run single episode"""
        print(f"Running episode {episode_id}...")
        
        # Reset environment
        x0 = self.env.reset()
        
        # Episode data
        trajectory = {
            'states': [],
            'controls': [],
            'distances': [],
            'violations': [],
            'solve_times': [],
            'cbf_values': []
        }
        
        t = 0.0
        dt = 0.02  # 50Hz
        steps = int(duration / dt)
        
        violation_occurred = False
        
        for step in tqdm(range(steps), desc=f"Episode {episode_id}"):
            # Get state
            x = self.env.get_state()
            
            # Compute control
            t_start = time.time()
            u, info = self.controller.solve(x)
            solve_time = (time.time() - t_start) * 1000  # ms
            
            # Apply control
            self.env.step(u)
            
            # Compute metrics
            d_min = self.env.compute_distance()
            d_threshold = 0.15  # Safety threshold
            
            violation = d_min < d_threshold
            if violation:
                violation_occurred = True
            
            # CBF value
            with torch.no_grad():
                x_torch = torch.FloatTensor(x).unsqueeze(0)
                h_val = self.cbf(x_torch).item()
            
            # Store
            trajectory['states'].append(x)
            trajectory['controls'].append(u)
            trajectory['distances'].append(d_min)
            trajectory['violations'].append(violation)
            trajectory['solve_times'].append(solve_time)
            trajectory['cbf_values'].append(h_val)
            
            t += dt
        
        # Compute episode metrics
        results = {
            'episode_id': episode_id,
            'scenario': self.scenario,
            'method': self.method,
            'violation_occurred': violation_occurred,
            'violation_rate': np.mean(trajectory['violations']),
            'min_distance': np.min(trajectory['distances']),
            'mean_distance': np.mean(trajectory['distances']),
            'safety_margin': np.mean(np.array(trajectory['distances']) - d_threshold),
            'avg_solve_time': np.mean(trajectory['solve_times']),
            'max_solve_time': np.max(trajectory['solve_times']),
            'throughput': self.efficiency_metrics.compute_throughput(trajectory),
            'energy': self.efficiency_metrics.compute_energy(trajectory),
            'trajectory': trajectory
        }
        
        return results
    
    def run_multiple_episodes(self, n_episodes: int = 30) -> list:
        """Run multiple episodes"""
        all_results = []
        
        for i in range(n_episodes):
            result = self.run_single_episode(i)
            all_results.append(result)
            
            print(f"Episode {i} summary:")
            print(f"  Violation: {result['violation_occurred']}")
            print(f"  Min distance: {result['min_distance']:.3f}m")
            print(f"  Avg solve time: {result['avg_solve_time']:.1f}ms")
        
        return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=int, default=3,
                       choices=[1, 2, 3, 4])
    parser.add_argument('--method', type=str, default='LBCF-MPC',
                       choices=['LBCF-MPC', 'CBF-Manual', 'MPC-Soft', 
                               'CBF-Reactive', 'Safe-RL', 'PID-Trad'])
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--output', type=str, 
                       default='data/processed/experimental_results.pkl')
    
    args = parser.parse_args()
    
    # Run experiments
    runner = ExperimentRunner(
        scenario=args.scenario,
        method=args.method,
        visualize=args.visualize
    )
    
    results = runner.run_multiple_episodes(n_episodes=args.runs)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_path}")
    
    # Summary statistics
    violations = sum(r['violation_occurred'] for r in results)
    avg_min_dist = np.mean([r['min_distance'] for r in results])
    avg_solve = np.mean([r['avg_solve_time'] for r in results])
    
    print("\n" + "="*50)
    print(f"SUMMARY - Scenario {args.scenario}, Method {args.method}")
    print("="*50)
    print(f"Total runs: {args.runs}")
    print(f"Violations: {violations}/{args.runs} ({100*violations/args.runs:.1f}%)")
    print(f"Average min distance: {avg_min_dist:.3f}m")
    print(f"Average solve time: {avg_solve:.1f}ms")


if __name__ == "__main__":
    main()