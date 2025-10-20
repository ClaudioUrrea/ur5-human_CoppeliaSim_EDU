"""
CoppeliaSim Remote API Interface for LBCF-MPC
Python wrapper for CoppeliaSim simulation control

This module provides a simplified interface to CoppeliaSim's ZMQ Remote API
for controlling the UR5 robot and human operator simulation used in the
LBCF-MPC experiments.

Requirements:
    - CoppeliaSim EDU 4.10.0 or later
    - zmqRemoteApi Python package
    
Usage:
    from coppeliasim.python_remote_api import CoppeliaSimInterface
    
    sim = CoppeliaSimInterface()
    sim.connect()
    sim.start_simulation()
    # ... your control loop ...
    sim.stop_simulation()
    sim.disconnect()

Author: Claudio Urrea
Institution: Universidad de Santiago de Chile
Paper: Learning-Based Control Barrier Functions for Safe HRC (Mathematics 2025)
GitHub: https://github.com/ClaudioUrrea/ur5-human_CoppeliaSim_EDU
Dataset: https://doi.org/10.6084/m9.figshare.30282127
"""

import numpy as np
import time
import logging
from typing import Optional, List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoppeliaSimInterface:
    """
    Interface for CoppeliaSim Remote API communication.
    
    This class provides methods to control the UR5 robot simulation,
    retrieve human operator position, and manage the simulation state
    for LBCF-MPC experiments.
    
    Attributes:
        client: ZMQ Remote API client
        sim: CoppeliaSim simulation handle
        robot_handle: UR5 robot object handle
        human_handle: Human operator object handle
        joint_handles: List of UR5 joint handles
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 23000):
        """
        Initialize CoppeliaSim interface.
        
        Args:
            host: CoppeliaSim server host address
            port: CoppeliaSim ZMQ Remote API port
        """
        self.host = host
        self.port = port
        self.client = None
        self.sim = None
        self.connected = False
        
        # Object handles (initialized on connect)
        self.robot_handle = None
        self.human_handle = None
        self.joint_handles = []
        self.human_keypoint_handles = []
        
        logger.info(f"CoppeliaSim Interface initialized (host={host}, port={port})")
    
    def connect(self) -> bool:
        """
        Connect to CoppeliaSim via ZMQ Remote API.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            ImportError: If zmqRemoteApi is not installed
            ConnectionError: If cannot connect to CoppeliaSim
        """
        try:
            from zmqRemoteApi import RemoteAPIClient
            
            logger.info("Connecting to CoppeliaSim...")
            self.client = RemoteAPIClient(self.host, self.port)
            self.sim = self.client.require('sim')
            
            # Verify connection
            version = self.sim.getInt32Param(self.sim.intparam_program_version)
            logger.info(f"✓ Connected to CoppeliaSim version: {version}")
            
            # Initialize object handles
            self._initialize_handles()
            
            self.connected = True
            return True
            
        except ImportError:
            logger.error("zmqRemoteApi not installed. Install with: pip install zmqRemoteApi")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to CoppeliaSim: {e}")
            logger.error("Make sure CoppeliaSim is running with the correct scene loaded")
            raise ConnectionError(f"Cannot connect to CoppeliaSim: {e}")
    
    def _initialize_handles(self):
        """Initialize handles for robot, human, and other objects in scene."""
        try:
            # Get UR5 robot handle
            self.robot_handle = self.sim.getObject('/UR5')
            logger.info(f"✓ UR5 robot handle: {self.robot_handle}")
            
            # Get UR5 joint handles
            joint_names = [
                '/UR5/joint',
                '/UR5/link1/joint',
                '/UR5/link2/joint',
                '/UR5/link3/joint',
                '/UR5/link4/joint',
                '/UR5/link5/joint'
            ]
            
            self.joint_handles = []
            for joint_name in joint_names:
                try:
                    handle = self.sim.getObject(joint_name)
                    self.joint_handles.append(handle)
                except:
                    logger.warning(f"Could not find joint: {joint_name}")
            
            logger.info(f"✓ Found {len(self.joint_handles)} UR5 joints")
            
            # Get human operator handle
            try:
                self.human_handle = self.sim.getObject('/Human')
                logger.info(f"✓ Human operator handle: {self.human_handle}")
                
                # Get human keypoint handles for motion tracking
                keypoint_names = ['/Human/torso', '/Human/leftArm', 
                                 '/Human/rightArm', '/Human/head']
                self.human_keypoint_handles = []
                for kp_name in keypoint_names:
                    try:
                        handle = self.sim.getObject(kp_name)
                        self.human_keypoint_handles.append(handle)
                    except:
                        logger.warning(f"Could not find keypoint: {kp_name}")
                
                logger.info(f"✓ Found {len(self.human_keypoint_handles)} human keypoints")
            except:
                logger.warning("Human operator not found in scene")
                self.human_handle = None
            
        except Exception as e:
            logger.error(f"Error initializing handles: {e}")
            raise
    
    def start_simulation(self):
        """Start CoppeliaSim simulation."""
        if not self.connected:
            raise RuntimeError("Not connected to CoppeliaSim")
        
        self.sim.startSimulation()
        logger.info("✓ Simulation started")
        time.sleep(0.5)  # Allow simulation to stabilize
    
    def stop_simulation(self):
        """Stop CoppeliaSim simulation."""
        if not self.connected:
            return
        
        self.sim.stopSimulation()
        logger.info("✓ Simulation stopped")
        time.sleep(0.5)
    
    def disconnect(self):
        """Disconnect from CoppeliaSim."""
        if self.connected:
            self.stop_simulation()
            self.client = None
            self.sim = None
            self.connected = False
            logger.info("✓ Disconnected from CoppeliaSim")
    
    def get_robot_joint_positions(self) -> np.ndarray:
        """
        Get current UR5 joint positions.
        
        Returns:
            Array of 6 joint angles in radians
        """
        if not self.connected or not self.joint_handles:
            raise RuntimeError("Not connected or joints not initialized")
        
        positions = np.zeros(6)
        for i, handle in enumerate(self.joint_handles[:6]):
            positions[i] = self.sim.getJointPosition(handle)
        
        return positions
    
    def set_robot_joint_positions(self, positions: np.ndarray):
        """
        Set UR5 joint target positions.
        
        Args:
            positions: Array of 6 target joint angles in radians
        """
        if not self.connected or not self.joint_handles:
            raise RuntimeError("Not connected or joints not initialized")
        
        if len(positions) != 6:
            raise ValueError(f"Expected 6 joint positions, got {len(positions)}")
        
        for i, handle in enumerate(self.joint_handles[:6]):
            self.sim.setJointTargetPosition(handle, float(positions[i]))
    
    def get_robot_joint_velocities(self) -> np.ndarray:
        """
        Get current UR5 joint velocities.
        
        Returns:
            Array of 6 joint velocities in rad/s
        """
        if not self.connected or not self.joint_handles:
            raise RuntimeError("Not connected or joints not initialized")
        
        velocities = np.zeros(6)
        for i, handle in enumerate(self.joint_handles[:6]):
            velocities[i] = self.sim.getJointVelocity(handle)
        
        return velocities
    
    def get_human_position(self) -> Optional[np.ndarray]:
        """
        Get human operator position (torso center).
        
        Returns:
            3D position [x, y, z] in meters, or None if human not found
        """
        if not self.connected or self.human_handle is None:
            return None
        
        position = self.sim.getObjectPosition(self.human_handle, -1)
        return np.array(position)
    
    def get_human_keypoints(self) -> Optional[np.ndarray]:
        """
        Get positions of human body keypoints for motion tracking.
        
        Returns:
            Array of shape (n_keypoints, 3) with xyz positions,
            or None if keypoints not available
        """
        if not self.connected or not self.human_keypoint_handles:
            return None
        
        keypoints = []
        for handle in self.human_keypoint_handles:
            position = self.sim.getObjectPosition(handle, -1)
            keypoints.append(position)
        
        return np.array(keypoints)
    
    def get_minimum_distance(self) -> float:
        """
        Calculate minimum distance between robot and human.
        
        Returns:
            Minimum distance in meters
            
        Note:
            This is a simplified calculation. For accurate results,
            use the collision detection functions in CoppeliaSim.
        """
        if not self.connected or self.human_handle is None:
            return float('inf')
        
        # Get robot end-effector position
        ee_handle = self.sim.getObject('/UR5/link6/connection')
        ee_pos = np.array(self.sim.getObjectPosition(ee_handle, -1))
        
        # Get human position
        human_pos = self.get_human_position()
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(ee_pos - human_pos)
        
        return distance
    
    def get_simulation_time(self) -> float:
        """
        Get current simulation time.
        
        Returns:
            Simulation time in seconds
        """
        if not self.connected:
            return 0.0
        
        return self.sim.getSimulationTime()
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get complete state vector for LBCF-MPC controller.
        
        Returns:
            State vector [38D]: 
            - Robot joint positions (6D)
            - Robot joint velocities (6D)
            - Robot joint accelerations (6D) - approximated from velocities
            - Human keypoints positions (12D) - 4 keypoints × 3 coords
            - Environment features (8D) - placeholder for objects, velocities
        """
        state = np.zeros(38)
        
        # Robot state (18D)
        joint_pos = self.get_robot_joint_positions()
        joint_vel = self.get_robot_joint_velocities()
        state[0:6] = joint_pos
        state[6:12] = joint_vel
        # Joint accelerations (approximated, would need history)
        state[12:18] = 0.0  # Placeholder
        
        # Human state (12D)
        human_keypoints = self.get_human_keypoints()
        if human_keypoints is not None:
            state[18:30] = human_keypoints.flatten()[:12]
        
        # Environment state (8D) - placeholder
        state[30:38] = 0.0
        
        return state
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def __repr__(self) -> str:
        """String representation."""
        status = "connected" if self.connected else "disconnected"
        return f"<CoppeliaSimInterface({self.host}:{self.port}, {status})>"


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("CoppeliaSim Remote API Test")
    print("="*70)
    
    try:
        # Create interface
        sim_interface = CoppeliaSimInterface()
        
        # Connect
        sim_interface.connect()
        
        # Start simulation
        sim_interface.start_simulation()
        
        # Test: Get robot state
        print("\nRobot State:")
        joint_pos = sim_interface.get_robot_joint_positions()
        print(f"  Joint positions: {joint_pos}")
        
        joint_vel = sim_interface.get_robot_joint_velocities()
        print(f"  Joint velocities: {joint_vel}")
        
        # Test: Get human state
        print("\nHuman State:")
        human_pos = sim_interface.get_human_position()
        if human_pos is not None:
            print(f"  Position: {human_pos}")
        else:
            print("  Human not found in scene")
        
        # Test: Calculate distance
        distance = sim_interface.get_minimum_distance()
        print(f"\nMinimum distance: {distance:.3f}m")
        
        # Test: Get full state vector
        state = sim_interface.get_state_vector()
        print(f"\nFull state vector shape: {state.shape}")
        
        # Run for a few seconds
        print("\nRunning simulation for 5 seconds...")
        start_time = sim_interface.get_simulation_time()
        while (sim_interface.get_simulation_time() - start_time) < 5.0:
            time.sleep(0.02)  # 50 Hz
        
        print("✓ Test completed successfully")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    finally:
        # Cleanup
        sim_interface.stop_simulation()
        sim_interface.disconnect()
    
    print("\n" + "="*70)
    print("Test finished")
    print("="*70)