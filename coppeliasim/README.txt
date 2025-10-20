# CoppeliaSim Simulation Files

This directory contains the CoppeliaSim simulation scene and interface files used for LBCF-MPC experiments.

---

## Files

| File | Description |
|------|-------------|
| `Scene_in_CoppeliaSim_for_Mathematics.ttt` | Complete simulation scene with UR5 robot and human operator |
| `Escena_3.png` | Screenshot of Scenario 3 (Simultaneous Operation) |
| `python_remote_api.py` | Python interface for CoppeliaSim ZMQ Remote API |

---

## Setup Instructions

### 1. Install CoppeliaSim

Download and install **CoppeliaSim EDU 4.10.0** or later:
- Website: https://www.coppeliarobotics.com/downloads
- Choose EDU version (free for academic use)
- Supported platforms: Windows, Linux, macOS

### 2. Install Python Package

```bash
pip install zmqRemoteApi
```

### 3. Load the Scene

1. Open CoppeliaSim
2. File → Open scene...
3. Select `Scene_in_CoppeliaSim_for_Mathematics.ttt`
4. The scene should load with:
   - UR5 robot (right side)
   - Human operator (left side)
   - Conveyor belt with colored objects
   - Destination stations (foreground)
   - Workspace layout

---

## Using the Simulation

### Basic Usage

```python
from coppeliasim.python_remote_api import CoppeliaSimInterface

# Create interface
sim = CoppeliaSimInterface()

# Connect to CoppeliaSim
sim.connect()

# Start simulation
sim.start_simulation()

# Get robot state
joint_positions = sim.get_robot_joint_positions()
joint_velocities = sim.get_robot_joint_velocities()

# Get human position
human_pos = sim.get_human_position()

# Calculate safety distance
distance = sim.get_minimum_distance()
print(f"Distance: {distance:.3f}m")

# Control robot
new_positions = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]
sim.set_robot_joint_positions(new_positions)

# Stop when done
sim.stop_simulation()
sim.disconnect()
```

### With Context Manager

```python
from coppeliasim.python_remote_api import CoppeliaSimInterface

with CoppeliaSimInterface() as sim:
    sim.start_simulation()
    
    # Your control loop
    for _ in range(100):
        state = sim.get_state_vector()
        # Process state...
        time.sleep(0.02)  # 50 Hz
```

---

## Scene Details

### Robot: Universal Robots UR5

- **Workspace:** Spherical, radius ~850mm
- **Degrees of Freedom:** 6
- **Joint Limits:** Standard UR5 limits
- **Control Mode:** Position control via Remote API

### Human Operator

- **Model:** Simplified humanoid with 4 keypoints
- **Motion:** Based on CMU MoCap database
- **Keypoints:** Torso, left arm, right arm, head
- **Tracking:** Real-time position updates

### Objects

- **Colored Parts:** Red, green, blue, yellow cylinders
- **Conveyor Belt:** 0.4 m/s nominal speed
- **Destination Stations:** 4 stations color-coded

### Safety Parameters

- **Minimum Distance Threshold:** 0.15m (ISO/TS 15066)
- **Collision Detection:** FCL library integration
- **Update Rate:** 50 Hz

---

## API Reference

### CoppeliaSimInterface

#### Methods

**Connection:**
- `connect()` - Connect to CoppeliaSim
- `disconnect()` - Disconnect from CoppeliaSim
- `start_simulation()` - Start simulation
- `stop_simulation()` - Stop simulation

**Robot Control:**
- `get_robot_joint_positions()` - Get joint angles [6D]
- `set_robot_joint_positions(positions)` - Set target positions
- `get_robot_joint_velocities()` - Get joint velocities [6D]

**Human Tracking:**
- `get_human_position()` - Get torso position [3D]
- `get_human_keypoints()` - Get all keypoints [4×3D]

**Safety:**
- `get_minimum_distance()` - Calculate robot-human distance
- `get_state_vector()` - Get complete state [38D]

**Utilities:**
- `get_simulation_time()` - Current simulation time

---

## Scenarios

The scene supports 4 experimental scenarios:

### Scenario 1: Coexistence
- Human walks through workspace
- Robot performs independent pick-and-place
- Typical distance: 0.8-1.2m

### Scenario 2: Sequential Collaboration
- Human pre-processes objects
- Places in handover zone
- Robot retrieves and transports
- Typical distance: 0.3-0.6m

### Scenario 3: Simultaneous Operation 
- Human picks defective parts (red)
- Robot picks good parts (green/blue/yellow)
- Both work on same conveyor
- Typical distance: 0.15-0.35m
- **Most challenging scenario**

### Scenario 4: Adaptive Coordination
- Dynamic task allocation
- Real-time replanning
- Ergonomic considerations
- Typical distance: 0.18-0.40m

---

## Troubleshooting

### Cannot connect to CoppeliaSim

**Problem:** `ConnectionError: Cannot connect to CoppeliaSim`

**Solutions:**
1. Ensure CoppeliaSim is running
2. Check the scene is loaded
3. Verify ZMQ Remote API is enabled in CoppeliaSim
4. Check firewall settings (port 23000)

### Object handles not found

**Problem:** `Could not find joint: /UR5/joint`

**Solutions:**
1. Verify the scene file is loaded correctly
2. Check object names in CoppeliaSim scene hierarchy
3. Ensure you're using the correct scene file version

### Simulation runs slowly

**Solutions:**
1. Reduce simulation dt (Edit → User Settings → Simulation)
2. Disable unnecessary rendering
3. Use headless mode for batch experiments
4. Consider simplifying collision detection

---

## Additional Resources

- **CoppeliaSim Documentation:** https://www.coppeliarobotics.com/helpFiles/
- **ZMQ Remote API:** https://www.coppeliarobotics.com/helpFiles/en/zmqRemoteApiOverview.htm
- **UR5 Manual:** https://www.universal-robots.com/download/
- **Paper:** Mathematics (MDPI) 2025 - DOI: 10.3390/mathXXXXXXX
- **Dataset:** FigShare - DOI: 10.6084/m9.figshare.30282127

---

## Support

For issues specific to the simulation setup:
- GitHub Issues: https://github.com/ClaudioUrrea/ur5-human_CoppeliaSim_EDU/issues
- Email: claudio.urrea@usach.cl

---

**Note:** The scene file and Python interface are configured for the experimental setup described in the paper. Modifications may be needed for different robot models or scenarios.