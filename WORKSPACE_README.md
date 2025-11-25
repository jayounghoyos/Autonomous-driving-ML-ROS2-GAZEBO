# Kaiju Autonomous Driving Workspace

ROS 2 Kilted Kaiju workspace for autonomous driving simulation with ML/Neural Networks.

## 📁 Workspace Structure

```
kaiju_ws/
├── src/                          # ROS 2 packages
│   ├── vehicle_description/      # URDF/SDF robot models
│   ├── vehicle_gazebo/           # Gazebo worlds & launch files
│   ├── ml_perception/            # ML perception (YOLO, lane detection)
│   ├── data_collection/          # Record training data
│   ├── vehicle_control/          # Control algorithms
│   └── autonomous_nav/           # Path planning & navigation
├── datasets/                     # ML training datasets
├── training/                     # ML training scripts
├── build/                        # Colcon build artifacts
├── install/                      # Installed ROS packages
└── log/                          # Build logs
```

## 🚀 Quick Start

### 1. Build the Workspace

```bash
./run.sh
# Inside container:
colcon build
source install/setup.bash
```

### 2. Test the Build

```bash
ros2 pkg list | grep -E "vehicle|ml_perception|autonomous"
```

You should see:
- autonomous_nav
- data_collection
- ml_perception
- vehicle_control
- vehicle_description
- vehicle_gazebo

## 📦 Package Descriptions

### **vehicle_description**
Robot models in URDF/SDF format with sensors (camera, LiDAR, IMU, GPS).

**Key files:**
- `urdf/` - Vehicle model definitions
- `meshes/` - 3D meshes for visualization
- `config/` - Sensor configurations

### **vehicle_gazebo**
Gazebo Ionic simulation environment.

**Key files:**
- `worlds/` - Gazebo world files (roads, obstacles)
- `models/` - Custom Gazebo models
- `launch/` - Launch files to start simulation
- `config/` - Gazebo bridge configurations

### **ml_perception**
ML-based perception using PyTorch and YOLO.

**Nodes:**
- `yolo_detector` - Object detection (cars, pedestrians, signs)
- `lane_detector` - Lane line detection

**Topics:**
- `/camera/image_raw` (input)
- `/detections` (output)
- `/lane_markers` (output)

### **data_collection**
Record sensor data for ML training.

**Nodes:**
- `data_recorder` - Saves images + control commands

**Output:**
- Images → `datasets/images/`
- Labels → `datasets/labels/`
- Metadata → `datasets/metadata.json`

### **vehicle_control**
Low-level control algorithms.

**Nodes:**
- `pid_controller` - PID control for steering/throttle
- `teleop_keyboard` - Manual keyboard control

### **autonomous_nav**
High-level planning and decision making.

**Nodes:**
- `waypoint_follower` - Follow waypoint paths
- `obstacle_avoider` - Collision avoidance

## 🎯 Next Steps

1. **Chunk 3**: Create a basic vehicle model
2. **Chunk 4**: Set up Gazebo world with roads
3. **Chunk 5**: Implement YOLO object detection
4. **Chunk 6**: Implement data collection
5. **Chunk 7**: Create autonomous navigation pipeline

## 🐛 Troubleshooting

**Build fails:**
```bash
# Clean build
rm -rf build install log
colcon build
```

**GPU not detected:**
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Gazebo won't start:**
```bash
# Check if running
gz sim --version
```
