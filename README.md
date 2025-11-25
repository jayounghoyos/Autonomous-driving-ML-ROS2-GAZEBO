# Kaiju Autonomous Driving Project

ROS 2 Kilted Kaiju + Gazebo Ionic + ML/Neural Networks for autonomous driving simulation.

## 🚀 Quick Start

### 1. Build the Docker Image (First Time Only)

```bash
chmod +x build.sh run.sh stop.sh
./build.sh
```

This will:
- Create Ubuntu 24.04 container
- Install ROS 2 Kilted Kaiju
- Install Gazebo Ionic
- Install PyTorch and ML dependencies

**Expected time:** 10-15 minutes

### 2. Run the Development Container

```bash
./run.sh
```

This opens a bash shell inside the container.

### 3. Build the ROS 2 Workspace (Inside Container)

```bash
cd /workspace
colcon build
source install/setup.bash
```

### 4. Test Gazebo

```bash
gz sim shapes.sdf
```

You should see the Gazebo GUI with basic shapes.

### 5. Stop the Container

```bash
exit  # Exit the container shell
./stop.sh  # Stop the container
```

---

## 📁 Project Structure

```
kaiju_ws/
├── src/                    # ROS 2 packages
├── datasets/               # ML training datasets
├── training/               # ML training scripts
├── build/                  # Build artifacts (in Docker volume)
├── install/                # Install artifacts (in Docker volume)
├── log/                    # Build logs (in Docker volume)
├── Dockerfile              # Container definition
├── docker-compose.yml      # Container orchestration
├── build.sh                # Build Docker image
├── run.sh                  # Run container
└── stop.sh                 # Stop container
```

---

## 🛠️ Development Workflow

1. **Edit code** on your host machine (in `src/`, `datasets/`, `training/`)
2. **Build & run** inside the container via `./run.sh`
3. **Changes persist** - volumes are mounted from host

---

## 📦 Installed Software

- **ROS 2 Kilted Kaiju** - Latest rolling release
- **Gazebo Ionic** - Official simulator for Kilted
- **PyTorch** - Deep learning framework
- **Ultralytics (YOLO)** - Object detection
- **OpenCV** - Computer vision
- **Nav2** - ROS 2 navigation stack

---

## 🐛 Troubleshooting

### Gazebo GUI doesn't appear
```bash
# On host, allow Docker X11 access:
xhost +local:docker
```

### Permission errors
```bash
# Fix workspace permissions:
sudo chown -R $USER:$USER .
```

### Container won't start
```bash
# Clean rebuild:
docker compose down -v
./build.sh
```

---

## 📚 Next Steps

See the project phases in the main documentation.
