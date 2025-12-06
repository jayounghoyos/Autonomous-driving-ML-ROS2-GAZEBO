#  Quick Start - RL Training

## Ready to Run Commands

### **Test Gazebo Environment (Do This First!)**

```bash
# Terminal 1: Launch test world
cd ~/PersonalPorjects/Autonomous-driving-ML-ROS2-GAZEBO
gz sim src/vehicle_gazebo/worlds/rl_test_world.sdf
```

You should see:
- Gray ground plane (30×30m)
- Green goal cylinder at (10, 10)
- Red obstacle box at (5, 0)
- Green car with black LiDAR on top

**Test the car manually:**
```bash
# Terminal 2: Move the car with keyboard
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap /cmd_vel:=/cmd_vel
```

**Check LiDAR data:**
```bash
# Terminal 3: Verify LiDAR is publishing
ros2 topic echo /scan --once
```

---

### **Start RL Training (Once Gazebo Test Works)**

```bash
# Terminal 1: Launch Gazebo (headless for faster training)
gz sim -s src/vehicle_gazebo/worlds/rl_training_world.sdf

# Terminal 2: Start training with 8 parallel environments
source ~/ros2-ml-env/bin/activate
source install/setup.zsh
ros2 run rl_training train_ppo \
  --n-envs 8 \
  --timesteps 500000 \
  --device cuda \
  --save-dir models/ppo_navigation \
  --log-dir logs/ppo_navigation

# Terminal 3: Monitor training progress
source ~/ros2-ml-env/bin/activate
tensorboard --logdir logs/ppo_navigation
# Open: http://localhost:6006
```

---

## Training Parameters

### Default (Good Starting Point)
- **Environments**: 8 parallel
- **Timesteps**: 500,000 (~2-3 days)
- **Device**: CUDA (RTX 5060 Ti)
- **Learning Rate**: 3e-4

### Quick Test (Validate Setup)
```bash
# Just 10k steps to verify everything works
ros2 run rl_training train_ppo --n-envs 4 --timesteps 10000 --device cuda
```

### Extended Training (For Best Results)
```bash
# 2 million steps, 16 environments
ros2 run rl_training train_ppo --n-envs 16 --timesteps 2000000 --device cuda
```

---

## What to Expect

### Phase 1: First 10 Minutes (0-10k steps)
- Agent explores randomly
- Lots of collisions
- Reward very low (-50 to 0)
- **This is normal!**

### Phase 2: First Hour (10k-50k steps)
- Agent learns to avoid walls
- Starts moving toward goal sometimes
- Reward slowly increasing
- Success rate: ~10-20%

### Phase 3: First Day (50k-200k steps)
- Smooth obstacle avoidance
- Reaches goal regularly
- Reward stabilizing around +20-50
- Success rate: ~40-60%

### Phase 4: Full Training (200k-500k steps)
- Near-optimal paths
- Rarely collides
- Reward: +50-80
- Success rate: ~80%+

---

## Monitoring Training

### TensorBoard Metrics to Watch

1. **rollout/ep_rew_mean**: Average episode reward
   - Should increase over time
   - Target: +50 or higher

2. **rollout/ep_len_mean**: Average episode length
   - Should decrease (faster to goal)
   - Target: <500 steps

3. **train/policy_loss**: Policy network loss
   - Should decrease and stabilize
   - Spikes are normal

4. **train/value_loss**: Value network loss
   - Should decrease and stabilize

---

## Troubleshooting

### "No module named 'gymnasium'"
```bash
source ~/ros2-ml-env/bin/activate
pip install gymnasium stable-baselines3
```

### "CUDA out of memory"
Reduce parallel environments:
```bash
ros2 run rl_training train_ppo --n-envs 4 --device cuda
```

### Gazebo crashes / freezes
Use headless mode:
```bash
gz sim -s  # Server only, no GUI
```

### Training is very slow
- Check GPU usage: `nvidia-smi`
- Reduce `n_envs` if CPU bottlenecked
- Try `--device cpu` to isolate GPU issues

### Robot doesn't move
```bash
# Check topics are published
ros2 topic list

# Verify cmd_vel works
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.0}}" --once
```

---

## File Locations

### Models (Checkpoints)
- `models/ppo_navigation/ppo_checkpoint_50000_steps.zip`
- `models/ppo_navigation/ppo_checkpoint_100000_steps.zip`
- `models/ppo_navigation/best_model/best_model.zip`  (best performance)
- `models/ppo_navigation/final_model.zip`

### Logs (TensorBoard)
- `logs/ppo_navigation/PPO_*/events.out.tfevents.*`

### Training Worlds
- `src/vehicle_gazebo/worlds/rl_test_world.sdf` - Manual testing
- `src/vehicle_gazebo/worlds/rl_training_world.sdf` - Parallel training

### Vehicle Model
- `src/vehicle_description/urdf/rl_training_car.sdf`

---

## Next Steps After Training

### 1. Test Trained Agent
```bash
# TODO: Create inference node
# ros2 run rl_training test_agent --model models/ppo_navigation/best_model/best_model.zip
```

### 2. Add Obstacle Randomization
Edit `rl_training_world.sdf` or create Python script to spawn random obstacles

### 3. Add Moving Obstacles
Implement dynamic obstacles with velocity plugins

### 4. Improve Reward Function
Tune based on training results:
- Increase goal reward if agent doesn't reach it
- Increase collision penalty if too risky
- Add smoothness reward for better paths

---

## Success Criteria

 **Environment launches without errors**
 **LiDAR data visible in RViz / topic echo**
 **Robot responds to /cmd_vel commands**
 **Training starts and shows progress bar**
 **TensorBoard shows reward curves**
 **Checkpoints save every 50k steps**
 **Agent reaches goal >50% after 200k steps**
 **Agent reaches goal >80% after 500k steps**

---

## Expo Demo Plan

1. **Live Training Visualization**: Show TensorBoard curves
2. **Click-to-Navigate**: Set goal, watch agent drive
3. **Obstacle Challenge**: Add random boxes, agent avoids
4. **Comparison**: Show random policy vs trained policy
5. **Architecture Diagram**: Explain RL training loop

---

**Current Status**:  Ready to test!
**Next Command**: Launch `rl_test_world.sdf` in Gazebo
