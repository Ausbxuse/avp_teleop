up+ and down is z
left+ and right is y
front+ and behind x

# Data Collection Instruction
1. install python tv environment on your computer
2. connect your computer to local router wifi
3. open robot and set to developer mode
4. run `python main.py --task_name <your_task_name>` under avp_teleop/teleop/ directory and wait until the robot is in ready state (90-degree arm and clenched fists). The terminal should signal both "master" and "worker" processes are waiting for starting signal
5. connect to robot using https://<your_local_ip>:8012/?ws=wss://<your_local_ip>:8012
6. enter s to start recording an episode. For the first episode, also record a video of it using a stable camera at a nice angle.
7. type `q`  if the episode is successful, `d` otherwise to discard the last session
8. repeat by pressing `s` to start recording the next episode. record 40 episodes for each task
9. type `exit` to finish recording the task

# Todos
- [ ] task list comparison (agibot/fourier)
  - [ ] task list refinement
- [ ] lerobot integration
  - [ ] data visualizer
- [ ] data validater (check if collected data is correct/complete)
- [ ] unitests for single files

- [ ] FIXME: first ik and second ik has missing data in between for 1.5s (possibly due to visionpro+motor safety code in arm_controller)
