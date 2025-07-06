# agent_system_assignment

## 実行方法
```bash
source ~/agent_system_ws/devel/setup.bash
roscore
```
```bash
source ~/agent_system_ws/devel/setup.bash
rosrun choreonoid_ros choreonoid projects/shorttrack.cnoid
```
```bash
source ~/agent_system_ws/devel/setup.bash
cd agent_system_ws/src/agent_system_assignment
python3 src/pid_controller.py
```
```bash
rqt_image_view
```

## src/Go2InferenceController.cppを更新したとき
```bash
cd ~/agent_system_ws
catkin build agent_system_assignment
```
