In order to use the QTrobot module follow the instructions:
- Make sure to create a QTrobot workspace environment using ROS noetic
- Clone the repository in the workspace
- Copy the files qt_action_player.py, real_time_pose_prediction.py and subscribe_node.py at the same level of the main.py
- qt_action_player.py: Reproduces trajectories generated with GMR and saved in the folder /data/test_nao/GMM_learned_actions/for_execution
- real_time_pose_prediction.py: Executes the real time imitation, reproducing it on the robot
- subscribe_node.py: It is used to perform babbling
- Run the files with ROS 
