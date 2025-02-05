In order to use the GEN3 module follow the instructions:
- Git clone the kinova kortex: https://github.com/Kinovarobotics/kortex
- Copy the file freddy_controller.py in the folder kortex/api_python/examples/102-Movement_high_level (or create your own example)
- In paralled copy the folder robot_configuration_files and the files homogeneous_transformation.py and robot.py from utils
- In the file robot.py comment the line import simulate_position
- Create the folders to store recorded information in the case of the babbling 
- Copy the folder /data/test_gen3/GMM_learned_actions/for_execution to execute the desired trajectories
- Run the freddy_controller.py file with the console
