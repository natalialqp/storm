In order to use the NAO module follow the instructions:
- Make sure to connect NAO with Choreograph in Ubuntu 16
- Create a folder to paste the following documents
- Copy the file nao_controller.py in the folder
- In paralled copy the nao files from the folder robot_configuration_files at the same label of the nao_controller.py file
- Copy the files homogeneous_transformation.py and robot.py from utils
- In the file robot.py comment the line import simulate_position
- Copy the files from /data/test_nao/GMM_learned_actions/for_execution at the same label of the nao_controller.py file to execute the desired trajectories
- Run the nao_controller.py file with the console 
