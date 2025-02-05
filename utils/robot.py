import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from utils.homogeneous_transformation import HT
import utils.simulate_position as simulate_position

class Robot(object):
    """
    Represents the robot as such as an object.
    This class contains methods to import robot configuration data from a YAML file,
    calculate the forward kinematics of the robot, and convert the position vectors of
    the robot's joints to a dictionary representation.

    Attributes:
        robot: The robot object.
        baseAngles: List of base angles.
        baseDistance: List of base distances.
        leftArmAngles: List of left arm angles.
        rightArmAngles: List of right arm angles.
        headAngles: List of head angles.
        leftArmDistance: List of left arm distances.
        rightArmDistance: List of right arm distances.
        headDistance: List of head distances.
        physical_limits_left: List of physical limits for the left arm.
        physical_limits_right: List of physical limits for the right arm.
        physical_limits_head: List of physical limits for the head.
        robotDict: Dictionary containing robot information.
        robotName: The name of the robot.
    """

    def __init__(self, robotName):
        """
        Initialize the Robot object.

        Args:
            robotName (str): The name of the robot.

        Attributes:
            robot: The robot object.
            baseAngles: List of base angles.
            baseDistance: List of base distances.
            leftArmAngles: List of left arm angles.
            rightArmAngles: List of right arm angles.
            headAngles: List of head angles.
            leftArmDistance: List of left arm distances.
            rightArmDistance: List of right arm distances.
            headDistance: List of head distances.
            physical_limits_left: List of physical limits for the left arm.
            physical_limits_right: List of physical limits for the right arm.
            physical_limits_head: List of physical limits for the head.
            robotDict: Dictionary containing robot information.
            robotName: The name of the robot.
        """
        self.robot = None
        self.baseAngles = []
        self.baseDistance = []
        self.leftArmAngles = []
        self.rightArmAngles = []
        self.headAngles = []
        self.leftArmDistance = []
        self.rightArmDistance = []
        self.headDistance = []
        self.physical_limits_left = []
        self.physical_limits_right = []
        self.physical_limits_head = []
        self.robotDict = {}
        self.robotName = robotName

    def check_angle_limits(self, angles, arm_type):
        """
        Check if the given angles are within the physical limits of the specified arm type.

        Args:
            angles (list): A list of angles to be checked.
            arm_type (str): The type of arm to check the angles for. Can be "left", "right", or "head".

        Returns:
            bool: True if the angles are within the physical limits, False otherwise.
        """
        valid_set = True
        if "left" in arm_type:
            for index, (key, value) in enumerate(self.physical_limits_left.items()):
                if not (value[0] <= angles[index] <= value[0]):
                    valid_set = False
        if "right" in arm_type:
            for index, (key, value) in enumerate(self.physical_limits_right.items()):
                if not (value[0] <= angles[index] <= value[0]):
                    valid_set = False
        if "head" in arm_type:
            for index, (key, value) in enumerate(self.physical_limits_head.items()):
                if not (value[0] <= angles[index] <= value[0]):
                    valid_set = False
        return valid_set

    def read_yaml_file(self, file_path):
        """
        Reads a YAML file and returns the data as a dictionary.

        Args:
            file_path (str): The path to the YAML file.

        Returns:
            dict: The data read from the YAML file.
        """
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data

    def import_robot(self, file_path):
        """
        Imports robot configuration data from a YAML file.

        Args:
            file_path (str): The path to the YAML file containing the robot configuration data.

        Returns:
            None

        Raises:
            FileNotFoundError: If the specified file_path does not exist.

        """
        data = self.read_yaml_file(file_path)
        self.baseAngles = data["base"]["base-distance"]
        self.baseDistance = data["base"]["base-angle"]
        self.leftArmAngles = data["angles-left-arm"]
        self.leftArmDistance = data["distances-left-arm"]
        self.rightArmAngles = data["angles-right-arm"]
        self.rightArmDistance = data["distances-right-arm"]
        self.headAngles = data["angles-head"]
        self.headDistance = data["distances-head"]
        self.robotDict = data["robot-dict"]
        self.robotAgularDict = data["robot-angular-dict"]
        self.physical_limits_left = data["real-angles-left-arm"]
        self.physical_limits_right = data["real-angles-right-arm"]
        self.physical_limits_head = data["real-angles-head"]
        self.dof_arm = data["DOF-right-arm"]

    def forward_kinematics_qt(self, angles):
        '''
        Calculates the spatial position of the end effector and the elbow based on the given angle configuration.

        Args:
            angles (list): A list of angle values representing the joint angles of the robot.

        Returns:
            tuple: A tuple containing the position of all joints of both arms from the closest joint to the base to the end-effector at the end. The tuple contains three lists: pos_leftArm, pos_rightArm, and pos_head.

        Note:
            The angle configuration should be provided in the following order:
            - left_shoulder_pitch
            - left_shoulder_roll
            - left_elbow_roll
            - right_shoulder_pitch
            - right_shoulder_roll
            - right_elbow_roll
            - head_yaw
            - head_pitch
        '''
        a = HT()
        HT_leftArm = []
        HT_rightArm = []
        HT_head = []
        chain_leftArm = []
        chain_rightArm = []
        chain_head = []
        leftArmAngles = {}
        rightArmAngles = {}
        headAngles = {}
        pos_leftArm = []
        pos_rightArm = []
        pos_head = []

        leftArmAngles["collar"] = self.leftArmAngles["collar"]
        leftArmAngles["elbow"] = self.leftArmAngles["elbow"] + np.array([angles[2], 0, 0])
        leftArmAngles["shoulder-roll"] = self.leftArmAngles["shoulder-roll"] + np.array([angles[1], 0, 0])
        leftArmAngles["shoulder-pitch"] = self.leftArmAngles["shoulder-pitch"] + np.array([0, angles[0], 0])
        leftArmAngles["hand"] = self.leftArmAngles["hand"]

        rightArmAngles["collar"] = self.rightArmAngles["collar"]
        rightArmAngles["elbow"] = self.rightArmAngles["elbow"] + np.array([angles[5], 0, 0])
        rightArmAngles["shoulder-roll"] = self.rightArmAngles["shoulder-roll"] + np.array([angles[4], 0, 0])
        rightArmAngles["shoulder-pitch"] = self.rightArmAngles["shoulder-pitch"] + np.array([0, -angles[3], 0])
        rightArmAngles["hand"] = self.rightArmAngles["hand"]

        headAngles["head-yaw"] = self.headAngles["head-yaw"] + np.array([0, 0, angles[6]])
        headAngles["head-pitch"] = self.headAngles["head-pitch"] + np.array([0, angles[7], 0])
        headAngles["collar"] = self.headAngles["collar"]
        headAngles["camera"] = self.headAngles["camera"]

        for key in self.leftArmDistance:
            HT_leftArm.append(a.get_homogeneous_transform(leftArmAngles[key], self.leftArmDistance[key]))
            HT_rightArm.append(a.get_homogeneous_transform(rightArmAngles[key], self.rightArmDistance[key]))

        for key in self.headDistance:
            HT_head.append(a.get_homogeneous_transform(headAngles[key], self.headDistance[key]))

        chain_leftArm.append(HT_leftArm[0])
        chain_rightArm.append(HT_rightArm[0])
        chain_head.append(HT_head[0])

        for i in range(len(self.leftArmDistance) - 1):
            chain_leftArm.append(np.dot(chain_leftArm[i], HT_leftArm[i + 1]))
            chain_rightArm.append(np.dot(chain_rightArm[i], HT_rightArm[i + 1]))

        for i in range(len(self.headDistance) - 1):
            chain_head.append(np.dot(chain_head[i], HT_head[i + 1]))

        for i in range(0, len(chain_leftArm)):
            pos_leftArm.append(a.get_translation(chain_leftArm[i]))
            pos_rightArm.append(a.get_translation(chain_rightArm[i]))

        for i in range(0, len(chain_head)):
            pos_head.append(a.get_translation(chain_head[i]))

        return pos_leftArm, pos_rightArm, pos_head

    def forward_kinematics_nao(self, angles):
        '''
        Calculates the spatial position of the end effector and the elbow based on the given angle configuration.

        Args:
            angles (list): A list of angles representing the joint configuration of the NAO robot.

        Returns:
            tuple: A tuple containing the position of all joints of both arms from the closest joint to the base to the end-effector at the end.

        The joint angles are specified as follows:
        - left_shoulder_pitch: 0
        - left_shoulder_roll: 1
        - left_elbow_yaw: 2
        - left_elbow_roll: 3
        - left_wrist-yaw: 4
        - right_shoulder_pitch: 5
        - right_shoulder_roll: 6
        - right_elbow_yaw: 7
        - right_elbow_roll: 8
        - right_wrist-yaw: 9
        - head_yaw: 10
        - head_pitch: 11
        '''
        a = HT()
        HT_leftArm = []
        HT_rightArm = []
        HT_head = []
        chain_leftArm = []
        chain_rightArm = []
        chain_head = []
        leftArmAngles = {}
        rightArmAngles = {}
        headAngles = {}
        pos_leftArm = []
        pos_rightArm = []
        pos_head = []

        # Calculate left arm angles
        leftArmAngles["torso"] = self.leftArmAngles["torso"]
        leftArmAngles["shoulder-pitch"] = self.leftArmAngles["shoulder-pitch"] + np.array([0, angles[0], 0])
        leftArmAngles["shoulder-roll"] = self.leftArmAngles["shoulder-roll"] + np.array([0, 0, angles[1]])
        leftArmAngles["elbow-yaw"] = self.leftArmAngles["elbow-yaw"] + np.array([angles[2], 0, 0])
        leftArmAngles["elbow-roll"] = self.leftArmAngles["elbow-roll"] + np.array([0, 0, angles[3]])
        leftArmAngles["wrist-yaw"] = self.leftArmAngles["wrist-yaw"] + np.array([angles[4], 0, 0])
        leftArmAngles["hand"] = self.leftArmAngles["hand"]

        # Calculate right arm angles
        rightArmAngles["torso"] = self.rightArmAngles["torso"]
        rightArmAngles["shoulder-pitch"] = self.rightArmAngles["shoulder-pitch"] + np.array([0, angles[5], 0])
        rightArmAngles["shoulder-roll"] = self.rightArmAngles["shoulder-roll"] + np.array([0, 0, angles[6]])
        rightArmAngles["elbow-yaw"] = self.rightArmAngles["elbow-yaw"] + np.array([angles[7], 0, 0])
        rightArmAngles["elbow-roll"] = self.rightArmAngles["elbow-roll"] + np.array([0, 0, angles[8]])
        rightArmAngles["wrist-yaw"] = self.rightArmAngles["wrist-yaw"] + np.array([angles[9], 0, 0])
        rightArmAngles["hand"] = self.rightArmAngles["hand"]

        # Calculate head angles
        headAngles["torso"] = self.headAngles["torso"]
        headAngles["head-yaw"] = self.headAngles["head-yaw"] + np.array([0, 0, angles[10]])
        headAngles["head-pitch"] = self.headAngles["head-pitch"] + np.array([0, angles[11], 0])
        headAngles["camera-top"] = self.headAngles["camera-top"]

        # Calculate homogeneous transforms for left arm
        for key in self.leftArmDistance:
            HT_leftArm.append(a.get_homogeneous_transform(leftArmAngles[key], self.leftArmDistance[key]))

        # Calculate homogeneous transforms for right arm
        for key in self.rightArmDistance:
            HT_rightArm.append(a.get_homogeneous_transform(rightArmAngles[key], self.rightArmDistance[key]))

        # Calculate homogeneous transforms for head
        for key in self.headDistance:
            HT_head.append(a.get_homogeneous_transform(headAngles[key], self.headDistance[key]))

        chain_leftArm.append(HT_leftArm[0])
        chain_rightArm.append(HT_rightArm[0])
        chain_head.append(HT_head[0])

        for i in range(len(self.leftArmDistance) - 1):
            chain_leftArm.append(np.dot(chain_leftArm[i], HT_leftArm[i + 1]))

        for i in range(len(self.headDistance) - 1):
            chain_head.append(np.dot(chain_head[i], HT_head[i + 1]))

        for i in range(0, len(chain_leftArm)):
            pos_leftArm.append(a.get_translation(chain_leftArm[i]))
            pos_rightArm.append(a.get_translation(chain_rightArm[i]))

        for i in range(0, len(chain_head)):
            pos_head.append(a.get_translation(chain_head[i]))

        return pos_leftArm, pos_rightArm, pos_head

    def forward_kinematics_gen3(self, angles):
        '''
        Calculates the spatial position of the end effector and the elbow based on the given angle configuration.

        Args:
            angles (list): A list of joint angles representing the configuration of the robot arm.

        Returns:
            tuple: A tuple containing the position of all joints of both arms from the closest joint to the base to the end-effector at the end. The tuple contains three lists: pos_leftArm, pos_rightArm, and pos_head.

        Note:
            The joint angles are indexed as follows:
            - left_shoulder_pitch --> 0
            - left_shoulder_roll ---> 1
            - left_elbow_yaw -------> 2
            - left_elbow_roll ------> 3
            - left_wrist-yaw -------> 4
            - right_shoulder_pitch -> 5
            - right_shoulder_roll --> 6
            - right_elbow_yaw ------> 7
            - right_elbow_roll -----> 8
            - right_wrist-yaw ------> 9
            - head_yaw ------------> 10
            - head_pitch-----------> 11
        '''
        a = HT()
        HT_leftArm = []
        HT_rightArm = []
        HT_head = []
        chain_leftArm = []
        chain_rightArm = []
        chain_head = []
        leftArmAngles = {}
        rightArmAngles = {}
        headAngles = {}
        pos_leftArm = []
        pos_rightArm = []
        pos_head = []

        leftArmAngles["torso"] = self.leftArmAngles["torso"]
        leftArmAngles["base"] = self.leftArmAngles["base"]
        leftArmAngles["shoulder_link"] = self.leftArmAngles["shoulder_link"] - np.array([0, 0, angles[0]])
        leftArmAngles["half_arm_1_link"] = self.leftArmAngles["half_arm_1_link"] + np.array([0, angles[1], 0])
        leftArmAngles["half_arm_2_link"] = self.leftArmAngles["half_arm_2_link"] + np.array([0, angles[2], 0])
        leftArmAngles["forearm_link"] = self.leftArmAngles["forearm_link"] + np.array([0, angles[3], 0])
        leftArmAngles["spherical_wrist_1_link"] = self.leftArmAngles["spherical_wrist_1_link"] + np.array([0, angles[4], 0])
        leftArmAngles["spherical_wrist_2_link"] = self.leftArmAngles["spherical_wrist_2_link"] + np.array([0, angles[5], 0])
        leftArmAngles["bracelet_link"] = self.leftArmAngles["bracelet_link"] + np.array([0, angles[6], 0])
        leftArmAngles["end_effector_link"] = self.leftArmAngles["end_effector_link"]

        rightArmAngles["torso"] = self.rightArmAngles["torso"]
        rightArmAngles["base"] = self.rightArmAngles["base"]
        rightArmAngles["shoulder_link"] = self.rightArmAngles["shoulder_link"] - np.array([0, 0, angles[7]])
        rightArmAngles["half_arm_1_link"] = self.rightArmAngles["half_arm_1_link"] + np.array([0, angles[8], 0])
        rightArmAngles["half_arm_2_link"] = self.rightArmAngles["half_arm_2_link"] + np.array([0, angles[9], 0])
        rightArmAngles["forearm_link"] = self.rightArmAngles["forearm_link"] + np.array([0, angles[10], 0])
        rightArmAngles["spherical_wrist_1_link"] = self.rightArmAngles["spherical_wrist_1_link"] + np.array([0,  angles[11], 0])
        rightArmAngles["spherical_wrist_2_link"] = self.rightArmAngles["spherical_wrist_2_link"] + np.array([0, angles[12], 0])
        rightArmAngles["bracelet_link"] = self.rightArmAngles["bracelet_link"] + np.array([0, angles[13], 0])
        rightArmAngles["end_effector_link"] = self.rightArmAngles["end_effector_link"]

        headAngles["torso"] = self.headAngles["torso"]

        for key in self.leftArmDistance:
            HT_leftArm.append(a.get_homogeneous_transform(leftArmAngles[key], self.leftArmDistance[key]))
            HT_rightArm.append(a.get_homogeneous_transform(rightArmAngles[key], self.rightArmDistance[key]))

        for key in self.headDistance:
            HT_head.append(a.get_homogeneous_transform(headAngles[key], self.headDistance[key]))

        chain_leftArm.append(HT_leftArm[0])
        chain_rightArm.append(HT_rightArm[0])
        chain_head.append(HT_head[0])

        for i in range(len(self.leftArmDistance) - 1):
            chain_leftArm.append(np.dot(chain_leftArm[i], HT_leftArm[i + 1]))
            chain_rightArm.append(np.dot(chain_rightArm[i], HT_rightArm[i + 1]))

        for i in range(len(self.headDistance) - 1):
            chain_head.append(np.dot(chain_head[i], HT_head[i + 1]))

        for i in range(0, len(chain_leftArm)):
            pos_leftArm.append(a.get_translation(chain_leftArm[i]))
            pos_rightArm.append(a.get_translation(chain_rightArm[i]))

        for i in range(0, len(chain_head)):
            pos_head.append(a.get_translation(chain_head[i]))

        return pos_leftArm, pos_rightArm, pos_head

    def pos_vec_to_robot_dict(self, pos_left, pos_right, pos_head):
        """
        Converts position vectors of the left arm, right arm, and head joints to a dictionary representation.

        Args:
            pos_left (list): List of position values for the left arm joints.
            pos_right (list): List of position values for the right arm joints.
            pos_head (list): List of position values for the head joints.

        Returns:
            dict: A dictionary representation of the robot's joint positions, with keys in the format "joint<Side>_<Index>".
                  The position values are rounded to 2 decimal places.
        """
        for count, i in enumerate(pos_left):
            self.robotDict["jointLeft_" + str(count)] = np.around(i, decimals=2)
        for count, i in enumerate(pos_right):
            self.robotDict["jointRight_" + str(count)] = np.around(i, decimals=2)
        if self.robotName != "gen3":
            for count, i in enumerate(pos_head):
                self.robotDict["jointHead_" + str(count)] = np.around(i, decimals=2)
        return self.robotDict

    def pos_mat_to_robot_mat_dict(self, pos_left, pos_right, pos_head):
        """
        Converts a matrix of positions to a dictionary of robot matrices.

        Args:
            pos_left (list): A list of left positions.
            pos_right (list): A list of right positions.
            pos_head (list): A list of head positions.

        Returns:
            list: A list of robot matrices.

        """
        vec = []
        for i in range(len(pos_left)):
            aux_vec = self.pos_vec_to_robot_dict(pos_left[i], pos_right[i], pos_head[i])
            vec.append(aux_vec.copy())
        return vec

    def angular_vec_to_dict(self, vec):
        """
        Converts an angular vector to a dictionary.

        Args:
            vec (list): The angular vector to be converted.

        Returns:
            dict: A dictionary where the keys are defined in `self.robotAgularDict` and the values are the elements of `vec`.

        """
        angular_dict = {}
        for count, key in enumerate(self.robotAgularDict):
            angular_dict[key] = vec[count]
        return angular_dict

    def angular_mat_to_mat_dict(self, mat):
        """
        Converts an angular matrix to a list of dictionaries.

        Args:
            mat (list): The angular matrix to be converted.

        Returns:
            list: A list of dictionaries representing the angular matrix.
        """
        vec = []
        for i in range(len(mat)):
            vec.append(self.angular_vec_to_dict(mat[i]))
        return vec

    def calculate_distance_of_joints(self):
        """
        Calculates the distances of the joints in the robot's body.

        Returns:
            left_distances (list): List of distances for the left arm joints.
            right_distances (list): List of distances for the right arm joints.
            head_distances (list): List of distances for the head joints.
        """
        left_distances = [np.linalg.norm(self.leftArmDistance[i]) for i in self.leftArmDistance]
        right_distances = [np.linalg.norm(self.rightArmDistance[i]) for i in self.rightArmDistance]
        head_distances = [np.linalg.norm(self.headDistance[i]) for i in self.headDistance]
        return left_distances, right_distances, head_distances

def linear_angular_mapping_gen3(vec):
    """
    Maps the elements of the input vector to a new vector based on a linear-angular mapping.

    Args:
        vec (list): The input vector.

    Returns:
        list: The new vector with elements mapped based on the linear-angular mapping.
    """
    new_vec = []
    for i in vec:
        if i > 0:
            new_vec.append(i)
        else:
            new_vec.append(2*np.pi + i)
    return new_vec

if __name__ == "__main__":
    robotName = "gen3"
    file_path = "./robot_configuration_files/" + robotName + ".yaml"
    myRobot = Robot(robotName)
    myRobot.import_robot(file_path)

    left = []
    right = []
    head = []
    for i in range(0, 370, 10):
        # angles = np.array( [6.283185307179586, 0.08539816, 4.204237027179586, 5.588849777179586, 6.283185307179586, 6.283185307179586, 6.283185307179586,
        #                     6.283185307179586, 5.4977871471795865, 5.351358157179586, 4.8236190171795865, 6.283185307179586, 6.283185307179586, 6.283185307179586])
        # angles = linear_angular_mapping_gen3(angles)
        angles = np.deg2rad(np.array([0, 0, 40, 90, 120, 30, 0,
                                      0, 0, 40, 120, -120, -10, 0]))
        angles = linear_angular_mapping_gen3(angles)
        print(angles)
        pos_left, pos_right, pos_head = myRobot.forward_kinematics_gen3(angles)
        left.append(pos_left)
        right.append(pos_right)
        head.append(pos_head)
    s = simulate_position.RobotSimulation(left, right, head)
    s.animate()