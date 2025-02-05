<!--
* STORM: A Graph-Based Learning from Demonstration Framework
*
* Copyright (c) 2024 Natalia Quiroga Perez. All rights reserved.
*
* This software may be modified and distributed
* under the terms of the BSD 3-Clause license.
*
* Refer to the LICENSE file for details.
*
-->

<h1>Graph-Based Learning from Demonstration Framework<sup>®</sup></h1>

<a id="markdown-description" name="description"></a>
# Description

The official repository contains documentation and datasets for QTrobot, NAO and Kinova gen3. This Framework runs in Python.
The repository has been tested on Ubuntu 20.04.

<h1>Table of Contents</h1>

<!-- TOC -->

- [Description](#description)
- [Configuration and Setup](#config)
- [Application](#application)
- [Quick Start](#quickstart)
- [Reference](#reference)
      - [Useful Links](#useful-links)

<a id="description" name="description"></a>
<h1> Description</h1>

This repository stores a learning from demonstration framework. It focuses on the upper-body limb for the robots QTrobot, NAO and Freddy (2 Kinova arms Gen3 on a Kelo base), any other humanoid robot can be added following the instruccions.

You are going to find main code in the file main.py. The one has different options to perform different steps of learning from demonstration.

The results are placed in the folder data. Where the sub-folder test_qt, test_nao and test_gen3 contain self-exploration files (babbling files), objects/obstacles to add in the world, paths and trajectories generated for pre-recorded actions, robot's joint's graphs and plots.


<a id="config" name="Configuration"></a>
<h1> Configuration and Setup</h1>

<li>For the config.yaml file please fill the following details:</li>
<ul>
      <li> Add the name of the robot (qt, nao or gen3) or your own</li>
      <li> Define the amount of babbling points (30, 100 or 150) or your own</li>
      <li> The iteration-id indicate which iteration over the pre-recorded actions is performed. This value is only required in order to keep track of the saved plots.  </li>
      <li> minimum-distance is the quality of the graph d_min in mm </li>
      <li> low-edge-* and high-edge-* are oposite edges of a squared workspace for a robot, the values are in mm. To add a new robot, add the robot-name and the desired world </li>
      <li> obstacle-name refers to a obstacle/object that needs to be created or incorporated into the robot's workspace </li>
      <li> low-edge-* and high-edge-* are oposite edges of a squared object/obstacle wished to be added into the robot's workspace </li>
      <li> For the pre-recorded actions the actions-names are defined. To include a new action, make sure to add it to the combined_actions_*.csv, human_pose*.csv and robot_angles_*.csv               files with its respective used id</li>
      <li> users-id for the pre-recorded actions go from 1 to 20, make sure to start the count from 21 for new actions and users </li>
      <li> The amount of epochs used to train the Neural Network to learn robot angles from human skeleton angles </li>
      <li> To read a new action change the value of read-new-action to True, the default is False to read pre-recorded actions</li>
      <li> For any new action specify the name: new-action-name </li>
      <li> users-id-new-action can take any integer value exepting 1 to 20</li>
      <li> To read a trajectory, the name of it is the userId+actionName: for example "5spoon"
      </li>
      <li> The number of components in the Gaussian Mixture Model is num-components</li>
      <li> Number of components in the Gaussian Mixture Regression using the gmr library
      num-clusters </li>
      <li> To perform GMR and GMM the trajectory-name has to be specified</li>
</ul>

<a id="application" name="Application"></a>
<h1> Application</h1>

<li> Depending on your application, select the corresponding function to run the main.py file: </li>
<ul>
      <li>"path-planning": To calculate the path planning of the robot arms or limbs. The planning takes place in the workspace of the robot (each joit graph)</li>
      <li>"pose-predicition": Iterates over the actions and the users and predicts the pose of the robot, calculates posible new nodes inside the robot's graph and updates the graph and               the path library</li>
      <li>"explore-world": Reads the babbling points explored with the physical robot and calculates the forward kinematics of the robot to update the robot's graph, interpolates nodes in             beetween babbling points and updates the stores the joint's graphs</li>
      <li>"create-object": Creates an object/obstacle with the specified edges and stores it on a file</li>
      <li>"object-in-robot-graph": Reads the object/obstacle from a file and stores it on a graph, calculates the overlap between the object and the robot's graph and updates the robot's             graph removing the overlapped nodes</li>
      <li>"read-library-paths": Read the existing trajectories from the library and adjust (performs path planning) the trajectory according to the inclusion of a new obstacle/object in               the robot's workspace</li>
      <li>"gaussian-mixture-regression": Calculates the GMR of the actions and stores the GMR trajectories in a file ready to be used by the robot</li>
</ul>


<li>For action reproduction on the robots:</li>
<ul>
      <li> "reproduce-action": Reproduces the action on the robot, make sure to specify the robot's name and the action name</li>
      <li> "motor-babbling": Babble with the robot, make sure to specify the robot's name and the amount of babbling points</li>
      <li> In order to use NAO, the nao-ip: "192.168.0.101" has to be adjusted
      </li>
      <li> In order to use Freddy, make sure to adjust the IP address of both arms or one arm, make sure to set left and right as ip-1 and ip-2 
      </li>
      <li> Specify the action that needs to be reproduced on the robot (NAO, QTrobot or Freddy) in action-name
      </li>
</ul>

<a id="quickstart" name="quickstart"></a>
<h1> Quick Start </h1>
  <li>To run the Python examples you will need to install the Python3 and run the requirements.txt file</li>
  <ul>
      <li>  QTrobot: uses ROS Noetic, make sure to connect to the robot following the LuxAI documentation</li>
      <li>  NAO: runs on Ubuntu 16, make sure to have python 2.7 an Choregraphe installed </li>
      <li>  Freddy: clone the kortex repository outside of the imitation_pkg and copy the required files, follow the instructions in freddy_controller.py</li>
  </ul>
<h1> Reference </h1>

Useful Links

|  |  |
| ---: | --- |
| QTrobot quick start: | [https://docs.luxai.com/docs/intro_code](https://docs.luxai.com/docs/intro_code) |
| Kinova kortex: | [https://www.kinovarobotics.com](https://github.com/Kinovarobotics/kortex)|
| NAO documentation: | [http://doc.aldebaran.com/2-8/family/nao_technical/index_naov6.html](http://doc.aldebaran.com/2-8/family/nao_technical/index_naov6.html)|
| GMR repository: | [https://github.com/AlexanderFabisch/gmr](https://github.com/AlexanderFabisch/gmr)|
