# Setting up the package

## Creating the package wrapper
  First we need to create a package wrapper to run this module. As such, we should open a terminal and execute the following commands:

> cd ~/dev_ws/src
> ros2 pkg create --build-type ament_python 'usi_angry_turtle'


## Setting files
  Once this is done, unzip and copy the whole folder to ~/dev_ws/src .


# Running the module
  At this point everything is nearly ready to run the module. We just need to compile and source the new build. As such, open 3 terminals and do the following in order:

On terminal 1:
> cd ~/dev/ws
> colcon build

On terminal 2:
> source ~/dev_ws/install/setup.bash
> ros2 run turtlesim turtlesim_node

On terminal 3:
> source ~/dev_ws/install/setup.bash
> ros2 run usi_angry_turtle move2goal_node


If everything went as expected, the module should have started running in the simulator.


# Additional Notes
  All the exercises, 1 through 4, have been fulfilled. That is, the turtle effectively writes 'USI' and in the meantime gets angry if a turtle gets too close, chasing it until it manages to kill it. Once having killed the offender, the turtle goes back to writing. In the meantime, if it encounters another turtle too close, it will start chasing it again.

  Every time an offender is being chased, the turtle predicts its path according to its current pose, linear and angular velocities, up to 0.5 seconds later. When an offender is killed, a timer will ensure that another is spawned to take its place.

The module is a node that encompasses all the logic for both the offenders and the main turtle.