import argparse

from manager import TeleopManager

# TODO: make it client server
# create a tv.step() thread and request image

# Teleop and datacollector
# Starts lidar process and robot arm/hand controllers
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Teleoperation Data Collector")
    parser.add_argument(
        "--task_name", type=str, default="default_task", help="Name of the task"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    manager = TeleopManager(task_name=args.task_name, debug=args.debug)
    manager.start_processes()
    manager.run_command_loop()
    # TODO: run in two separate terminals for debuggnig
