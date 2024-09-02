import time

import mujoco
import mujoco.viewer


def main():
    m = mujoco.MjModel.from_xml_path(
        # "/Users/zmk5/Developer/robot-learning/mujoco_menagerie/unitree_a1/a1.xml"
        "/Users/zmk5/Developer/robot-learning/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
    )
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < 10:
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
