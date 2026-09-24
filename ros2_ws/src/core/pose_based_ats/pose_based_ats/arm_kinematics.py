"""Kinematics of the two TacTip arms, for the side-to-side pinch grasp 

The grasp controller needs two things, both from the same chain:

  * ROTATIONS, to express each pad's force in the world. The grip law regulates the part pressing
    into the object's face and reads the weight off the part running along it, and the pads meet
    the faces ~25 deg off square, so a raw sensor axis is not enough.
  * POSITIONS, to measure the object. The shoulder angle at first contact is the width, and the
    grasp axis (the line joining the pads) is the direction the squeeze acts along. Both are read
    off the chain rather than linearised, so a re-measured tube or a re-posed forearm stays right.

ARM NAMING: arm 1 is LEFT (body +y, TacTip B1); arm 2 is RIGHT (body -y, B2).

Frame chain, per arm:

    sensor  (as published by tactip_ros2_driver -- tactip_angle_deg is already applied there,
             so +Z points out of the dome and +Y runs along the housing's mark)
      -> forearm link      MOUNT_R, fixed by how the housing is clamped on the tube
      -> body, URDF FLU    forward kinematics on the MEASURED joint angles
      -> body, PX4 FRD     constant flip
      -> world NED         PX4 attitude quaternion
      -> world ENU, z up   constant swap, so the weight is a plain sum
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


# URDF joint origins (xyz, rpy) and axes, per arm 
# Transcribed from martijn_dual_arm.xacro. The two arms are NOT mirror images in joint space:
# they run the same shoulder locus, and the mirroring comes from the sign of the forearm angle
# (arm 1 folds negative, arm 2 positive). Commanding both forearms the same sign puts one arm out
# to the side instead of under the drone.
_ARM = {
    1: {
        'shoulder_xyz': [0.015, 0.0, 0.0],
        'shoulder_rpy': [1.5708, -1.5708, -1.5708], 'shoulder_axis': [0.0, 0.0, -1.0],
        'motor_xyz':    [0.097, 0.0, 0.014],
        'motor_rpy':    [0.0, 1.5708, 0.0],
        'elbow_xyz':    [0.0, 0.0, 0.0125],
        'elbow_rpy':    [1.5708, 1.5708, 3.14],     'elbow_axis':    [0.0, 0.0, 1.0],
        'forearm_xyz':  [-0.31, 0.0, 0.0],
        'forearm_rpy':  [0.0, 0.0, 0.0],            'forearm_axis':  [0.0, -1.0, 0.0],
    },
    2: {
        'shoulder_xyz': [-0.015, 0.0, 0.0],
        'shoulder_rpy': [1.5708, -1.5708, 1.5708],  'shoulder_axis': [0.0, 0.0, 1.0],
        'motor_xyz':    [0.097, 0.0, 0.0145],
        'motor_rpy':    [0.0, 1.5708, 0.0],
        'elbow_xyz':    [0.0, 0.0, 0.0125],
        'elbow_rpy':    [1.5708, 1.5708, 0.0],      'elbow_axis':    [0.0, 0.0, 1.0],
        'forearm_xyz':  [-0.31, 0.0, 0.0],
        'forearm_rpy':  [0.0, 0.0, 0.0],            'forearm_axis':  [0.0, -1.0, 0.0],
    },
}

# Distance from the tube end (tactip_joint) to the dome's centre, and the dome's radius. 
DOME_OFFSET_M = 0.04075
DOME_RADIUS_M = 0.01125

# Default carbon tube length, forearm joint -> tube end. 0.228 m is the value after the planned
# 50 mm cut, which puts the dome apex 0.280 m from the joint (0.228 + 0.04075 + 0.01125).
DEFAULT_TUBE_M = 0.228

# Body FLU (URDF, z up) -> body FRD (PX4, z down).
R_FRD_FLU = np.diag([1.0, -1.0, -1.0])

# World NED -> world ENU (z up). East = North_y, North = North_x swapped, Up = -Down.
R_ENU_NED = np.array([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0]])


def _T(xyz, rpy):
    M = np.eye(4)
    M[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
    M[:3, 3] = xyz
    return M


def _Rev(axis, q):
    M = np.eye(4)
    M[:3, :3] = R.from_rotvec(np.asarray(axis, dtype=float) * q).as_matrix()
    return M


def mount_rotation(mark_along_plus_z=True):
    """Sensor frame -> forearm link frame, set by how the TacTip is clamped on the tube.

    The mark lies in the arm's folding plane, along the forearm link's +z (or -z), and with the
    mark along +z_forearm: sensor +Z (out of the dome) = -x_forearm, sensor +Y (the mark) =
    +z_forearm, sensor +X = -y_forearm. Use `mark_plus_z()` to get the right sign per arm.

    If the clamp forces a different clocking, measure the offset once and pass it as an extra
    rotation about the sensor's own Z.
    """
    z_s = np.array([-1.0, 0.0, 0.0])
    y_s = np.array([0.0, 0.0, 1.0]) if mark_along_plus_z else np.array([0.0, 0.0, -1.0])
    x_s = np.cross(y_s, z_s)
    return np.column_stack((x_s, y_s, z_s))


# Which end of the forearm tube's OWN z axis the mark lies along, per arm, when it is mounted
# along the arm's FOLDING DIRECTION -- the way the pad travels as the arm folds up.
_MARK_PLUS_Z_WHEN_ALONG_FOLD = {1: True, 2: False}


def mark_plus_z(arm, along_fold_direction=True):
    """Translate the PHYSICAL mounting convention into the per-arm z sign mount_rotation wants.

    `along_fold_direction=True` means the housing's mark points the way the pad travels as the arm
    folds up -- the same gesture on both arms, which is how they are actually mounted.
    """
    plus = _MARK_PLUS_Z_WHEN_ALONG_FOLD[arm]
    return plus if along_fold_direction else (not plus)


def T_body_from_forearm(arm, q1, q2, q3):
    """Forearm link -> body (URDF FLU), full pose, from the measured joint angles of that arm."""
    p = _ARM[arm]
    M = _T(p['shoulder_xyz'], p['shoulder_rpy']) @ _Rev(p['shoulder_axis'], q1)
    M = M @ _T(p['motor_xyz'], p['motor_rpy'])
    M = M @ _T(p['elbow_xyz'], p['elbow_rpy']) @ _Rev(p['elbow_axis'], q2)
    M = M @ _T(p['forearm_xyz'], p['forearm_rpy']) @ _Rev(p['forearm_axis'], q3)
    return M


def R_body_from_forearm(arm, q1, q2, q3):
    """Forearm link -> body (URDF FLU), rotation only."""
    return T_body_from_forearm(arm, q1, q2, q3)[:3, :3]


def R_body_from_sensor(arm, q1, q2, q3, mark_along_plus_z=True):
    """Sensor -> body (URDF FLU)."""
    return R_body_from_forearm(arm, q1, q2, q3) @ mount_rotation(mark_along_plus_z)


def dome_centre_body(arm, q1, q2, q3, tube_m=DEFAULT_TUBE_M):
    """Centre of the contact dome, in body FLU coordinates."""
    M = T_body_from_forearm(arm, q1, q2, q3)
    return (M @ np.array([-(tube_m + DOME_OFFSET_M), 0.0, 0.0, 1.0]))[:3]


def sensor_axis_body(arm, q1, q2, q3):
    """Unit vector out of the dome (the sensor's +Z), in body FLU coordinates."""
    return R_body_from_forearm(arm, q1, q2, q3) @ np.array([-1.0, 0.0, 0.0])


def grasp_axis_body(q_arm1, q_arm2, tube_m=DEFAULT_TUBE_M):
    """Unit vector from the right pad toward the left pad (arm 2 -> arm 1), in body FLU.

    This is the direction the squeeze acts along. Arm 1 presses along -axis, arm 2 along +axis.
    Taken from the measured angles rather than assumed to be body +y, so a shoulder that has not
    reached its commanded angle does not silently rotate the force decomposition.
    """
    c1 = dome_centre_body(1, *q_arm1, tube_m=tube_m)
    c2 = dome_centre_body(2, *q_arm2, tube_m=tube_m)
    d = c1 - c2
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None
    return d / n


def press_direction_body(arm, q_arm1, q_arm2, tube_m=DEFAULT_TUBE_M):
    """Unit vector along which `arm`'s pad pushes the object, in body FLU."""
    a = grasp_axis_body(q_arm1, q_arm2, tube_m=tube_m)
    if a is None:
        return None
    return -a if arm == 1 else a


def grasp_width_m(q_arm1, q_arm2, tube_m=DEFAULT_TUBE_M):
    """Distance between the two pads' CONTACT POINTS, along the grasp axis.

    A sphere resting on a flat face touches at its extreme point, one dome radius in from the
    centre, so the two radii come off the centre-to-centre distance. At first contact this is the
    object's width; squeezing further makes it read the width minus the total indentation.
    """
    a = grasp_axis_body(q_arm1, q_arm2, tube_m=tube_m)
    if a is None:
        return None
    c1 = dome_centre_body(1, *q_arm1, tube_m=tube_m)
    c2 = dome_centre_body(2, *q_arm2, tube_m=tube_m)
    return float((c1 - c2) @ a - 2.0 * DOME_RADIUS_M)


def contact_angle_deg(arm, q_arm1, q_arm2, tube_m=DEFAULT_TUBE_M):
    """Angle between the pad's sensor axis and the face it presses, measured off square.

    Zero means the dome meets the face head-on. This is the quantity that decides how far outside
    the TacTip models' +-25 deg trained tilt the grasp is sitting, so the controller logs it rather
    than assuming the design value. In a side grasp the faces are vertical, so the reference is
    the grasp axis.
    """
    n = press_direction_body(arm, q_arm1, q_arm2, tube_m=tube_m)
    if n is None:
        return None
    q = q_arm1 if arm == 1 else q_arm2
    axis = sensor_axis_body(arm, *q)
    return float(np.degrees(np.arccos(np.clip(abs(axis @ n), 0.0, 1.0))))


def R_world_from_body(px4_quat_wxyz):
    """Body (URDF FLU) -> world ENU (z up), from the PX4 attitude quaternion.

    px4_quat_wxyz is PX4's [w, x, y, z] describing NED <- FRD. Returns None rather than a guess
    when the attitude is unusable, so a missing quaternion shows up as "no estimate" instead of a
    silently wrong force direction.
    """
    q = np.asarray(px4_quat_wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1.0) > 0.1:
        return None
    # scipy wants [x, y, z, w]
    R_ned_frd = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return R_ENU_NED @ R_ned_frd @ R_FRD_FLU


def world_force(arm, q1, q2, q3, force_sensor, px4_quat_wxyz, mark_along_plus_z=True):
    """The pad's force expressed in the world (ENU, z up).

    The models report the force the pad applies TO the object, so no sign flip is needed: the
    vertical components of the two pads sum to the object's weight in steady hover.
    Returns None when the attitude is not yet available.
    """
    R_wb = R_world_from_body(px4_quat_wxyz)
    if R_wb is None:
        return None
    return R_wb @ R_body_from_sensor(arm, q1, q2, q3, mark_along_plus_z) @ np.asarray(
        force_sensor, dtype=float)


def decompose_force(f_world, press_dir_world):
    """Split a pad force into the part pressing into the face and the part running along it.

    Returns (normal_n, tangential_vec_world). `normal_n` is positive when the pad is genuinely
    pressing; the tangential part is what friction has to hold, and for a hanging object it is
    dominated by the object's weight.
    """
    f = np.asarray(f_world, dtype=float)
    n = float(f @ press_dir_world)
    return n, f - n * np.asarray(press_dir_world, dtype=float)
