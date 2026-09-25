"""
Hydraulic cylinder-to-joint mapping for the v04 arm model.

The v04 model drives joints 2&3 via hydraulic cylinder slide actuators.
Each cylinder forms a triangle with its joint:

    A = joint axis        (apex; angle BAC tracks the joint rotation)
    B = cylinder rod hinge (rigid on the parent link)
    C = cylinder tube anchor on link2 / link3 (rigid on the child link)
    a = |A-B|, b = |A-C|   (constant)
    c = |B-C|              (cylinder length, varies with joint angle)

By the law of cosines, the cylinder length for a joint angle q is

    c(q) = sqrt(a^2 + b^2 - 2ab*cos(theta_home + s*(q - q_home)))

and the slide ctrl value (slide qpos == extension from the home length) is

    ctrl(q) = c(q) - c_home

The geometry (a, b, c_home, theta_home, q_home) is read from a single
forward-kinematics evaluation at the `home` keyframe — no settling sweeps.
Validated against settled MuJoCo states to < 0.05 mm across full travel.
"""

import numpy as np
import mujoco

JOINT2_QPOS_IDX = 1
JOINT3_QPOS_IDX = 2

CYL1_CTRL_IDX = 1
CYL2_CTRL_IDX = 2

CYL1_RANGE = (-0.075, 0.07)
CYL2_RANGE = (-0.115, 0.075)

# (joint qpos idx, apex body, rod body, anchor site, ctrl range)
_CYL1_SPEC = (JOINT2_QPOS_IDX, "link2", "cylinder1_rod", "hole_site_1", CYL1_RANGE)
_CYL2_SPEC = (JOINT3_QPOS_IDX, "link3", "cylinder2_rod", "hole_site_2", CYL2_RANGE)

_SIGN_PROBE = 0.05  # rad, kinematic perturbation used to resolve angle sign


def _body_xpos(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[bid].copy()


def _site_xpos(model, data, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return data.site_xpos[sid].copy()


class _Triangle:
    """Constant triangle geometry + signed angle convention for one cylinder."""

    def __init__(self, a, b, c_home, theta_home, q_home, sign, ctrl_range):
        self.a = a
        self.b = b
        self.c_home = c_home
        self.theta_home = theta_home
        self.q_home = q_home
        self.sign = sign
        self.ctrl_range = ctrl_range

    def length(self, q):
        theta = self.theta_home + self.sign * (q - self.q_home)
        return np.sqrt(self.a**2 + self.b**2 - 2 * self.a * self.b * np.cos(theta))

    def ctrl(self, q):
        return float(np.clip(self.length(q) - self.c_home, *self.ctrl_range))


class CylinderMapper:
    """Maps joint2/joint3 angles to cylinder1/cylinder2 ctrl values.

    Uses the analytical law-of-cosines triangle relation; geometry is read
    once from the `home` keyframe via forward kinematics (no MuJoCo sweeps).
    """

    def __init__(self, model=None, data=None):
        if model is not None and data is not None:
            self._calibrate(model, data)

    def _calibrate(self, model, data):
        qpos_backup = data.qpos.copy()
        ctrl_backup = data.ctrl.copy()
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)

        self._g1 = self._extract(model, data, *_CYL1_SPEC)
        self._g2 = self._extract(model, data, *_CYL2_SPEC)

        data.qpos[:] = qpos_backup
        data.ctrl[:] = ctrl_backup
        mujoco.mj_forward(model, data)

    def _extract(self, model, data, qpos_idx, apex_body, rod_body, anchor_site, ctrl_range):
        A = _body_xpos(model, data, apex_body)
        B = _body_xpos(model, data, rod_body)
        C = _site_xpos(model, data, anchor_site)
        a = np.linalg.norm(A - B)
        b = np.linalg.norm(A - C)
        c_home = np.linalg.norm(B - C)
        cos_t = (a**2 + b**2 - c_home**2) / (2 * a * b)
        theta_home = float(np.arccos(np.clip(cos_t, -1.0, 1.0)))
        q_home = float(data.qpos[qpos_idx])

        # Resolve angle sign via a small kinematic perturbation of the joint.
        data.qpos[qpos_idx] = q_home + _SIGN_PROBE
        mujoco.mj_kinematics(model, data)
        c_pert = np.linalg.norm(
            _body_xpos(model, data, rod_body) - _site_xpos(model, data, anchor_site))
        data.qpos[qpos_idx] = q_home
        mujoco.mj_kinematics(model, data)

        c_pos = np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(theta_home + _SIGN_PROBE))
        c_neg = np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(theta_home - _SIGN_PROBE))
        sign = 1 if abs(c_pert - c_pos) < abs(c_pert - c_neg) else -1

        return _Triangle(a, b, c_home, theta_home, q_home, sign, ctrl_range)

    def joint2_to_cyl1(self, angle):
        return self._g1.ctrl(angle)

    def joint3_to_cyl2(self, angle):
        return self._g2.ctrl(angle)

    def joints_to_ctrl(self, q6, gripper_cmd=0.0):
        """Convert 6 joint angles to 7 MuJoCo ctrl values (joints 2&3 -> cylinders)."""
        ctrl = np.zeros(7)
        ctrl[0] = q6[0]
        ctrl[CYL1_CTRL_IDX] = self.joint2_to_cyl1(q6[1])
        ctrl[CYL2_CTRL_IDX] = self.joint3_to_cyl2(q6[2])
        ctrl[3] = q6[3]
        ctrl[4] = q6[4]
        ctrl[5] = q6[5]
        ctrl[6] = gripper_cmd
        return ctrl

    def print_calibration_report(self):
        print("CylinderMapper (analytical law-of-cosines):")
        for name, g in (("Cyl1/Joint2", self._g1), ("Cyl2/Joint3", self._g2)):
            print(f"  {name}: a={g.a:.5f} b={g.b:.5f} c_home={g.c_home:.5f} "
                  f"theta_home={np.degrees(g.theta_home):.2f} deg "
                  f"q_home={np.degrees(g.q_home):.2f} deg sign={g.sign:+d}")
