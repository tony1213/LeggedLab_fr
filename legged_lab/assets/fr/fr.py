"""Configuration for FR02 humanoid robot.

FR02 is a miniature humanoid robot (~50cm tall, ~1.2kg) with 27 revolute joints.

Kinematic structure:
  - Legs (6 DOF each): hip_pitch -> hip_roll -> hip_yaw -> knee_pitch -> ankle_pitch -> ankle_roll
  - Waist/Chest (3 DOF): waist_yaw -> chest_roll -> chest_pitch
  - Arms (5 DOF each): shoulder_pitch -> shoulder_roll -> upper_arm_yaw -> elbow_pitch -> wrist_roll
  - Head (2 DOF): head_yaw -> head_pitch
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

FR02_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/fr/fr02/fr02.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_pitch_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            "l_shoulder_roll_joint": -1.3,
            "r_shoulder_roll_joint": 1.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_pitch_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 0.4,
                ".*_hip_roll_joint": 1.6,
                ".*_hip_pitch_joint": 1.6,
                ".*_knee_pitch_joint": 1.6,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 10.472,
                ".*_hip_roll_joint": 6.283,
                ".*_hip_pitch_joint": 6.283,
                ".*_knee_pitch_joint": 6.283,
            },
            stiffness=9.156,
            damping=0.305,
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 1.6,
                ".*_ankle_roll_joint": 1.2,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 6.283,
                ".*_ankle_roll_joint": 6.283,
            },
            stiffness=9.156,
            damping=0.305,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim={"waist_yaw_joint": 1.2},
            velocity_limit_sim={"waist_yaw_joint": 6.283},
            stiffness=4.883,
            damping=0.153,
            armature=0.01,
        ),
        "chest": ImplicitActuatorCfg(
            joint_names_expr=["chest_roll_joint", "chest_pitch_joint"],
            effort_limit_sim={
                "chest_roll_joint": 1.2,
                "chest_pitch_joint": 1.2,
            },
            velocity_limit_sim={
                "chest_roll_joint": 6.283,
                "chest_pitch_joint": 6.283,
            },
            stiffness=2.442,
            damping=0.153,
            armature=0.01,
        ),
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 0.4,
                ".*_shoulder_roll_joint": 0.4,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 5.236,
                ".*_shoulder_roll_joint": 5.236,
            },
            stiffness=1.831,
            damping=0.092,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_upper_arm_yaw_joint",
                ".*_elbow_pitch_joint",
            ],
            effort_limit_sim={
                ".*_upper_arm_yaw_joint": 0.2,
                ".*_elbow_pitch_joint": 0.4,
            },
            velocity_limit_sim={
                ".*_upper_arm_yaw_joint": 8.378,
                ".*_elbow_pitch_joint": 5.236,
            },
            stiffness=1.831,
            damping=0.092,
            armature=0.01,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_roll_joint"],
            effort_limit_sim={".*_wrist_roll_joint": 0.14},
            velocity_limit_sim={".*_wrist_roll_joint": 4.189},
            stiffness=1.831,
            damping=0.031,
            armature=0.01,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit_sim={
                "head_yaw_joint": 0.2,
                "head_pitch_joint": 0.4,
            },
            velocity_limit_sim={
                "head_yaw_joint": 8.378,
                "head_pitch_joint": 6.283,
            },
            stiffness=1.831,
            damping=0.092,
            armature=0.01,
        ),
    },
)
