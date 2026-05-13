import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.config import ROUGH_TERRAINS_CFG
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp         # 导入所有自定义的 rewards, observations,  curriculum , velocity_command 函数

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)

'''
prims 场景树结构示例（基于当前的配置）：
/World
├── ground                                    ← TerrainImporterCfg (line 46)
│   └── (地形网格，9×21 块子地形)
│
├── skyLight                                  ← AssetBaseCfg (line 80)
│
└── envs                                      ← 由 ManagerBasedRLEnvCfg 自动创建
    ├── env_0                                 ← 第 0 个环境（间距 2.5m）
    │   ├── Robot                             ← ROBOT_CFG.replace(prim_path) (line 66)
    │   │   └── (G1 29DOF USD 内容展开)
    │   │       ├── torso_link                ← height_scanner 挂载点 (line 70)
    │   │       ├── left_hip_pitch_link
    │   │       ├── right_hip_pitch_link
    │   │       ├── left_ankle_roll_link      ← contact_forces 检测目标
    │   │       ├── right_ankle_roll_link     ← contact_forces 检测目标
    │   │       └── ... (所有 body/joint)
    │   └── ...
    │
    ├── env_1
    │   └── Robot
    │       └── ... (同上)
    │
    ├── env_2
    │   └── ...
    │
    ├── ...
    │
    └── env_4095                              ← 共 4096 个并行环境 (line 363)
        └── ...
===========================================
场景树（Stage Tree） 和 刚体树（Articulation Tree）是两个不同的概念。
场景树就是上面的prims，从robot子树展开就是以下的样子
/Robot
├── torso_link                    ← 一个 Xform prim
│   ├── torso_geo                 ← 视觉网格（Mesh）
│   ├── left_shoulder_visual      ← 视觉网格
│   └── ...                       ← 纯视觉/装饰用的 prim
├── left_hip_pitch_link           ← 另一个 Xform prim
│   └── left_hip_geo              ← 视觉网格
├── left_hip_pitch_joint          ← Joint prim（非刚体）
├── ...
----------------------------------------------
刚体树（Physics Articulation Tree）

torso_link (root body, free-floating)
├── left_hip_pitch_link (通过 left_hip_pitch_joint 连接)
│   └── left_knee_link (通过 left_knee_joint 连接)
│       └── left_ankle_pitch_link (通过 left_ankle_pitch_joint 连接)
│           └── left_ankle_roll_link
├── right_hip_pitch_link
│   └── right_knee_link
│       └── ...
├── waist_yaw_link
│   ├── waist_roll_link
│   │   └── waist_pitch_link
│   │       ├── left_shoulder_pitch_link ...
│   │       └── right_shoulder_pitch_link ...
└── ...

场景树	Window → Stage（快捷键，或者去Stage面板）
刚体树 tools → Physics → Physics Inspector
Robot (Articulation root prim)
├── torso_link                    ← RigidBodyAPI + PhysicsMassAPI ✅
├── left_hip_pitch_link           ← RigidBodyAPI + PhysicsMassAPI ✅
├── left_hip_roll_link            ← RigidBodyAPI + PhysicsMassAPI ✅
├── left_knee_link                ← RigidBodyAPI + PhysicsMassAPI ✅
├── left_ankle_roll_link          ← RigidBodyAPI + PhysicsMassAPI ✅
├── right_hip_pitch_link          ← RigidBodyAPI + PhysicsMassAPI ✅
├── ...                           ← 以上每个都有独立的质量/惯量
├── left_shoulder_pitch_link
├── ...
└── right_wrist_yaw_link

=========================================
RobotScene 结构定义：
包含几个资产
- terrain: 由 TerrainImporterCfg 定义，生成一个 8m×8m 的地形网格，分为 9×21 块子地形，每块地形的高度由 Perlin 噪声生成，难度从 0（平坦）到 1（崎岖）不等。
- robot: 由 ROBOT_CFG 定义，基于 G1 29DOF USD 创建，放置在每个环境的 "/Robot" 路径下。
- height_scanner: 由 RayCasterCfg 定义，挂载在机器人 torso_link 上方 20m 处，向下发射网格状的射线，用于测量地面高度。
- contact_forces: 由 ContactSensorCfg 定义，监测机器人所有与地面接触的关节，用于计算碰撞力。
- sky_light: 由 AssetBaseCfg 定义，创建一个 DomeLight，使用 HDRI 贴图提供环境光照。
======================================

scene是整个场景， 依据RobotSceneCfg可配置场景。（包括 object、Articulation、Sensor、terrain等资产的类型、数量、属性等
一个场景下可以包含多个资产，（包括地板、灯光、机器人等可见资产，与一些抽象传感器不可见资产；可见资产可以在stage上的树里面找到，但不可见资产不可找到）。

每个资产可以包含多个刚体和关节，每个资产可以挂载多个传感器。
每个环境（env_0, env_1, ...）都会包含一个 robot资产实例，命名为 "/envs/env_i/Robot"，每个 robot 资产实例都会包含它自己的刚体树和传感器实例（如 height_scanner 和 contact_forces），这些都是独立的，互不干扰的。

SceneEntityCfg 用于指定(选中）场景中某个特定资产的配置，包括刚体树和传感器。

一个scene下有一个prim树（场景树），每个资产在这个树上占据一个子树（比如 robot 资产占据 "/envs/env_i/Robot" 子树），
每个子树下又有一个刚体树（物理层面的连接关系）和一个传感器列表（感知层面的配置）。
通过 SceneEntityCfg，我们可以指定某个资产的哪些刚体和传感器参与到特定的事件、奖励、观察等计算中。

需要注意的是 刚体树和资产树可能不一样（很多资产可能只是视觉装饰，没有物理属性），所以我们通过 SceneEntityCfg 来明确指定我们关注的刚体和传感器。


'''
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=ROUGH_TERRAINS_CFG,  # None, ROUGH_TERRAINS_CFG  COBBLESTONE_ROAD_CFG
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,    # G1可以直接上难度
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots    
    #  即基于 ROBOT_CFG（UNITREE_G1_29DOF_CFG）创建一个新实例，仅将 prim_path 字段替换为 "{ENV_REGEX_NS}/Robot"，其他字段保持不变。  replaces是 @configclass 造的一个函数
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore  # 上面Import了 UNITREE_G1_29DOF_CFG as ROBOT_CFG, 这里直接用它，并且替换了 prim_path

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",     # USD里面有定义
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),     # type: ignore
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],    # 检测的目标（只检查地面）#  第46行的名字对应
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, debug_vis=True)
    # 这个资产只存在于代码层面，在Isaacsim界面上并不可见，没有prims tree中的对应物。
    # 监测机器人 !所有! body   缓存最近 3 步的接触数据      追踪腾空/触地时间
    # 这个传感器监测 Robot 的所有 body（/Robot/.* 匹配所有子 prim），记录它们与场景中其他物体（这里主要是地面 /World/ground）的碰撞信息。包括：
    # 接触力（法向力的大小）
    # 接触状态（是否在接触）
    # 腾空/接触时间（track_air_time=True，记录每个 body 连续腾空或触地持续了多久）
    # 这样设计会报告机器人body和所有其他刚体之间的接触。但是如果我指向要地面的可以设置如下：
    # ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", filter_prim_paths_expr=["/World/ground"],) # 只报告与地面的接触
    # 保留最近3步的历史接触信息。多帧历史可用于实现简单的低通滤波，减少接触检测的"抖动"

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,    # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),   # 指向96行的robot 资产，body_names=".*"表示匹配机器人下的所有刚体(body link)
            "static_friction_range": (0.3, 1.0),                       # 通过 PhysX 的底层 API 直接设置到物理引擎的内部缓冲区中，绕过了 USD 层。IsaacSim界面查不到  （但界面可以看到一个绿色的等号， shader.outputs:out）
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),         # 这个环境不需要弹性碰撞，所以恢复系数设置为0
            "num_buckets": 64,   # 将 4096 个环境分成 64 组，每组内的材质相同，减少物理引擎的材质更新开销。
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,       # type:ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),  # # 质量变化范围（kg）
            "operation": "add",                       #  # 在原始质量上「加」随机值 
        },
    )

    # reset   
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),                      # 可以修改试试，注意不要 0-1000这种不明显   
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},    # 世界坐标系（world frame）下的刚体根速度（root velocity）。
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    # CommandTermCfg的子类 所以本质上还是 CommandTermCfg  和其他都类型的Term都可以对应起来
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),   # 命令重采样时间范围, 每隔该时间段重新生成一个新的目标速度
        rel_standing_envs=0.02,               # 静止环境比例, 在每个环境中，随机选择 rel_standing_envs 比例的环境作为静止环境，其目标速度为 (0, 0, 0)
        rel_heading_envs=1.0,                 # 朝向环境比例, 在每个环境中，随机选择 rel_heading_envs 比例的环境作为朝向环境，其目标朝向为 (0, 0, 0)
        heading_command=False,                # 是否使用朝向命令, 如果为 True，则在朝向环境中生成朝向命令，否则生成速度命令
        debug_vis=True,     # 显示箭头（蓝色Actual和绿色Target的）
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))  # c:\Users\17547\miniconda3\envs\IsaacLab51\Lib\site-packages\isaaclab\source\isaaclab\isaaclab\envs\mdp\observations.py 第 63~67 行：
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))  # [num_envs, 3]
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}) # [num_envs, 3]
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))             # [num_envs, num_joints]
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))   # [num_envs, num_joints]
        last_action = ObsTerm(func=mdp.last_action)   # [num_envs, num_joints]  上一步的action
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        # sum = [num_envs, num_obs_dim*self.history_length]
        #  if concatenated_terms==True
        # 5*linvel(3)+5*angvel(3)+5*proj_grav(3)+5*vel_cmd(3)+5*jpos(num_joints)+5*jvel(num_joints)+5*actions(num_joints)+5*height_scan(num_scan_points)

        # if concatenated_terms==False
        # (linvel(3)+angvel(3)+proj_grav(3)+vel_cmd(3)+jpos(num_joints)+jvel(num_joints)+actions(num_joints)+height_scan(num_scan_points))*5

        def __post_init__(self):
            self.history_length = 5         # (num_envs, num_obs_dim*5)
            self.enable_corruption = True   # 是否添加上面定义的noise
            self.concatenate_terms = True   # 是否拼接成一个向量

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        height_scanner = ObsTerm(func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),    # 如果在policy中也有这个的话，要注意 clip用于限幅；对于real的情况，如果不限幅，policy会输出不可预料的值。对于sim中训练时，就是策略无法收敛
        )   # [num_envs, num_scan_points]

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, 
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,   
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)     # type: ignore
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)  # type:ignore    # 进去后 可以看到和上面commandCfg的关联  base_velocity


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)  # random_choice 这里要打开，上面也要打开，才会生效   如果有其他object的话，还要注意replicate_physics要设为false
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt      # 接触力传感器采样时间
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt  # 高度扫描仪传感器采样时间，模仿现实激光雷达的采样逻辑，通常激光雷达连接上位机采用低频更新。

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):    # 上面设置的 那些配置，对play模式也完全适用，所以直接继承，并且不修改任何字段
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2      # type: ignore
        self.scene.terrain.terrain_generator.num_cols = 10      # type: ignore
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
