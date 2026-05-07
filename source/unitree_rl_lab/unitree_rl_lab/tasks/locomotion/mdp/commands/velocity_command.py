from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass

# C:\Users\17547\miniconda3\envs\IsaacLab51\Lib\site-packages\isaaclab\source\isaaclab\isaaclab\envs\mdp\commands\velocity_command.py 可以看到具体定义
@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    # 父类UniformVelocityCommandCfg的Ranges已经定义了lin_vel_x, lin_vel_y, ang_vel_z的范围,
    # 且父类中有一个ranges实例，用于记录当前范。
    # 这里我们添加一个和Range结构一样的新的属性来限制这个范围的最大值
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING   # MISSING表示创建时必须手动提供，否则报错
