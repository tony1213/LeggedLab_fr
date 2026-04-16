# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.g1.g1_config import (
    G1FlatAgentCfg,
    G1FlatEnvCfg,
    G1PlaneAgentCfg,
    G1PlaneEnvCfg,
    G1RoughAgentCfg,
    G1RoughEnvCfg,
)
from legged_lab.envs.fr02.fr02_config import (
    FR02FlatAgentCfg,
    FR02FlatEnvCfg,
    FR02PlaneAgentCfg,
    FR02PlaneEnvCfg,
    FR02RoughAgentCfg,
    FR02RoughEnvCfg,
)
from legged_lab.envs.gr2.gr2_config import (
    GR2FlatAgentCfg,
    GR2FlatEnvCfg,
    GR2RoughAgentCfg,
    GR2RoughEnvCfg,
)
from legged_lab.envs.h1.h1_config import (
    H1FlatAgentCfg,
    H1FlatEnvCfg,
    H1RoughAgentCfg,
    H1RoughEnvCfg,
)
from legged_lab.utils.task_registry import task_registry

task_registry.register("h1_flat", BaseEnv, H1FlatEnvCfg(), H1FlatAgentCfg())
task_registry.register("h1_rough", BaseEnv, H1RoughEnvCfg(), H1RoughAgentCfg())
task_registry.register("g1_flat", BaseEnv, G1FlatEnvCfg(), G1FlatAgentCfg())
task_registry.register("g1_rough", BaseEnv, G1RoughEnvCfg(), G1RoughAgentCfg())
task_registry.register("g1_plane", BaseEnv, G1PlaneEnvCfg(), G1PlaneAgentCfg())
task_registry.register("fr02_flat", BaseEnv, FR02FlatEnvCfg(), FR02FlatAgentCfg())
task_registry.register("fr02_rough", BaseEnv, FR02RoughEnvCfg(), FR02RoughAgentCfg())
task_registry.register("fr02_plane", BaseEnv, FR02PlaneEnvCfg(), FR02PlaneAgentCfg())
task_registry.register("gr2_flat", BaseEnv, GR2FlatEnvCfg(), GR2FlatAgentCfg())
task_registry.register("gr2_rough", BaseEnv, GR2RoughEnvCfg(), GR2RoughAgentCfg())
