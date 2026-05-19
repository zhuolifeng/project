import os
import copy
import time
import cv2 
import random
import re
import numpy as np  
from pydantic import dataclasses, validator 
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import dm_control 
from dm_control.utils.transformations import mat_to_quat
from pyquaternion import Quaternion
from rocobench.envs.base_env import MujocoSimEnv, EnvState
from rocobench.envs.robot import SimRobot
from rocobench.envs.constants import UR5E_ROBOTIQ_CONSTANTS,PANDA_CONSTANTS

ROPE_FRONT_BODY="CB0"
ROPE_BACK_BODY="CB24"
ROPE_TASK_OBJECTS=[  
    "rope",
    "table_top",
    "obstacle_wall",
    "groove_left_end",
    "groove_right_end",
]
ROPE_ENDS = ["rope_front_end", "rope_back_end"]
GROOVE_ENDS = ["groove_left_end", "groove_right_end"]
ROPE_INIT_RANGE = (
    np.array([-1.2, 0.6, 0.2]),
    np.array([-1.3, 0.45, 0.2]),
)
OBSTACLE_RANGE = (
    np.array([-0.15, 0.5, 0.2]),
    np.array([0, 0.5, 0.3]),
)


OBSTACLE_CORNER_NAMES = [
    "obstacle_wall_front_top",
    # "obstacle_wall_front_bottom",
    "obstacle_wall_back_top",
    # "obstacle_wall_back_bottom",
]
ROPE_TASK_CONTEXT="""2 robots, Alice and Bob, together must pick up a long rope and put it precisely into a narrow groove slot. They must lift up the rope together, each grasping one side of the rope. 
There's an obstacle block wall between the rope and the groove, so the robots must lift the rope **high** above the obstacle block top before putting it into the groove.
The rope is heavy, so they must hold the rope at the same time, and distance between their grippers must stay fixed, so the rope doesn't drop.
At each round, if given 'Scene description' and 'Environment feedback', use it to reason about the task and improve any previous plans. 
Each robot should reach for the closet target, if a robot failed IK to reach a goal, change strategy to a different action.

CRITICAL ROPE STATE MACHINE:
- If both robots are holding nothing, both robots must PICK, and Alice/Bob must PICK two different rope ends.
- If both robots are already holding rope ends, both robots must PUT, and the two rope ends must enter two different groove endpoints.
- Never PUT both rope ends into the same groove endpoint.
- Never keep PICKing after both robots are holding rope; never PUT before both robots are holding rope.
- If the previous feedback reported IK failure for the same robot and coordinate, do not repeat the same end target or key waypoint. Swap assignment, choose a different route, or raise/shift middle waypoints.
- If the previous feedback reported obstacle_wall collision or RRT timeout, the next PUT must raise middle waypoints above obstacle top plus margin while keeping z < 0.55, and route around the wall instead of only changing the final goal.
- Once both rope ends are in the groove, do not move the rope again.
"""

ROPE_ACTION_SPACE="""
[Action Options]
1) PICK <obj> PATH <path>: only PICK if your gripper is empty, <object> can be either rope_front_end or rope_back_end
2) PUT <obj> <location> PATH <path>: PUT on either groove_left_end or groove_right_end only if you have already PICKed a rope end.
Only these actions are valid: PICK and PUT. Do not output MOVE, WAIT, PLACE, DRAG, or any other action.
Choose a different <obj>, robot assignment, or waypoint route if Environment Feedback failed.
Each <path> must contain exactly four coordinates, each coordinate is (x,y,z).
All PATH z values must stay above the table and lower than 0.55.
For PICK, plan a top-down approach to the selected rope end.
For PUT, lift the rope first. Middle waypoints should usually use z about 0.48 to 0.52, route around obstacle_wall, and then descend into the groove endpoint.
When both robots are holding rope, Alice and Bob must PUT their held rope ends into different groove endpoints.
Example PICK:
EXECUTE
NAME Alice ACTION PICK rope_front_end PATH [(0.00, 0.10, 0.48), (-0.30, 0.22, 0.48), (-0.70, 0.32, 0.48), (-1.10, 0.42, 0.48)]
NAME Bob ACTION PICK rope_back_end PATH [(0.05, 0.95, 0.48), (-0.10, 0.78, 0.48), (-0.35, 0.62, 0.48), (-0.55, 0.50, 0.48)]
Example PUT:
EXECUTE
NAME Alice ACTION PUT rope_front_end groove_left_end PATH [(-0.95, 0.32, 0.52), (-0.55, 0.14, 0.52), (-0.15, 0.14, 0.52), (0.20, 0.50, 0.52)]
NAME Bob ACTION PUT rope_back_end groove_right_end PATH [(-0.35, 0.62, 0.52), (-0.15, 0.88, 0.52), (0.45, 0.88, 0.52), (1.00, 0.50, 0.52)]

[Action Output Instruction]
First output 'EXECUTE\n', then give exactly one ACTION per robot, each on a new line.
Example: 'EXECUTE\nNAME Alice ACTION PUT rope_front_end groove_left_end PATH <path>\nNAME Bob ACTION PUT rope_back_end groove_right_end PATH <path>\n'
"""

ROPE_TASK_CHAT_PROMPT="""They discuss to find the best paths. Carefully consider environment feedback and others' responses. Robots must coordinate paths to avoid collision.
They talk in order [Alice],[Bob],[Alice],..., after reaching agreement, plan exactly one ACTION per robot, output an EXECUTE to summarize the plan, and stop talking.
Their chat and final plan are: """

ROPE_TASK_PLAN_PROMPT="""Plan one action for each robot. Analyze the task status, choose the best ACTION for each robot based on its current capability, and plan PATH that efficiently achieves the task and avoids collision:"""

class MoveRopeTask(MujocoSimEnv):
    def __init__( 
        self,
        filepath: str = "rocobench/envs/task_rope.xml",
        one_obj_each: bool = False,
        **kwargs,
    ):    
        self.robot_names = ["ur5e_robotiq", "panda"] 
        self.robot_name_map = {
            "ur5e_robotiq": "Alice",
            "panda": "Bob", 
        }
        self.robot_name_map_inv = {
            "Alice": "ur5e_robotiq",
            "Bob": "panda", 
        }
        self.robots = dict()  

        super(MoveRopeTask, self).__init__(
            filepath=filepath, 
            task_objects=ROPE_TASK_OBJECTS,
            agent_configs=dict(
                ur5e_robotiq=UR5E_ROBOTIQ_CONSTANTS,
                panda=PANDA_CONSTANTS,
            ),
            **kwargs
        ) 
        robotiq_config = UR5E_ROBOTIQ_CONSTANTS.copy()
        # robotiq_config["ik_joint_names"].remove("ur5e_0_base_joint")
        # robotiq_config["all_joint_names"].remove("ur5e_0_base_joint")
        self.robots[
            self.robot_name_map["ur5e_robotiq"]
            ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **robotiq_config,
        )
        panda_config = PANDA_CONSTANTS.copy()
        # panda_config["ik_joint_names"].remove("panda_base_joint")
        # panda_config["all_joint_names"].remove("panda_base_joint")
        self.robots[
            self.robot_name_map["panda"]
        ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **panda_config,
        )
         
        self.align_threshold = 0.2
        self.groove_pos = dict()
        for side in ["left", "right"]:
            self.groove_pos[f"groove_{side}_end"] = self.physics.data.site(f"groove_{side}_end").xpos.copy()
        self.rope_length = np.linalg.norm(
            self.physics.data.body(ROPE_FRONT_BODY).xpos - self.physics.data.body(ROPE_BACK_BODY).xpos
            )
        
    @property
    def waypoint_std_threshold(self):
        return 0.3

    def _rope_end_from_contacts(self, contacts: Set[str]) -> Optional[str]:
        contact_text = ",".join(str(contact) for contact in contacts)
        if ROPE_BACK_BODY in contact_text or "rope_back_end" in contact_text:
            return "rope_back_end"
        if ROPE_FRONT_BODY in contact_text or "rope_front_end" in contact_text:
            return "rope_front_end"
        return None

    def _agent_holding_rope_end(self, obs: EnvState, agent_name: str) -> Optional[str]:
        robot_name = self.robot_name_map_inv[agent_name]
        robot_state = getattr(obs, robot_name)
        return self._rope_end_from_contacts(robot_state.contacts)

    def _rope_end_position(self, rope_end: str) -> np.ndarray:
        body_name = ROPE_FRONT_BODY if rope_end == "rope_front_end" else ROPE_BACK_BODY
        return self.physics.data.body(body_name).xpos.copy()

    def _groove_end_position(self, groove_end: str) -> np.ndarray:
        return self.physics.data.site(groove_end).xpos.copy()

    def _table_height(self) -> float:
        return float(self.physics.data.body("table_top").xpos[2] + 0.15)

    def _obstacle_top_height(self) -> float:
        return float(max(self.physics.data.site(name).xpos[2] for name in OBSTACLE_CORNER_NAMES))

    def _safe_path_z(self, prefer: float = 0.50) -> float:
        min_z = max(self._table_height() + 0.08, self._obstacle_top_height() + 0.04)
        return float(np.clip(max(prefer, min_z), self._table_height() + 0.08, 0.53))

    def _format_path(self, points: List[np.ndarray]) -> str:
        return "[" + ", ".join(
            f"({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
            for point in points
        ) + "]"

    def _format_actions(self, actions: Dict[str, str]) -> str:
        return "\n".join([
            "EXECUTE",
            f"NAME Alice ACTION {actions['Alice']}",
            f"NAME Bob ACTION {actions['Bob']}",
        ])

    def _extract_rope_actions(self, response: str) -> Dict[str, str]:
        actions = {}
        for line in str(response).splitlines():
            match = re.search(
                r"\bNAME\s+(Alice|Bob)\s+ACTION\s+(.+)$",
                line,
                flags=re.IGNORECASE,
            )
            if match is None:
                continue
            agent_name = "Alice" if match.group(1).lower() == "alice" else "Bob"
            action = match.group(2).strip().strip(" \t'\"`")
            if "PATH" in action.upper() and "[" in action and "]" in action:
                action = action[:action.rfind("]") + 1]
            actions[agent_name] = action
        return actions

    def _parse_rope_action(self, action: str) -> Tuple[str, Optional[str], Optional[str]]:
        parts = action.strip().split()
        if len(parts) == 0:
            return "", None, None
        verb = parts[0].upper()
        obj = parts[1] if len(parts) > 1 else None
        location = None
        if verb == "PUT" and len(parts) > 2:
            location = parts[2]
        return verb, obj, location

    def _pick_assignments(self, obs: EnvState) -> Dict[str, str]:
        # Default to the empirically stable division, but give Bob the higher-y
        # end when rope_back_end is pulled too far toward Alice's side.
        front_y = self._rope_end_position("rope_front_end")[1]
        back_y = self._rope_end_position("rope_back_end")[1]
        if back_y < 0.40 and front_y > back_y + 0.05:
            return {"Alice": "rope_back_end", "Bob": "rope_front_end"}
        return {"Alice": "rope_front_end", "Bob": "rope_back_end"}

    def _build_pick_path(self, obs: EnvState, agent_name: str, rope_end: str) -> str:
        robot_name = self.robot_name_map_inv[agent_name]
        start = getattr(obs, robot_name).ee_xpos.copy()
        target = self._rope_end_position(rope_end)
        z = min(0.50, max(self._table_height() + 0.18, target[2] + 0.30))
        start_top = np.array([start[0], start[1], z])
        target_top = np.array([target[0], target[1], z])
        points = [
            start_top * (1 - alpha) + target_top * alpha
            for alpha in [0.25, 0.50, 0.75, 1.00]
        ]
        return self._format_path(points)

    def _build_put_path(self, obs: EnvState, agent_name: str, groove_end: str) -> str:
        robot_name = self.robot_name_map_inv[agent_name]
        start = getattr(obs, robot_name).ee_xpos.copy()
        target = self._groove_end_position(groove_end)
        z = self._safe_path_z(prefer=0.52)

        wall_sites = [self.physics.data.site(name).xpos.copy() for name in OBSTACLE_CORNER_NAMES]
        wall_x = float(np.mean([site[0] for site in wall_sites]))
        wall_front_y = float(min(site[1] for site in wall_sites))
        wall_back_y = float(max(site[1] for site in wall_sites))
        if agent_name == "Alice":
            detour_y = max(0.08, wall_front_y - 0.12)
        else:
            detour_y = min(1.05, wall_back_y + 0.12)
        before_x = wall_x - 0.18
        after_x = max(wall_x + 0.28, 0.45) if agent_name == "Bob" else wall_x + 0.28

        detour_before = np.array([before_x, detour_y, z])
        lift_toward_detour = np.array([
            start[0] * 0.55 + before_x * 0.45,
            start[1] * 0.55 + detour_y * 0.45,
            z,
        ])
        detour_after = np.array([after_x, detour_y, z])
        target_top = np.array([target[0], target[1], z])
        return self._format_path([lift_toward_detour, detour_before, detour_after, target_top])

    def _make_pick_actions(self, obs: EnvState) -> Dict[str, str]:
        assignments = self._pick_assignments(obs)
        return {
            agent_name: (
                f"PICK {rope_end} PATH "
                f"{self._build_pick_path(obs, agent_name, rope_end)}"
            )
            for agent_name, rope_end in assignments.items()
        }

    def _make_put_actions(
        self,
        obs: EnvState,
        requested_actions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        held = {
            agent_name: self._agent_holding_rope_end(obs, agent_name)
            for agent_name in ["Alice", "Bob"]
        }
        if held["Alice"] is None:
            held["Alice"] = "rope_front_end"
        if held["Bob"] is None or held["Bob"] == held["Alice"]:
            held["Bob"] = "rope_back_end" if held["Alice"] == "rope_front_end" else "rope_front_end"

        groove_targets = {"Alice": "groove_left_end", "Bob": "groove_right_end"}
        if requested_actions is not None:
            requested_targets = {
                agent_name: self._parse_rope_action(requested_actions.get(agent_name, ""))[2]
                for agent_name in ["Alice", "Bob"]
            }
            if (
                requested_targets["Alice"] == requested_targets["Bob"]
                and requested_targets["Alice"] in GROOVE_ENDS
            ):
                shared_target = requested_targets["Alice"]
                other_target = "groove_right_end" if shared_target == "groove_left_end" else "groove_left_end"
                if shared_target == "groove_left_end":
                    groove_targets = {"Alice": shared_target, "Bob": other_target}
                else:
                    groove_targets = {"Alice": other_target, "Bob": shared_target}

        return {
            agent_name: (
                f"PUT {held[agent_name]} {groove_targets[agent_name]} PATH "
                f"{self._build_put_path(obs, agent_name, groove_targets[agent_name])}"
            )
            for agent_name in ["Alice", "Bob"]
        }

    def correct_action_response(
        self,
        obs: EnvState,
        response: str,
        previous_actions: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str]:
        """Rope-only state-machine guard for action_and_path responses."""
        actions = self._extract_rope_actions(response)
        held = {
            agent_name: self._agent_holding_rope_end(obs, agent_name)
            for agent_name in ["Alice", "Bob"]
        }
        both_empty = held["Alice"] is None and held["Bob"] is None
        both_holding = held["Alice"] is not None and held["Bob"] is not None

        if both_empty:
            corrected = self._format_actions(self._make_pick_actions(obs))
            if corrected != response:
                return corrected, "Rope state: both grippers are empty; forcing PICK of two different rope ends with four-point PATHs."
            return response, ""

        if both_holding:
            corrected = self._format_actions(self._make_put_actions(obs, requested_actions=actions))
            if corrected != response:
                return corrected, "Rope state: both robots hold rope; forcing PUT of held ends into different groove endpoints with raised detour PATHs."
            return response, ""

        if len(actions) != 2:
            if any(end is not None for end in held.values()):
                corrected = self._format_actions(self._make_put_actions(obs, requested_actions=actions))
                return corrected, "Could not find two valid actions; completing the rope phase with PUT actions only."
            corrected = self._format_actions(self._make_pick_actions(obs))
            return corrected, "Could not find two valid actions; starting the rope phase with PICK actions only."

        cleaned = self._format_actions(actions)
        if cleaned != response:
            return cleaned, "Cleaned rope action output format."
        return response, ""

    def get_target_pos(self, agent_name, target_name) -> Optional[np.ndarray]: 
        ret = None 
        robot_name = self.robot_name_map_inv[agent_name]
        if 'groove' in target_name:
            sname = "groove_left_end" if 'left' in target_name else "groove_right_end"
            ret = self.physics.data.site(sname).xpos.copy() 
        
        elif target_name in ["obstacle_wall_front_top", "obstacle_wall_back_top"]:
            ret = self.physics.data.site(target_name).xpos.copy()
            ret[2] += 0.1
        
        elif target_name == "rope_front_end":
            ret = self.physics.data.body(ROPE_FRONT_BODY).xpos.copy()
            ret[2] += 0.1
        elif target_name == "rope_back_end":
            ret = self.physics.data.body(ROPE_BACK_BODY).xpos.copy()
            ret[2] += 0.1 

        return ret 
         
    def get_target_quat(self, agent_name, target_name) -> Optional[np.ndarray]:
        ret = None
        robot_name = self.robot_name_map_inv[agent_name]
        if 'groove' in target_name or target_name in ["obstacle_wall_front_top", "obstacle_wall_back_top"]:
            sname = target_name 

        elif target_name in ["rope_front_end", "rope_back_end"]:
            body_name = ROPE_FRONT_BODY if target_name == "rope_front_end" else ROPE_BACK_BODY
            return self.physics.data.body(body_name).xquat.copy() 

        else:
            return None

        xmat = self.physics.data.site(target_name).xmat.copy()
        ret = mat_to_quat(xmat.reshape(3,3))
        return ret

    def get_graspable_objects(self):
        graspables = [
            # "rope_front_end",
            # "rope_back_end",
            "groove_left_end",
        ]
        return dict(
            Alice=graspables,
            Bob=graspables, 
        )

    def get_grasp_site(self, obj_name: str = "rope_end") -> Optional[str]:

        if obj_name in ["rope_front_end", "rope_back_end"]:
            return f"{obj_name}"
        
        if obj_name in ["rope_front", "rope_back"]:
            return f"{obj_name}_end"
        
        if 'groove_left' in obj_name:
            return "groove_left_end"
        
        if 'groove_right' in obj_name:
            return "groove_right_end"
        
        else:
            return None
    
    def get_reward_done(self, obs): 
        # task specific!
        rew = 0
        done = False 
        groove_left = self.physics.data.site("groove_left_end").xpos.copy()[:2]
        groove_right = self.physics.data.site("groove_right_end").xpos.copy()[:2]
        rope_front = self.physics.data.body(ROPE_FRONT_BODY).xpos.copy()[:2]
        rope_back = self.physics.data.body(ROPE_BACK_BODY).xpos.copy()[:2]

        dist_lf = np.linalg.norm(groove_left - rope_front)
        dist_lb = np.linalg.norm(groove_left - rope_back)
        dist_rf = np.linalg.norm(groove_right - rope_front)
        dist_rb = np.linalg.norm(groove_right - rope_back)
        
        touch_bottom = 'groove_bottom' in obs.objects['rope'].contacts

        if (dist_lf < self.align_threshold and dist_rb < self.align_threshold) or \
            (dist_lb < self.align_threshold and dist_rf < self.align_threshold):
            done = True and touch_bottom
            rew = 1 if done else 0
        return rew, done

    def get_robot_name(self, agent_name):
        return self.robot_name_map_inv[agent_name]
    
    def get_agent_name(self, robot_name):
        return self.robot_name_map[robot_name]
    
    def get_robot_config(self) -> Dict[str, Dict[str, Any]]:
        return self.agent_configs
    
    def get_sim_robots(self) -> Dict[str, SimRobot]:
        """NOTE this is indexed by agent name, not actual robot names"""
        return self.robots

    def get_robot_reach_range(self, robot_name: str) -> Dict[str, Tuple[float, float]]:
        if robot_name == "ur5e_robotiq" or robot_name == self.robot_name_map["ur5e_robotiq"]:
            return dict(x=(-1.4, 1.6), y=(-0.4, 1.5), z=(0, 1))
        elif robot_name == "panda" or robot_name == self.robot_name_map["panda"]:
            return dict(x=(-1.3, 1.6), y=(0, 1.5), z=(0, 1))
        else:
            raise NotImplementedError
    
    def sample_initial_scene(self): 
        # sample locations of the cabinet
        low, high = ROPE_INIT_RANGE
        new_pos = self.random_state.uniform(low, high) 
        new_angle = self.random_state.uniform(low=-np.pi/6, high=np.pi/6)
        new_quat = Quaternion(
            axis=[0,0,1], angle=new_angle
            ) 
        new_quat = np.array([new_quat.w, new_quat.x, new_quat.y, new_quat.z]) 
        if abs(new_angle) > 0.3:
            new_pos[0] = -1.1
        self.reset_body_pose(
            body_name="rope",
            pos=new_pos,
            # quat=new_quat,
        )  
        self.reset_qpos(
            jnt_name="rope_joint",
            pos=new_pos,
            quat=new_quat,  
        )
        # apply random force to init the rope: 
        rope_body = self.random_state.choice(range(8, 16))
        rope_body = f'CB{rope_body}'
        self.physics.named.data.xfrc_applied[[rope_body], 2] = self.random_state.uniform(1, 1.4, size=1).reshape((1, 1))
        wall_pos = self.random_state.uniform(low=OBSTACLE_RANGE[0], high=OBSTACLE_RANGE[1])
        new_quat = Quaternion(
            axis=[0,0,1], angle=self.random_state.uniform(low=-np.pi/5, high=np.pi/5)
            ) 
        new_quat = np.array([new_quat.w, new_quat.x, new_quat.y, new_quat.z]) 
        
        self.reset_body_pose(
            body_name="obstacle_wall",
            pos=wall_pos,
            quat=new_quat, # no resample
        )
             
        self.physics.forward()
        for i in range(3):
            self.physics.step(50) 

        self.physics.named.data.xfrc_applied[[rope_body], 2] = 0
        self.physics.forward()
        for i in range(6):
            self.physics.step(50) 

        
        self.physics.forward()
        self.physics.step(50)
    
    def get_allowed_collision_pairs(self) -> List[Tuple[int, int]]:
        rope_ids = self.get_all_body_ids("rope")
        table_id = self.physics.model.body("table").id
        groove_ids = self.get_all_body_ids("groove")
        ret = []

        for rope_id in rope_ids:
            ret.append((table_id, rope_id))
            for groove_id in groove_ids:
                ret.append((groove_id, rope_id))

        for link_id in self.robots["Alice"].all_link_body_ids + self.robots["Bob"].all_link_body_ids:
            for rope_id in rope_ids:
                ret.append((link_id, rope_id)) 

        return ret
    
    def get_obs(self):
        obs = super().get_obs()
        for name in self.robot_names:
            assert getattr(obs, name) is not None, f"Robot {name} is not in the observation"
        return obs 
    
    def describe_robot_state(self, obs, robot_name: str = "panda"):
        robot_state = getattr(obs, robot_name)
        x, y, z = robot_state.ee_xpos
        contacts = robot_state.contacts 
        obj = self._rope_end_from_contacts(contacts)
        if obj is None:
            obj = "nothing"
        agent_name = self.robot_name_map[robot_name]
        robot_desp = f"{agent_name}'s gripper: ({x:.2f}, {y:.2f}, {z:.2f}), holding {obj}"
        return robot_desp  
 
    def describe_obs(self, obs: EnvState):
        object_desp =  "[Scene description]\n"
        table_height = self.physics.data.body("table_top").xpos[2] + 0.15
        object_desp += f"robots must move lower than 0.55 but higher than table height {table_height:.2f}\n"
        for side in ["left", "right"]:
            sname = f"groove_{side}_end"
            x,y,z = self.groove_pos[sname]
            object_desp += f"{sname}: ({x:.2f}, {y:.2f}, {z:.2f})\n"
       
        for name, body_name in zip(
            ["rope_front_end", "rope_back_end"],
            [ROPE_FRONT_BODY, ROPE_BACK_BODY],
        ):
            x,y,z = self.physics.data.body(body_name).xpos
            object_desp += f"{name}: ({x:.2f}, {y:.2f}, {z:.2f})\n"
        
        object_desp += "The obstacle wall location: "
        for name in OBSTACLE_CORNER_NAMES:
            x,y,z = self.physics.data.site(name).xpos 
            object_desp += f"{name}: ({x:.2f}, {y:.2f}, {z:.2f})\n"

        robot_desp = "The robots:\n"
        for robot_name, agent_name in self.robot_name_map.items():
            robot_desp += self.describe_robot_state(obs, robot_name) + "\n"
            
        full_desp = object_desp + robot_desp
        return full_desp 
    
    def get_task_feedback(self, llm_plan, pose_dict):
        """Get the feedback on planned target poses for each robot at the same time step"""
        feedbacks = []
        obs = self.get_obs() 
        actions = llm_plan.action_strs
        parsed = {
            agent_name: self._parse_rope_action(action_str)
            for agent_name, action_str in actions.items()
        }
        held = {
            agent_name: self._agent_holding_rope_end(obs, agent_name)
            for agent_name in ["Alice", "Bob"]
        }
        # if len(obs.panda.contacts) > 0 and len(obs.ur5e_robotiq.contacts) > 0:
        #     # if ("rope_front_end" in obs.panda.contacts and "rope_back_end" in obs.ur5e_robotiq.contacts) or \
        #     #     ("rope_front_end" in obs.ur5e_robotiq.contacts and "rope_back_end" in obs.panda.contacts):
        #     pose1 = pose_dict["Alice"]
        #     pose2 = pose_dict["Bob"]
        #     dist = np.linalg.norm(pose1.position - pose2.position)
        #     if dist < self.rope_length * 0.8 or dist > self.rope_length * 1.2:
        #         task_feedback += f"PATH at: Alice {pose1.pos_string}, Bob: {pose2.pos_string} are wrong: distance between them is {dist}, but it must be {self.rope_length:.2f}"   
        for agent_name, action_str in actions.items():
            upper_action = action_str.upper()
            if any(bad_action in upper_action for bad_action in ["PLACE", "WAIT", "MOVE", "DRAG"]):
                feedbacks.append(f"{agent_name}'s ACTION is not supported. Rope action space only allows PICK and PUT.")

            verb, obj, location = parsed[agent_name]
            if verb not in {"PICK", "PUT"}:
                feedbacks.append(f"{agent_name}'s ACTION is not supported. Use only PICK or PUT.")
            if verb == "PICK" and held[agent_name] is not None:
                feedbacks.append(f"{agent_name} is already holding {held[agent_name]}; do not PICK after holding rope.")
            if verb == "PUT" and held[agent_name] is None:
                feedbacks.append(f"{agent_name} is not holding rope yet; both robots must PICK different rope ends before PUT.")

        pick_targets = [
            parsed[agent_name][1]
            for agent_name in ["Alice", "Bob"]
            if parsed.get(agent_name, ("", None, None))[0] == "PICK"
        ]
        if len(pick_targets) == 2:
            if any(target not in ROPE_ENDS for target in pick_targets):
                feedbacks.append("PICK target must be rope_front_end or rope_back_end.")
            if pick_targets[0] == pick_targets[1]:
                feedbacks.append("Alice and Bob must pick two different rope ends.")

        put_targets = [
            parsed[agent_name][2]
            for agent_name in ["Alice", "Bob"]
            if parsed.get(agent_name, ("", None, None))[0] == "PUT"
        ]
        if len(put_targets) == 2:
            if any(target not in GROOVE_ENDS for target in put_targets):
                feedbacks.append("PUT target must be groove_left_end or groove_right_end.")
            if put_targets[0] == put_targets[1]:
                feedbacks.append("Alice and Bob must put the two rope ends into different groove endpoints.")

        verbs = [parsed.get(agent_name, ("", None, None))[0] for agent_name in ["Alice", "Bob"]]
        if held["Alice"] is None and held["Bob"] is None and any(verb == "PUT" for verb in verbs):
            feedbacks.append("Both robots are empty-handed; the next valid step is PICK, not PUT.")
        if held["Alice"] is not None and held["Bob"] is not None and any(verb == "PICK" for verb in verbs):
            feedbacks.append("Both robots are already holding rope; the next valid step is PUT, not PICK.")
        if "PICK" in verbs and "PUT" in verbs:
            feedbacks.append("Alice and Bob must stay in the same rope phase: both PICK first, then both PUT.")

        if feedbacks:
            feedbacks.append("The previous waypoint caused IK failure; do not repeat the same waypoint, swap assignment or choose a different route.")
            feedbacks.append("Raise middle waypoints above obstacle top plus margin, while keeping z < 0.55.")
        return " ".join(feedbacks)

    def summarize_round(
        self,
        obs_before: EnvState,
        obs_after: EnvState,
        parsed_plan: str,
    ) -> str:
        held = {
            agent_name: self._agent_holding_rope_end(obs_after, agent_name)
            for agent_name in ["Alice", "Bob"]
        }
        rope_positions = {
            rope_end: self._rope_end_position(rope_end)
            for rope_end in ROPE_ENDS
        }
        groove_left = self._groove_end_position("groove_left_end")[:2]
        groove_right = self._groove_end_position("groove_right_end")[:2]
        front_xy = rope_positions["rope_front_end"][:2]
        back_xy = rope_positions["rope_back_end"][:2]
        front_in_left = np.linalg.norm(front_xy - groove_left) < self.align_threshold
        front_in_right = np.linalg.norm(front_xy - groove_right) < self.align_threshold
        back_in_left = np.linalg.norm(back_xy - groove_left) < self.align_threshold
        back_in_right = np.linalg.norm(back_xy - groove_right) < self.align_threshold
        both_in_groove = (front_in_left and back_in_right) or (front_in_right and back_in_left)

        lines = ["[Round Summary]"]
        lines.append("[Executed Action]")
        lines.append(parsed_plan.strip())
        lines.append("[Current Rope Holding]")
        for agent_name in ["Alice", "Bob"]:
            held_desc = held[agent_name] if held[agent_name] is not None else "nothing"
            lines.append(f"{agent_name}: holding {held_desc}")
        lines.append("[Current Rope End Positions]")
        for rope_end in ROPE_ENDS:
            x, y, z = rope_positions[rope_end]
            lines.append(f"{rope_end}: ({x:.2f}, {y:.2f}, {z:.2f})")
        lines.append("[Next Required Phase]")
        if both_in_groove:
            lines.append("Both rope ends are already aligned with different groove endpoints; do not move the rope again.")
        elif held["Alice"] is not None and held["Bob"] is not None:
            lines.append("Both robots are holding rope; next action must be PUT into different groove endpoints.")
        elif held["Alice"] is None and held["Bob"] is None:
            lines.append("Both robots are empty-handed; next action must be PICK two different rope ends.")
        else:
            lines.append("Only one robot is holding rope; recover by assigning the other robot to the other rope end and avoid mixed PICK/PUT phases.")
        lines.append("[Failure Recovery Rule]")
        lines.append("If the last attempt failed, do not repeat the same unreachable waypoint or same-groove collision; change assignment or route.")
        return "\n".join(lines)
    
    def get_agent_prompt(self, obs: EnvState, agent_name: str):
        robot_name = self.robot_name_map_inv[agent_name]
        other_robot = "Alice" if agent_name == "Bob" else "Bob"
 
        table_height = self.physics.data.body("table_top").xpos[2] + 0.15
        obj_desp = []
        for side in ["left", "right"]:
            sname = f"groove_{side}_end"
            x,y,z = self.groove_pos[sname]
            obj_desp.append(f"{sname}: ({x:.2f}, {y:.2f}, {z:.2f})")
       
        for name, body_name in zip(
            ["rope_front_end", "rope_back_end"],
            [ROPE_FRONT_BODY, ROPE_BACK_BODY],
        ):
            x,y,z = self.physics.data.body(body_name).xpos
            obj_desp.append(f"{name}: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        for name in OBSTACLE_CORNER_NAMES:
            x,y,z = self.physics.data.site(name).xpos 
            obj_desp.append(f"{name}: ({x:.2f}, {y:.2f}, {z:.2f})")
        obj_desp = '\n'.join(obj_desp)
        robot_desp = self.describe_robot_state(obs, robot_name).replace(f"{agent_name}'s", "Your")            

        agent_prompt = f"""
You are {agent_name}, together with {other_robot} you must pick up a long rope and put it precisely into a narrow groove slot. 
You two must lift up the rope together, each grasping one side of the rope. 
There's an obstacle block wall between rope and groove, so both of you must lift the rope **high** above the obstacle block top before putting it into the groove.
The rope is heavy, so you must always pick and put the rope at the same time.
Distance between your grippers must stay fixed, so the rope doesn't drop.

Your gripper must move lower than 0.55 but higher than table height {table_height:.2f}
At the current round: 
{robot_desp}
Object locations: 
{obj_desp}

Never forget you are {agent_name}!
Think step-by-step about the task and {other_robot}'s response. Check and correct {other_robot} if they made a mistake. 
Discuss with {other_robot} to come up with the best plan and smooth, collision-free paths, propose exactly one action for this round. 
Use [Environment Feedback] to improve your actions and waypoint paths. Reach for the closest target, if any waypoint failed IK, change strategy to a different action.

When you respond, tell {other_robot} about your status. Respond very concisely but informatively, and do not repeat what others have said.
Propose exactly one action for yourself at the **current** round, select from [Action Options].
End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan, must strictly follow [Action Output Instruction] and [Path Plan Instruction].
"""
        return agent_prompt

    def describe_robot_capability(self):
        return ""

    def describe_task_context(self):
        context = ROPE_TASK_CONTEXT
        return context

    def get_contact(self):
        
        contacts = super().get_contact()
        # temp fix!  
        contacts["ur5e_robotiq"] = [c for c in contacts["ur5e_robotiq"] if 'CB' in c]
        contacts["panda"] = [c for c in contacts["panda"] if 'CB' in c]
        return contacts

    def chat_mode_prompt(self, chat_history: List[str] = []):
        return ROPE_TASK_CHAT_PROMPT 

    def central_plan_prompt(self):
        return ROPE_TASK_PLAN_PROMPT 

    def get_action_prompt(self) -> str:
        return ROPE_ACTION_SPACE 
        
if __name__ == "__main__":
    env = MoveRopeTask()
    obs = env.reset()  
    print(env.get_agent_prompt(obs, "Alice"))
    print(env.get_agent_prompt(obs, "Bob"))
    breakpoint()
