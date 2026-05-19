import os 
import json
import pickle 
import requests
import numpy as np
import re
from rocobench.envs import MujocoSimEnv, EnvState
from rocobench.envs import SortOneBlockTask, CabinetTask, MoveRopeTask, SweepTask, MakeSandwichTask, PackGroceryTask
import openai
from datetime import datetime
from .feedback import FeedbackManager
from .parser import LLMResponseParser
from .ollama_client import query_ollama_chat
from .task_hints import build_task_hint
from typing import List, Tuple, Dict, Union, Optional, Any

PATH_PLAN_INSTRUCTION="""
[How to plan PATH]
Each <coord> is a tuple (x,y,z) for gripper location, follow these steps to plan:
1) Decide target location (e.g. an object you want to pick), and your current gripper location.
2) Plan a list of <coord> that move smoothly from current gripper to the target location.
3) The <coord>s must be evenly spaced between start and target.
4) Each <coord> must not collide with other robots, and must stay away from table and objects.  
[How to Incoporate [Enviornment Feedback] to improve plan]
    If IK fails, propose more feasible step for the gripper to reach. 
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects. 
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)] 
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
"""

SWEEP_TASK_PROMPT = """Alice（dustpan）和 Bob（broom）需协作清扫桌面上的所有方块。清扫规则：Alice 需将 dustpan 置于方块一侧，Bob 从对侧将方块 sweep 进簸箕。每轮任务需根据 “Scene description” 和 “Environment feedback” 迭代优化计划。请检查“History”中的内容，机器人当前位于上一次 MOVE 到的物体边缘，只有当两者处于同一物品边缘时才能进行 SWEEP 操作。当两者都到同一个物体旁边时，应该先进行 SWEEP 操作，在 SWEEP 操作后再思考下一步的 MOVE，只需要在最后进行一次 DUMP。本任务不需要路径规划！"""



def get_chat_prompt(env: MujocoSimEnv):
    robot_names = env.get_sim_robots().keys()
    talk_order_str = ",".join([f"[{name}]" for name in robot_names])
    chat_prompt = f"""
The robots discuss to find the best strategy. They carefully analyze others' responses and use [Environment Feedback] to improve their plan. 
They talk in order {talk_order_str}... Once they reach agreement, they summarize the plan by **strictly** following [Action Output Instruction] to format the output, then stop talking.
Their entire discussion and final plan are:
    """
    return chat_prompt 


def get_plan_prompt(env: MujocoSimEnv):
    return """
Reason about the task step-by-step, and find the best strategy to coordinate the robots. Propose a plan of **exactly** one action per robot.
Use [Environment Feedback] to improve your plan. Strictly follow [Action Output Instruction] to format and output the plan.
Your reasoning and final plan output are:
    """
    

class SingleThreadPrompter:
    """
    At each round, queries LLM once for each action plan, 
    query again with environment feedback if the action plan cannot be executed
    """
    def __init__(
        self, 
        env: MujocoSimEnv,
        parser: LLMResponseParser, 
        feedback_manager: FeedbackManager,
        comm_mode: str = "plan", # or chat
        use_waypoints: bool = False,
        use_history: bool = True,
        max_api_queries: int = 3,
        num_replans: int = 3,
        debug_mode: bool = False,   
        temperature: float = 0,
        max_tokens: int = 1000, 
        llm_source: str = "gpt-4",
    ):
        self.env = env 
        self.robot_agent_names = env.get_sim_robots().keys()
        self.feedback_manager = feedback_manager
        self.parser = parser
        self.comm_mode = comm_mode
        self.max_api_queries = max_api_queries
        self.num_replans = num_replans
        self.debug_mode = debug_mode 
        self.use_waypoints = use_waypoints
        self.use_history = use_history
        self.temperature = temperature
        self.llm_source = llm_source
        self.max_tokens = max_tokens

        self.round_history = [] # [obs_t, action_t] but only if action_t got executed
        self.failed_plans = [] # could inherit from previous round if the final plan failed to execute in env.
        self.response_history = [] # [response_t]
        self.last_executed_actions = None
        

    def save_state(self, save_path, fname = 'prompter_state.pkl'):
        state_dict = dict(
            round_history=self.round_history,
            failed_plans=self.failed_plans,
            last_executed_actions=self.last_executed_actions,
        )
        save_path = os.path.join(save_path, fname)
        with open(save_path, "wb") as f:
            pickle.dump(state_dict, f)

    def load_state(self, load_path, fname = 'prompter_state.pkl'):
        load_path = os.path.join(load_path, fname)
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
        self.round_history = state_dict["round_history"]
        self.failed_plans = state_dict["failed_plans"]
        self.last_executed_actions = state_dict.get("last_executed_actions", None)

    def compose_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[History]\n"
        for i, history in enumerate(self.round_history):
            ret += f"== Round#{i} ==\n{history}"
        ret += f"== Current Round ==\n"
        return ret
        
    def compose_system_prompt(
        self,
        obs_desp: str,
        plan_feedbacks: List[str] = [], 
        obs: EnvState = None,
        ):
        
        task_desp = self.env.describe_task_context() # should include task rules
        if isinstance(self.env, SweepTask):
            task_desp = SWEEP_TASK_PROMPT

        action_desp = re.sub(
            r"\[Action Output Instruction\].*",
            "",
            self.env.get_action_prompt(),
            flags=re.DOTALL,
        )
        if self.use_waypoints:
            action_desp += PATH_PLAN_INSTRUCTION

        full_prompt = f"{task_desp}\n{action_desp}\n" 
        
        if self.use_history:
            history_desp = self.compose_round_history() 
            full_prompt += history_desp + "\n" 
        
        full_prompt += obs_desp + "\n"

        if obs is not None:
            task_hint = build_task_hint(self.env, obs)
            if task_hint:
                full_prompt += task_hint + "\n"

        if len(self.failed_plans) > 0:
            execute_feedback = "Plans below failed to execute, improve them to avoid collision and smoothly reach the targets:\n"
            execute_feedback += "\n".join(self.failed_plans) 
            execute_feedback += "\n[Hard Rule] Do NOT re-emit any of the failed plans above. If reachability/collision/constraint failed, change the action, waypoint route, or robot assignment.\n"
            full_prompt += execute_feedback + "\n"

        if len(plan_feedbacks) > 0:
            feedback_prompt = "Previous Plans Require Improvement:\n"
            feedback_prompt += "\n".join(plan_feedbacks) + "\n"
            full_prompt += feedback_prompt
        
        if self.comm_mode == "plan":
            comm_prompt = get_plan_prompt(self.env)
        elif self.comm_mode == "chat":
            comm_prompt = get_chat_prompt(self.env) 
        else:
            raise NotImplementedError
        full_prompt += self.get_action_output_instruction(isinstance(self.env, MakeSandwichTask))
        full_prompt += comm_prompt
        full_prompt += (
            "\n[Think-then-Execute]\n"
            "Before the EXECUTE block, briefly reason about which object/action is valid for each robot. "
            "Then output exactly one EXECUTE block as specified above.\n"
        )

        return full_prompt 

    def prompt_one_round(self, obs: EnvState, save_path: str = ""): 
        plan_feedbacks = []
        response_history = []
        obs_desp = self.env.describe_obs(obs)
        for i in range(self.num_replans): 
            system_prompt = self.compose_system_prompt(obs_desp, plan_feedbacks, obs=obs)
            response, usage = self.query_once(
                system_prompt, user_prompt=""
                ) # NOTE: single_thread doesn't use user role
            raw_response = response
            response = self.parser.normalize_response(response)
            guard_feedback = ""
            if hasattr(self.env, "correct_action_response"):
                response, guard_feedback = self.env.correct_action_response(
                    obs,
                    response,
                    previous_actions=self.last_executed_actions,
                )
                if guard_feedback:
                    print(f"[Action guard] {guard_feedback}")
            if guard_feedback:
                response_history.append(
                    f"[Raw LLM Response]\n{raw_response}\n"
                    f"[Action Guard]\n{guard_feedback}\n"
                    f"[Corrected Action]\n{response}"
                )
            else:
                response_history.append(response)
            
            timestamp = datetime.now().strftime("%m%d-%H%M")
            tosave = [ 
                    {
                        "sender": "SystemPrompt",
                        "message": system_prompt,
                    },
                    {
                        "sender": "UserPrompt",
                        "message": "",
                    },
                    {
                        "sender": "Planner",
                        "message": response if not guard_feedback else (
                            f"[Raw LLM Response]\n{raw_response}\n"
                            f"[Action Guard]\n{guard_feedback}\n"
                            f"[Corrected Action]\n{response}"
                        ),
                    },
                    usage,
                ]
            fname = f'{save_path}/replan{i}_{timestamp}.json'
            json.dump(tosave, open(fname, 'w'))  
            
            curr_feedback = "None"
            # try parsing 
            parse_succ, parsed_str, llm_plans = self.parser.parse(obs, response) 
            if not parse_succ: 
                execute_str = 'EXECUTE' + response.split('EXECUTE')[-1]
                curr_feedback = f"""
Parsing failed! {parsed_str}
Previous response: {execute_str}
Re-format to strictly follow [Action Output Instruction]!
                """
                plan_feedbacks.append(curr_feedback)
                ready_to_execute = False  
            # give env. feedback 
            else:
                ready_to_execute = True
                for j, llm_plan in enumerate(llm_plans): 
                    ready_to_execute, env_feedback = self.feedback_manager.give_feedback(llm_plan)        
                    if not ready_to_execute:
                        curr_feedback = env_feedback
                        break
            
            plan_feedbacks.append(curr_feedback)
            tosave = [
                {
                    "sender": "Feedback",
                    "message": curr_feedback,
                },
                {
                    "sender": "Action",
                    "message": (response if not parse_succ else llm_plans[0].get_action_desp()),
                },
            ]
            timestamp = datetime.now().strftime("%m%d-%H%M")
            fname = f'{save_path}/replan{i}_feedback_{timestamp}.json'
            json.dump(tosave, open(fname, 'w')) 

            if ready_to_execute:
                plan_str = parsed_str
                break  
        self.response_history = response_history
        return ready_to_execute, llm_plans, plan_feedbacks, response_history


    def query_once(self, system_prompt, user_prompt=""):
        response = None
        usage = None   
        print('======= system prompt ======= \n ', system_prompt)
        print('======= user prompt ======= \n ', user_prompt)

        if self.debug_mode: # query human user input
            response = "EXECUTE\n"
            for aname in self.robot_agent_names:
                action = input(f"Enter action for {aname}:\n")
                response += f"NAME {aname} ACTION {action}\n"
            return response, dict()


        for n in range(self.max_api_queries):
            print('querying {}th time'.format(n))
            try:
                response, usage = query_ollama_chat(
                    model=self.llm_source,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                print('======= response ======= \n ', response)
                print('======= usage ======= \n ', usage)
                break
            except Exception as exc:
                print(f"API error, try again: {exc}")
            continue
        if response is None:
            raise RuntimeError(
                f"Failed to query Ollama model {self.llm_source!r} "
                f"after {self.max_api_queries} attempts"
            )
        return response, usage

    

    def _parse_executed_actions(self, parsed_plan: str) -> Dict[str, str]:
        actions = {}
        for line in parsed_plan.splitlines():
            if ":" not in line:
                continue
            agent_name, action = line.split(":", 1)
            actions[agent_name.strip()] = action.strip()
        return actions

    def post_execute_update(self, obs_desp: str, execute_success: bool, parsed_plan: str):
        if execute_success: 
            # clear failed plans, count the previous execute as full past round in history
            self.failed_plans = []
            self.last_executed_actions = self._parse_executed_actions(parsed_plan)
            if hasattr(self.env, "summarize_round") and len(obs_desp.strip()) > 0:
                self.round_history.append(obs_desp.strip() + "\n")
            else:
                responses = "\n".join(self.response_history)
                self.round_history.append(
                    f"[Response History]\n{responses}\n{obs_desp}\n[Executed Action]\n{parsed_plan}"
                )
        else:
            self.failed_plans.append(
                parsed_plan
            )
        return

    def post_episode_update(self):
        # clear for next episode
        self.round_history = []
        self.failed_plans = [] 
        self.response_history = []
        self.last_executed_actions = None

    def get_action_output_instruction(self, sw=False):
        if isinstance(self.env, SortOneBlockTask):
            return '''
[Output Instruction]
1. Generate **exactly one** EXECUTE action block per discussion round
2. EXECUTE action block must be different from Previous_Actions
3. Each robot must perform **one and only one** action per round
4. Valid action options:
        Alice can place panel {panel1,panel2,panel3} and can not place {panel4,panel5,panel6,panel7};
        Bob can place {panel3,panel4,panel5} and can not place {panel1,panel2,panel6,panel7};
        Chad can place {panel5,panel6,panel7} and can not place {panel1,panel2,panel3,panel4};
        The blue_square is unidirectionally picked and placed to panel2.
        The pink_polygon is unidirectionally picked and placed to panel4.
        The yellow_trapezoid is unidirectionally picked and placed to panel6.
Current Phase Objective: {Phase_Goal}
Environment Feedback: {Last_Step_Status}
Historical Actions: {Previous_Actions}

EXECUTE
NAME <Robot> ACTION <Action>
NAME <Robot> ACTION <Action>
NAME <Robot> ACTION <Action>

Failure to follow format will cause system errors!
'''

        if isinstance(self.env, MakeSandwichTask) and not sw:
            return '''
[Output Instruction]
## Mission Objective
Generate candidate action blocks per control cycle, then select **exactly one valid EXECUTE block** adhering to safety protocols.

## Standard Output
The final EXECUTE action block is:
EXECUTE
NAME <Robot> ACTION <Action>
NAME <Robot> ACTION <Action>

Failure to follow format will cause system errors!
'''

        if isinstance(self.env, MakeSandwichTask) and sw:
            return '''
[Output Instruction]
必须先输出'EXECUTE', 然后为每robot规划恰好一个ACTION，并确保每个动作单独占一行。
Example: '
EXECUTE
NAME Dave ACTION PICK bread_slice1 
NAME Chad ACTION PICK tomato'
'''

        if isinstance(self.env, SweepTask):
            return """
[Output Instruction]
必须先输出'EXECUTE', 然后为每robot规划恰好一个ACTION，并确保每个动作单独占一行。
Example#1:
'EXECUTE
NAME Alice ACTION MOVE red_cube
NAME Bob ACTION MOVE red_cube'
Example#2:
'EXECUTE
NAME Alice ACTION WAIT
NAME Bob ACTION SWEEP red_cube'
Example#3: 
'EXECUTE
NAME Alice ACTION DUMP
NAME Bob ACTION WAIT'
以上是仅有的几种可能ACTION组合。
"""

        if isinstance(self.env, PackGroceryTask):
            return '''
[Output Instruction]
必须先输出'EXECUTE', 然后为每robot规划恰好一个ACTION，并确保每个动作单独占一行。
例如: 
'EXECUTE
NAME Alice ACTION PICK soda_can PATH [(0.29, 0.06, 0.51), (0.1, 0.26, 0.46), (-0.05, 0.36, 0.41), (-0.25, 0.42, 0.39)]
NAME Bob ACTION PICK bread PATH [(0.35, 1.05, 0.62), (0.15, 0.81, 0.59), (-0.01, 0.68, 0.52), (-0.10, 0.58, 0.49)]'
我们需要的是路径点，因此不包含起始位置和终点位置。请严格保证路径点的数量为4。
'''

        if isinstance(self.env, MoveRopeTask):
            return """
[Output Instruction]
必须先输出'EXECUTE', 然后为每robot规划恰好一个ACTION，并确保每个动作单独占一行。
Example: '
EXECUTE
NAME Alice ACTION PUT rope_front_end groove_left_end PATH <path>
NAME Bob ACTION PUT rope_back_end groove_right_end PATH <path>'
"""

        if isinstance(self.env, CabinetTask):
            return """
[Action Output Instruction]
必须先输出'EXECUTE', 然后为每robot规划恰好一个ACTION，并确保每个动作单独占一行。
Example: '
EXECUTE
NAME Alice ACTION PICK mug PLACE mug_coaster
NAME Bob ACTION WAIT
NAME Chad ACTION OPEN left_door_handle'
"""

        return self.env.get_action_prompt()
