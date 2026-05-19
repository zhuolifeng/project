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

SYSTEM_PROMPT = "你是一个规划专家，你需要根据我的指令，规划机器人的动作。务必保证机器人的动作是合理且优化的，且如果需要路径规划，注意避免路径之间的碰撞和路径与物体的碰撞，保证路径不发生交叉，同时尽量**均匀地**规划路径，计算相邻路径间的距离，并保证均匀。因为每次交互只能对一个回合进行操作，所以每次只思考一个回合的规划。**不要过度思考**。**输出的结果必须严格符合[Output Instruction]中的要求输出。**"

SWEEP_TASK_PROMPT = """Alice（dustpan）和 Bob（broom）需协作清扫桌面上的所有方块。清扫规则：Alice 需将 dustpan 置于方块一侧，Bob 从对侧将方块 sweep 进簸箕。每轮任务需根据 “Scene description” 和 “Environment feedback” 迭代优化计划。请检查“History”中的内容，机器人当前位于上一次 MOVE 到的物体边缘，只有当两者处于同一物品边缘时才能进行 SWEEP 操作。当两者都到同一个物体旁边时，应该先进行 SWEEP 操作，在 SWEEP 操作后再思考下一步的 MOVE，只需要在最后进行一次 DUMP。本任务不需要路径规划！"""



def get_chat_prompt(env: MujocoSimEnv):
    return """请逐步对该任务进行分析推理，找出协调各机器人的最佳策略。为每台机器人精准规划恰好一个动作，并形成具体计划。
借助 [Environment Feedback] 信息来完善你的计划。严格遵循[Output Instruction]中的要求输出。"""


def get_plan_prompt(env: MujocoSimEnv):
    return """请逐步对该任务进行分析推理，找出协调各机器人的最佳策略。为每台机器人精准规划恰好一个动作，并形成具体计划。
借助 [Environment Feedback] 信息来完善你的计划。严格遵循[Output Instruction]中的要求输出。
请给出你的推理过程和最终计划：\n"""
    

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
        max_tokens: int = 2048, 
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
        self.round_history_brief = []
        self.failed_plans = [] # could inherit from previous round if the final plan failed to execute in env.
        self.response_history = [] # [response_t]
        self.old_obs_desp = None
        

    def save_state(self, save_path, fname = 'prompter_state.pkl'):
        state_dict = dict(
            round_history=self.round_history,
            round_history_brief=self.round_history_brief,
            failed_plans=self.failed_plans,
        )
        save_path = os.path.join(save_path, fname)
        with open(save_path, "wb") as f:
            pickle.dump(state_dict, f)

    def load_state(self, load_path, fname = 'prompter_state.pkl'):
        load_path = os.path.join(load_path, fname)
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
        self.round_history = state_dict["round_history"]
        self.round_history_brief = state_dict.get("round_history_brief", [])
        self.failed_plans = state_dict["failed_plans"]

    def compose_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[History]\n"
        pattern = r"\[Executed Action\]([\s\S]*)"
        for i, history in enumerate(self.round_history):
            match = re.search(pattern, history)
            history_f = match.group(1).strip() if match else history
            ret += f"== Round#{i} ==\n{history_f}\n"
        ret += "== Current Round ==\n"
        return ret

    def compose_round_history_brief(self):
        if len(self.round_history_brief) == 0:
            return self.compose_round_history()
        ret = "[History]\n"
        for i, history in enumerate(self.round_history_brief):
            ret += f"== Round#{i} ==\n{history}\n"
        ret += f"== Current Round ==\n"
        return ret

    def describe_robot_state(self, obs, agent_name):
        robot_name = self.env.robot_name_map_inv.get(agent_name, None)
        assert robot_name is not None, f"Agent {agent_name} is not found in the task env!"
        robot_state = getattr(obs, robot_name)
        x, y, z = robot_state.ee_xpos
        if agent_name == "Alice" or robot_name == "ur5e_robotiq":
            obj = "dustpan"
        else:
            obj = "broom"
        return f"{agent_name}'s gripper is at ({x:.1f}, {y:.1f}, {z:.1f}), holding {obj}"
        
    def compose_system_prompt(
        self,
        obs_desp: str,
        plan_feedbacks: List[str] = [], 
        ):
        
        self.old_obs_desp = obs_desp

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
            history_desp = self.compose_round_history_brief() 
            if isinstance(self.env, MakeSandwichTask):
                history_desp = self.compose_round_history()
            full_prompt += history_desp + "\n" 

        if isinstance(self.env, SweepTask):
            object_desp = "[Scene description]\n"
            for name in self.env.cube_names:
                object_desp += self.env.describe_cube_state(self.env.get_obs(), name) + "\n"
            robot_desp = ""
            for robot_name, agent_name in self.env.robot_name_map.items():
                robot_desp += self.describe_robot_state(self.env.get_obs(), agent_name=agent_name) + "\n"
            obs_desp = object_desp + robot_desp
        full_prompt += obs_desp + "\n"

        if len(self.failed_plans) > 0:
            execute_feedback = "以下计划执行失败，请对其进行优化以避免碰撞并平稳抵达目标：\n"
            execute_feedback += "\n".join(self.failed_plans)
            full_prompt += execute_feedback + "\n"

        if len(plan_feedbacks) > 0:
            feedback_prompt = "先前的计划是不可行的，思考原因并避免:\n"
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

        return full_prompt 

    def prompt_one_round(self, obs: EnvState, save_path: str = ""): 
        plan_feedbacks = []
        response_history = []
        obs_desp = self.env.describe_obs(obs)
        for i in range(self.num_replans): 
            system_prompt = self.compose_system_prompt(obs_desp, plan_feedbacks)
            response, usage = self.query_once(
                system_prompt, user_prompt=""
                ) # NOTE: single_thread doesn't use user role
            response_history.append(response)
            
            timestamp = datetime.now().strftime("%m%d-%H%M")
            tosave = [ 
                    {
                        "sender": "SystemPrompt",
                        "message": SYSTEM_PROMPT,
                    },
                    {
                        "sender": "UserPrompt",
                        "message": system_prompt,
                    },
                    {
                        "sender": "Planner",
                        "message": response,
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
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=system_prompt + user_prompt,
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

    

    def _describe_obs_for_summary(self, obs_desp) -> str:
        if isinstance(obs_desp, str):
            return obs_desp
        try:
            return self.env.describe_obs(obs_desp)
        except Exception:
            return ""

    def _extract_summary(self, summary_response: str) -> str:
        match = re.search(r"<summary>(.*?)</summary>", summary_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return summary_response.strip()

    def _summarize_round(self, obs_desp, parsed_plan: str) -> str:
        after_obs = self._describe_obs_for_summary(obs_desp).replace("[Scene description]", "")
        before_obs = (self.old_obs_desp or "").replace("[Scene description]", "")
        responses = "\n".join(self.response_history)
        summarize_prompt = "<Task Information>"
        summarize_prompt += f"The task descriptions are as follows:\n<Task Description>{self.env.describe_task_context()}</Task Description>\n\n"
        summarize_prompt += f"The action descriptions are as follows:\n<Action Description>{self.env.get_action_prompt()}</Action Description>\n\n"
        summarize_prompt += f"The LLM response is as follows:\n<LLM Response>{responses}</LLM Response>\n\n"
        summarize_prompt += f"The parsed action is as follows:\n<parsed_plan>{parsed_plan}</parsed_plan>\n\n"
        if before_obs:
            summarize_prompt += f"The environment information before executing the action is as follows:\n<Environment Observation>{before_obs}</Environment Observation>\n\n"
        if after_obs:
            summarize_prompt += f"The environment information after executing the action is as follows:\n<Environment Observation>{after_obs}</Environment Observation>\n\n"
        summarize_prompt += "</Task Information>\nSummarize what the LLM planned, what action was executed, and what changed in the environment. Your response must contain exactly one concise sentence wrapped in <summary></summary>."
        response, _ = query_ollama_chat(
            model=self.llm_source,
            system_prompt="You summarize robot-task execution history for future planning prompts.",
            user_prompt=summarize_prompt,
            temperature=0,
            max_tokens=min(self.max_tokens, 512),
        )
        return self._extract_summary(response)

    def post_execute_update(self, obs_desp: str, execute_success: bool, parsed_plan: str):
        if execute_success: 
            # clear failed plans, count the previous execute as full past round in history
            self.failed_plans = []
            responses = "\n".join(self.response_history)
            self.round_history.append(
                f"[Response History]\n{responses}\n{obs_desp}\n[Executed Action]\n{parsed_plan}"
            )
            try:
                self.round_history_brief.append(self._summarize_round(obs_desp, parsed_plan))
            except Exception as exc:
                print(f"Summary generation failed, falling back to action history: {exc}")
                self.round_history_brief.append(parsed_plan)
        else:
            self.failed_plans.append(
                parsed_plan
            )
        return

    def post_episode_update(self):
        # clear for next episode
        self.round_history = []
        self.round_history_brief = []
        self.failed_plans = [] 
        self.response_history = []

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
Generate 3 candidate action blocks per control cycle, then select **exactly one valid EXECUTE block** adhering to safety protocols.

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
NAME Alice ACTION PUT rope_front_end groove_right_end PATH <path>
NAME Bob ACTION PUT rope_back_end groove_left_end PATH <path>'
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
