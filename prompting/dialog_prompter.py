import os
import time
import json
import pickle
import openai
import requests
import numpy as np
import re
from datetime import datetime
from os.path import join
from typing import List, Tuple, Dict, Union, Optional, Any
from rocobench.subtask_plan import LLMPathPlan
from rocobench.rrt_multi_arm import MultiArmRRT
from rocobench.envs import MujocoSimEnv, EnvState
from .feedback import FeedbackManager
from .parser import LLMResponseParser
from .ollama_client import query_ollama_chat
from .text_utils import strip_think
from .task_hints import build_task_hint


PATH_PLAN_INSTRUCTION="""
[Path Plan Instruction]
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

SWEEP_TASK_PROMPT = """Alice（dustpan）和 Bob（broom）需协作清扫桌面上的所有方块。清扫规则：Alice 需将 dustpan 置于方块一侧，Bob 从对侧将方块 sweep 进簸箕。每轮任务需根据 “Scene description” 和 “Environment feedback” 迭代优化计划。每个机器人每轮严格只执行一个动作，因此只需要输出一组动作。请检查“History”中的内容，机器人当前位于上一次 MOVE 到的物体边缘，只有当两者处于同一物品边缘时才能进行 SWEEP 操作。只需要在最后进行一次 DUMP。"""

MAX_PARSE_FAILS_PER_ROUND = 3


class DialogPrompter:
    """
    Each round contains multiple prompts, query LLM once per each agent
    """
    def __init__(
        self,
        env: MujocoSimEnv,
        parser: LLMResponseParser,
        feedback_manager: FeedbackManager,
        max_tokens: int = 2048,
        debug_mode: bool = False,
        use_waypoints: bool = False,
        robot_name_map: Dict[str, str] = {"panda": "Bob"},
        num_replans: int = 3,
        max_calls_per_round: int = 10,
        use_history: bool = True,
        use_feedback: bool = True,
        temperature: float = 0,
        llm_source: str = "gpt-4"
    ):
        self.max_tokens = max_tokens
        self.debug_mode = debug_mode
        self.use_waypoints = use_waypoints
        self.use_history = use_history
        self.use_feedback = use_feedback
        self.robot_name_map = robot_name_map
        self.robot_agent_names = list(robot_name_map.values())
        self.num_replans = num_replans
        self.env = env
        self.feedback_manager = feedback_manager
        self.parser = parser
        self.round_history = []
        self.round_history_brief = []
        self.failed_plans = []
        self.latest_chat_history = []
        self.max_calls_per_round = max_calls_per_round
        self.temperature = temperature
        self.llm_source = llm_source
        self.old_obs = None

    def compose_system_prompt(
        self,
        obs: EnvState,
        agent_name: str,
        chat_history: List = [], # chat from previous replan rounds
        current_chat: List = [],  # chat from current round, this comes AFTER env feedback
        feedback_history: List = []
    ) -> str:
        action_desp = self.env.get_action_prompt()
        if self.use_waypoints:
            action_desp += PATH_PLAN_INSTRUCTION
        agent_prompt = self.env.get_agent_prompt(obs, agent_name)
        if self.env.__class__.__name__ == "SweepTask":
            agent_prompt = f"{SWEEP_TASK_PROMPT}\n{agent_prompt}"
        self.old_obs = obs

        round_history = self.get_round_history_brief() if self.use_history else ""

        execute_feedback = ""
        if len(self.failed_plans) > 0:
            execute_feedback = "Plans below failed to execute, improve them to avoid collision and smoothly reach the targets:\n"
            execute_feedback += "\n".join(self.failed_plans) + "\n"
            execute_feedback += "[Hard Rule] Do NOT re-emit any of the failed plans above. If reachability/collision/constraint failed, change the action or assign it to the other robot.\n"

        chat_history = "[Previous Chat]\n" + "\n".join(chat_history) if len(chat_history) > 0 else ""

        system_prompt = f"{action_desp}\n{round_history}\n{execute_feedback}{agent_prompt}\n{chat_history}\n"

        if self.use_feedback and len(feedback_history) > 0:
            system_prompt += "\n".join(feedback_history)

        if len(current_chat) > 0:
            system_prompt += "[Current Chat]\n" + "\n".join(current_chat) + "\n"

        task_hint = build_task_hint(self.env, obs)
        if task_hint:
            system_prompt += "\n" + task_hint + "\n"

        return system_prompt

    def _build_task_hint(self, obs: EnvState) -> str:
        return build_task_hint(self.env, obs)

    def _pack_hint(self, obs: EnvState) -> str:
        from .task_hints import pack_hint
        return pack_hint(self.env, obs)

    def _sandwich_hint(self, obs: EnvState) -> str:
        from .task_hints import sandwich_hint
        return sandwich_hint(self.env, obs)

    def get_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[History]\n"
        for i, history in enumerate(self.round_history):
            ret += f"== Round#{i} ==\n{history}\n"
        ret += f"== Current Round ==\n"
        return ret

    def get_round_history_brief(self):
        if len(self.round_history_brief) == 0:
            return self.get_round_history()
        ret = "[History]\n"
        for i, history in enumerate(self.round_history_brief):
            ret += f"== Round#{i} ==\n{history}\n"
        ret += f"== Current Round ==\n"
        return ret

    def prompt_one_round(self, obs: EnvState, save_path: str = ""):
        plan_feedbacks = []
        chat_history = []
        for i in range(self.num_replans):
            final_agent, final_response, agent_responses = self.prompt_one_dialog_round(
                obs,
                chat_history,
                plan_feedbacks,
                replan_idx=i,
                save_path=save_path,
            )
            chat_history += agent_responses
            parse_succ, parsed_str, llm_plans = self.parser.parse(obs, final_response)

            curr_feedback = "None"
            if not parse_succ:
                curr_feedback = f"""
This previous response from [{final_agent}] failed to parse!: '{final_response}'
{parsed_str} Re-format to strictly follow [Action Output Instruction]!"""
                ready_to_execute = False

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
                    "message": (final_response if not parse_succ else llm_plans[0].get_action_desp()),
                },
            ]
            timestamp = datetime.now().strftime("%m%d-%H%M")
            fname = f'{save_path}/replan{i}_feedback_{timestamp}.json'
            json.dump(tosave, open(fname, 'w'))

            if ready_to_execute:
                break
            else:
                print(curr_feedback)
        self.latest_chat_history = chat_history
        return ready_to_execute, llm_plans, plan_feedbacks, chat_history

    def prompt_one_dialog_round(
        self,
        obs,
        chat_history,
        feedback_history,
        replan_idx=0,
        save_path='data/',
        ):
        """
        keep prompting until an EXECUTE is outputted or max_calls_per_round is reached
        """

        agent_responses = []
        usages = []
        dialog_done = False
        num_responses = {agent_name: 0 for agent_name in self.robot_agent_names}
        n_calls = 0
        parse_fail_streak = 0

        while n_calls < self.max_calls_per_round:
            for agent_name in self.robot_agent_names:
                system_prompt = self.compose_system_prompt(
                    obs,
                    agent_name,
                    chat_history=chat_history,
                    current_chat=agent_responses,
                    feedback_history=feedback_history,
                    )

                agent_prompt = (
                    f"You are {agent_name}. First output a single line:\n"
                    f"  THINK: <which item is next, who can reach it, why this avoids the previous failure>\n"
                    f"Then output the EXECUTE block (strictly follow [Action Output Instruction]).\n"
                    f"Your response is:"
                )
                if n_calls == self.max_calls_per_round - 1:
                    agent_prompt = (
                        f"You are {agent_name}, this is the last call, you must end your response by "
                        f"incorporating all previous discussions and output the best plan via EXECUTE.\n"
                        f"First output a single THINK: line, then the EXECUTE block.\n"
                        f"Your response is:"
                    )
                response, usage = self.query_once(
                    system_prompt,
                    user_prompt=agent_prompt,
                    max_query=3,
                    )

                tosave = [
                    {
                        "sender": "SystemPrompt",
                        "message": system_prompt,
                    },
                    {
                        "sender": "UserPrompt",
                        "message": agent_prompt,
                    },
                    {
                        "sender": agent_name,
                        "message": response,
                    },
                    usage,
                ]
                timestamp = datetime.now().strftime("%m%d-%H%M")
                fname = f'{save_path}/replan{replan_idx}_call{n_calls}_agent{agent_name}_{timestamp}.json'
                json.dump(tosave, open(fname, 'w'))

                num_responses[agent_name] += 1
                pruned_response = strip_think(response).strip()
                agent_responses.append(
                    f"[{agent_name}]:\n{pruned_response}"
                    )
                usages.append(usage)
                n_calls += 1
                if 'EXECUTE' in response:
                    parse_fail_streak = 0
                    if replan_idx > 0 or all([v > 0 for v in num_responses.values()]):
                        dialog_done = True
                        break
                else:
                    parse_fail_streak += 1
                    if parse_fail_streak >= MAX_PARSE_FAILS_PER_ROUND:
                        print(f"[dialog] {parse_fail_streak} consecutive no-EXECUTE responses; breaking to outer replan loop")
                        dialog_done = True
                        break

                if self.debug_mode:
                    dialog_done = True
                    break

            if dialog_done:
                break

        return agent_name, response, agent_responses

    def query_once(self, system_prompt, user_prompt, max_query):
        response = None
        usage = None
        print('======= system prompt ======= \n ', system_prompt)
        print('======= user prompt ======= \n ', user_prompt)

        if self.debug_mode:
            response = "EXECUTE\n"
            for aname in self.robot_agent_names:
                action = input(f"Enter action for {aname}:\n")
                response += f"NAME {aname} ACTION {action}\n"
            return response, dict()


        for n in range(max_query):
            print('querying {}th time'.format(n))
            try:
                response, usage = query_ollama_chat(
                    model=self.llm_source,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=self.temperature,
                    max_tokens=2048,
                )

                print('======= response ======= \n ', response)
                print('======= usage ======= \n ', usage)
                break
            except Exception as exc:
                print(f"API error, try again: {exc}")
                time.sleep(2)
            continue
        if response is None:
            raise RuntimeError(
                f"Failed to query Ollama model {self.llm_source!r} "
                f"after {max_query} attempts"
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
        before_obs = ""
        if self.old_obs is not None:
            before_obs = self._describe_obs_for_summary(self.old_obs).replace("[Scene description]", "")
        chats = "\n".join(self.latest_chat_history)
        summarize_prompt = "<Task Information>"
        summarize_prompt += f"The task descriptions are as follows:\n<Task Description>{self.env.describe_task_context()}</Task Description>\n\n"
        for agent in self.robot_agent_names:
            if self.old_obs is not None:
                summarize_prompt += f"The prompt for LLM agent {agent} is as follows:\n<Agent Prompt>{self.env.get_agent_prompt(self.old_obs, agent)}</Agent Prompt>\n"
        summarize_prompt += f"\nThe action descriptions are as follows:\n<Action Description>{self.env.get_action_prompt()}</Action Description>\n\n"
        summarize_prompt += f"The chats from LLM agents are as follows:\n<LLM chats>{chats}</LLM chats>\n\n"
        summarize_prompt += f"The parsed action is as follows:\n<parsed_plan>{parsed_plan}</parsed_plan>\n\n"
        if before_obs:
            summarize_prompt += f"The environment information before executing the action is as follows:\n<Environment Observation>{before_obs}</Environment Observation>\n\n"
        if after_obs:
            summarize_prompt += f"The environment information after executing the action is as follows:\n<Environment Observation>{after_obs}</Environment Observation>\n\n"
        summarize_prompt += "</Task Information>\nSummarize what each LLM agent did, what action was executed, and what changed in the environment. Your response must contain exactly one concise sentence wrapped in <summary></summary>."
        response, _ = query_ollama_chat(
            model=self.llm_source,
            system_prompt="You summarize multi-agent robot-task execution history for future planning prompts.",
            user_prompt=summarize_prompt,
            temperature=0,
            max_tokens=min(self.max_tokens, 512),
        )
        return self._extract_summary(response)

    def post_execute_update(self, obs_desp: str, execute_success: bool, parsed_plan: str):
        if execute_success:
            # clear failed plans, count the previous execute as full past round in history
            self.failed_plans = []
            chats = "\n".join(self.latest_chat_history)
            self.round_history.append(
                f"[Chat History]\n{chats}\n[Executed Action]\n{parsed_plan}"
            )
            try:
                self.round_history_brief.append(self._summarize_round(obs_desp, parsed_plan))
            except Exception as exc:
                print(f"Summary generation failed, falling back to raw action history: {exc}")
                self.round_history_brief.append(parsed_plan)
        else:
            self.failed_plans.append(parsed_plan)
        return

    def post_episode_update(self):
        # clear for next episode
        self.round_history = []
        self.round_history_brief = []
        self.failed_plans = []
        self.latest_chat_history = []
