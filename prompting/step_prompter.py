from .plan_prompter import SingleThreadPrompter, get_plan_prompt


class StepPrompter(SingleThreadPrompter):
    """
    Single-step planner mode.

    This keeps the current compact history, parsing, feedback, summary, and
    Ollama handling from SingleThreadPrompter, but uses the direct plan prompt
    even when the selected communication mode is "step".
    """

    def compose_system_prompt(self, obs_desp: str, plan_feedbacks=[]):
        self.old_obs_desp = obs_desp

        task_desp = self.env.describe_task_context()
        action_desp = self.env.get_action_prompt()
        if self.use_waypoints:
            from .plan_prompter import PATH_PLAN_INSTRUCTION

            action_desp += PATH_PLAN_INSTRUCTION

        full_prompt = f"{task_desp}\n{action_desp}\n"

        if self.use_history:
            full_prompt += self.compose_round_history_brief() + "\n"

        full_prompt += obs_desp + "\n"

        if len(self.failed_plans) > 0:
            execute_feedback = "Plans below failed to execute, improve them to avoid collision and smoothly reach the targets:\n"
            execute_feedback += "\n".join(self.failed_plans)
            full_prompt += execute_feedback + "\n"

        if len(plan_feedbacks) > 0:
            feedback_prompt = "Previous Plans Require Improvement:\n"
            feedback_prompt += "\n".join(plan_feedbacks) + "\n"
            full_prompt += feedback_prompt

        full_prompt += get_plan_prompt(self.env)
        return full_prompt
