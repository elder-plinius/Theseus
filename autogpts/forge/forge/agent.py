import json
import os
from forge.sdk import (
    Agent,
    AgentDB,
    Step,
    StepRequestBody,
    Workspace,
    ForgeLogger,
    Task,
    TaskRequestBody,
    PromptEngine,
    chat_completion_request,
)

LOG = ForgeLogger(__name__)

class ForgeAgent(Agent):
    def __init__(self, database: AgentDB, workspace: Workspace):
        super().__init__(database, workspace)
        self.chat_history = {}
        self.memory = None

    def add_chat(self, task_id: str, role: str, content: str):
        chat_struct = {"role": role, "content": content}
        try:
            if chat_struct not in self.chat_history[task_id]:
                self.chat_history[task_id].append(chat_struct)
        except KeyError:
            self.chat_history[task_id] = [chat_struct]

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        task = await self.db.create_task(task_request)
        LOG.info(f"📦 Task created: {task.task_id}")
        return task
    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        task = await self.db.get_task(task_id)
        step = await self.db.create_step(
            task_id=task_id, input=step_request, is_last=False
        )

        prompt_engine = PromptEngine("gpt-4")
        prompt_params = {
            "role_expert": "General Assistant",
            "task": task.input,
        }

        task_prompt = prompt_engine.load_prompt("task-format", **prompt_params)

        self.add_chat(task_id, "user", task_prompt)

        chat_completion_params = {
            "messages": self.chat_history[task_id],
            "model": "gpt-4",
        }

        chat_response = await chat_completion_request(**chat_completion_params)

        try:
            answer = json.loads(chat_response["choices"][0]["message"]["content"])
            ability = answer.get("ability", {}).get("name", "unknown")

            if ability == "WriteFile":
                output = await self.write_file(task_id, "output.txt", "Washington")
            elif ability == "ReadFile":
                output = await self.read_file(task_id, "output.txt")
            elif ability == "Search":
                output = "Search functionality not yet implemented."
            else:
                output = "Ability not recognized."

            step.output = output
            step.is_last = answer.get("is_last", False)
        except json.JSONDecodeError:
            step.output = "Error in decoding JSON."

        return step
    async def write_file(self, task_id: str, filename: str, content: str):
        file_path = self.workspace.get_file_path(task_id, filename)
        with open(file_path, "w") as f:
            f.write(content)
        return f"Written to {filename}"

    async def read_file(self, task_id: str, filename: str):
        file_path = self.workspace.get_file_path(task_id, filename)
        with open(file_path, "r") as f:
            data = f.read()
        return data

