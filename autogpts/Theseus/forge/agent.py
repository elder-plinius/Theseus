import json
import os
import forge.sdk.memory.memstore
from datetime import datetime
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
    ProfileGenerator
)
from forge.sdk.memory.memstore import ChromaMemStore  # Make sure to import this correctly

LOG = ForgeLogger(__name__)

class ForgeAgent(Agent):
    def __init__(self, database: AgentDB, workspace: Workspace):
        super().__init__(database, workspace)
        self.chat_history = {}
        self.memory = None
    def add_chat(self, task_id: str, role: str, content: str):
        chat_struct = {"role": role, "content": content}
        if task_id not in self.chat_history:
            self.chat_history[task_id] = []
        self.chat_history[task_id].append(chat_struct)

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        task = await self.db.create_task(
            input=task_request.input,
            additional_input=task_request.additional_input
        )
        self.memory = ChromaMemStore(f"{os.getenv('AGENT_WORKSPACE')}/{task.task_id}")
        return task
    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        task = await self.db.get_task(task_id)
        step = await self.db.create_step(
            task_id=task_id,
            input=step_request,
            additional_input=step_request.additional_input,
            is_last=False
        )

        prompt_engine = PromptEngine("gpt-4")

        system_prompt = prompt_engine.load_prompt("system-format-last")
        self.add_chat(task_id, "system", system_prompt)

        task_prompt = prompt_engine.load_prompt(
            "ontology-format",
            role_expert="Testing",
            task=task.input,
            abilities=self.abilities.list_abilities_for_prompt()
        )

        self.memory.add(
            task_id=task_id,
            document=task_prompt,
            metadatas={"role": "user"}
        )

        self.add_chat(task_id, "user", task_prompt)

        chat_completion_parms = {
            "messages": self.chat_history[task_id],
            "model": "gpt-4"
        }

        chat_response = await chat_completion_request(**chat_completion_parms)
        answer = json.loads(chat_response["choices"][0]["message"]["content"])

        ability = answer["ability"]
        output = await self.abilities.run_ability(
            task_id,
            ability["name"],
            **ability.get("args", {})
        )

        self.memory.add(
            task_id=task_id,
            document=chat_response["choices"][0]["message"]["content"],
            metadatas={"role": "assistant"}
        )

        step.output = answer["thoughts"]["speak"]
        step.is_last = answer["thoughts"]["last_step"]

        return step
