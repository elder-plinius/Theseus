import json
import os
import pprint
import openai
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
from forge.sdk.memory.memstore import ChromaMemStore

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
        if chat_struct not in self.chat_history[task_id]:
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

        prompt_engine = PromptEngine("gpt-3.5-turbo")
        system_prompt = prompt_engine.load_prompt("system-format-last")
        self.add_chat(task_id, "system", system_prompt)

        profile_gen = ProfileGenerator(task=task, PromptEngine=prompt_engine)

        ontology_prompt_params = {
            "role_expert": profile_gen.role_find(),
            "task": task.input,
            "abilities": self.abilities.list_abilities_for_prompt()
        }

        task_prompt = prompt_engine.load_prompt(
            "ontology-format", **ontology_prompt_params)

        past_chat = self.memory.query(
            task_id=task_id, query=task_prompt, filters={"role": "assistant"})

        if past_chat["documents"]:
            past_convo_params = {"previous_chat": past_chat["documents"][0][:1]}
            past_convo_prompt = prompt_engine.load_prompt(
                "past-convo", **past_convo_params)
            self.add_chat(task_id, "user", past_convo_prompt)

        self.memory.add(task_id=task_id, document=task_prompt, metadatas={"role": "user"})
        self.add_chat(task_id, "user", task_prompt)

        chat_completion_parms = {
            "messages": self.chat_history[task_id],
            "model": "gpt-3.5-turbo"
        }

        chat_response = await chat_completion_request(**chat_completion_parms)
        answer = json.loads(chat_response["choices"][0]["message"]["content"])

        ability = answer["ability"]
        output = await self.abilities.run_ability(
            task_id, ability["name"], **ability.get("args", {}))
        output = str(output) if output else "Success"

        ability_json = {
            "ability": {
                "name": ability["name"],
                "args": ability.get("args", {})
            },
            "output": output
        }

        self.memory.add(task_id=task_id, document=json.dumps(ability_json), metadatas={"role": "assistant"})
        self.add_chat(task_id, "answer", json.dumps(str(ability_json)))

        step.output = answer["thoughts"]["speak"]
        step.is_last = answer["thoughts"]["last_step"]

        await self.db.update_step(task_id, step.step_id, "completed")
        return step
