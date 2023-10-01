import json
import pprint
import uuid
import os

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
        
        # initialize chat history per uuid
        self.chat_history = {}

        # memory storage
        self.memory = None

        # expert profile
        self.expert_profile = None

        # setup chatcompletion to achieve this task
        # with custom prompt that will generate steps
        self.prompt_engine = PromptEngine("gpt-3.5-turbo")

    def add_chat(self, 
        task_id: str, 
        role: str, 
        content: str,
        is_function: bool = False,
        function_name: str = None):
        
        if is_function:
            chat_struct = {
                "role": role,
                "name": function_name,
                "content": content
            }
        else:
            chat_struct = {
                "role": role, 
                "content": content
            }
        
        try:
            # cut down on messages being repeated in chat
            if chat_struct not in self.chat_history[task_id]:
                self.chat_history[task_id].append(chat_struct)
        except KeyError:
            self.chat_history[task_id] = [chat_struct]
    
    async def create_task(self, task_request: TaskRequestBody) -> Task:
        try:
            task = await self.db.create_task(
                input=task_request.input,
                additional_input=task_request.additional_input
            )

            LOG.info(
                f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
            )
        except Exception as err:
            LOG.error(f"create_task failed: {err}")
        
        # initalize memstore for task
        try:
            # initialize memstore
            self.memory = ChromaMemStore(
                f"{os.getenv('AGENT_WORKSPACE')}/{task.task_id}")
            LOG.info(f"🧠 Created memorystore @ {os.getenv('AGENT_WORKSPACE')}/{task.task_id}")
        except Exception as err:
            LOG.error(f"memstore creation failed: {err}")
        
        # get role for task
        # use AI to get the proper role experts for the task
        profile_gen = ProfileGenerator(
            task=task,
            prompt_engine=self.prompt_engine
        )

        try:
            self.expert_profile = json.loads(profile_gen.role_find())
        except Exception as err:
            LOG.error(f"role JSON failed: {err}")

        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        task = await self.db.get_task(task_id)

        # Create a new step in the database
        # have AI determine last step
        step = await self.db.create_step(
            task_id=task_id,
            input=step_request,
            additional_input=step_request.additional_input,
            is_last=False
        )

        

        # set up reply json with alternative created
        system_prompt = self.prompt_engine.load_prompt("system-format-last")

        # add to messages
        # wont memory store this as static
        self.add_chat(task_id, "system", system_prompt)
        
        ontology_prompt_params = {
            "name": self.expert_profile["name"],
            "expertise": self.expert_profile["expertise"],
            "role": self.expert_profile["role"],
            "task": task.input,
            "abilities": self.abilities.list_abilities_for_prompt()
        }

        task_prompt = self.prompt_engine.load_prompt(
            "ontology-format",
            **ontology_prompt_params
        )

        self.add_chat(task_id, "user", task_prompt)

        try:
            chat_completion_parms = {
                "messages": self.chat_history[task_id],
                "model": "gpt-3.5-turbo"
            }

            chat_response = await chat_completion_request(
                **chat_completion_parms)
            
            answer = json.loads(
                chat_response["choices"][0]["message"]["content"])
            
            LOG.info(pprint.pformat(answer))

            # Extract the ability from the answer
            ability = answer["ability"]

            LOG.info(f"🔨 Running Ability {ability}")

            # Run the ability and get the output
            if "args" in ability:
                output = await self.abilities.run_ability(
                    task_id,
                    ability["name"],
                    **ability["args"]
                )
            else:
                output = await self.abilities.run_ability(
                    task_id,
                    ability["name"]
                )

            output = str(output) if output else "None"

            # add to converstion
            # add arguments to function content, if any
            if "args" in ability:
                ccontent = f"[Arguments {ability['args']}]: {output} "
            else:
                ccontent = output

            self.add_chat(
                task_id=task_id,
                role="function",
                content=ccontent,
                is_function=True,
                function_name=ability["name"]
            )

            # Set the step output and is_last from AI
            step.output = answer["thoughts"]["speak"]
            step.is_last = answer["thoughts"]["last_step"]

        except json.JSONDecodeError as e:
            # Handle JSON decoding errors
            LOG.error(f"JSON error when decoding: {e}")
        except Exception as e:
            # Handle other exceptions
            LOG.error(f"Unable to generate chat response: {e}")

        LOG.info(f"chat log\n{pprint.pformat(self.chat_history[task_id])}")

        # Return the completed step
        return step
