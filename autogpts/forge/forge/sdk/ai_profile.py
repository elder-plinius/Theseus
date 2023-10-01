"""
PROFILE CONCEPT:

The profile generator is used to intiliase and configure an ai agent. 
It came from the obsivation that if an llm is provided with a profile such as:
```
Expert: 

```
Then it's performance at a task can impove. Here we use the profile to generate
a system prompt for the agent to use. However, it can be used to configure other
aspects of the agent such as memory, planning, and actions available.

The possibilities are limited just by your imagination.
"""

from forge.sdk import PromptEngine


class ProfileGenerator:
    def __init__(self, task: str, PromptEngine: PromptEngine):
        """
        Initialize the profile generator with the task to be performed.
        """
        self.task = task
    def set_expert(self) -> str:
    """
    Consults GPT-3.5 Turbo to determine the most suitable expert role for the given task.
    Returns the name of the expert role.
    """

    # Create a more descriptive and useful prompt
    prompt = f"Based on the following task description, identify the most suitable type of expert who would excel at completing it. The task is: '{self.task.input}'. Please return only the type of expert name, without any additional text."

    # Use GPT-3.5 Turbo for the model
    model = "text-davinci-003"

    # Generate the response
    response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=50, n=1, stop=None, temperature=0.7)

    # Extract and return the role name from the response
    return response.choices[0].text.strip()


