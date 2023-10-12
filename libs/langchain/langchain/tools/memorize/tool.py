from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.llms.base import TrainableLLM
from langchain.pydantic_v1 import Field
from langchain.tools.base import BaseTool


class Memorize(BaseTool):
    name: str = "Memorize"
    description: str = (
        "You should use this tool whenever you observed notable information from another tool's action outputs or human. The action input should include the observed information details in a natural language, then the tool will fine-tune yourself to remember the information."

        # "Useful whenever you observe any notable information from another tool's action outputs or human. "
        # "Input should be the information details in a natural language, which will be used to fine-tune yourself to remember the information. "
        # "Output is a JSON object including the loss"
    )
    llm: TrainableLLM = Field()

    def _run(
        self,
        information_to_learn: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        train_result = self.llm._train_unsupervised((information_to_learn,))
        return f"Train complete. Loss: {train_result.loss}"

    async def _arun(
        self,
        information_to_learn: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        train_result = await self.llm._atrain_unsupervised((information_to_learn,))
        return f"Train complete. Loss: {train_result.loss}"
