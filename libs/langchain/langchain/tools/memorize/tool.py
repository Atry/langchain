from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.llms.base import TrainableLLM
from langchain.pydantic_v1 import Field
from langchain.tools.base import BaseTool


class Memorize(BaseTool):
    name: str = "Memorize"
    description: str = (
        "Useful whenever you observed novel information "
        "from previous conversation history, "
        "i.e., another tool's action outputs or human comments. "
        "The action input should include observed information in detail, "
        "then the tool will fine-tune yourself to remember it."
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
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        train_result = await self.llm._atrain_unsupervised((information_to_learn,))
        return f"Train complete. Loss: {train_result.loss}"
