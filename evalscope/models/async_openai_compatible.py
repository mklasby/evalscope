"""Async variant of OpenAICompatibleAPI using openai.AsyncOpenAI."""

import asyncio
from typing import Any, Dict, Generator, List, Optional, Union

from openai import APIStatusError, AsyncOpenAI, BadRequestError, PermissionDeniedError, UnprocessableEntityError
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletion

from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils import get_logger
from evalscope.utils.argument_utils import get_supported_params
from evalscope.utils.function_utils import AsyncioLoopRunner, async_retry_call
from .openai_compatible import OpenAICompatibleAPI
from .utils.openai import (
    chat_choices_from_openai,
    collect_stream_response,
    model_output_from_openai,
    openai_chat_messages,
    openai_chat_tool_choice,
    openai_chat_tools,
    openai_completion_params,
    openai_handle_bad_request,
)

logger = get_logger()


class AsyncOpenAICompatibleAPI(OpenAICompatibleAPI):
    """OpenAI-compatible model API that uses AsyncOpenAI for non-blocking I/O.

    Each call to generate (invoked from worker threads by the
    DefaultEvaluator) submits a coroutine to a shared background event
    loop via AsyncioLoopRunner. This lets httpx.AsyncClient multiplex
    all in-flight HTTP requests over a single connection pool,
    dramatically improving throughput compared to one blocking
    httpx.Client per thread.
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        # Initialise sync client & shared state via parent.
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
            **model_args,
        )

        # Create async client with the same credentials.
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **model_args,
        )

    # --------------------------------------------------------------------- #
    # Core async generation
    # --------------------------------------------------------------------- #

    async def _agenerate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Async mirror of OpenAICompatibleAPI.generate."""
        tools, tool_choice, config = self.resolve_tools(tools, tool_choice, config)

        completion_params = self.completion_params(config=config, tools=len(tools) > 0)

        request = dict(
            messages=openai_chat_messages(input),
            tools=openai_chat_tools(tools) if len(tools) > 0 else NOT_GIVEN,
            tool_choice=openai_chat_tool_choice(tool_choice) if len(tools) > 0 else NOT_GIVEN,
            **completion_params,
        )

        self._validate_async_request_params(request)

        try:
            completion = await async_retry_call(
                self.async_client.chat.completions.create,
                retries=config.retries,
                sleep_interval=config.retry_interval,
                **request,
            )
            if not isinstance(completion, ChatCompletion):
                completion = collect_stream_response(completion)
            response = completion.model_dump()
            self.on_response(response)

            choices = self.chat_choices_from_completion(completion, tools)
            return model_output_from_openai(completion, choices)

        except (BadRequestError, UnprocessableEntityError, PermissionDeniedError) as ex:
            return self.handle_bad_request(ex)

    def _validate_async_request_params(self, params: Dict[str, Any]) -> None:
        """Like parent's validate_request_params but caches against the async client."""
        if not hasattr(self, '_async_valid_params'):
            self._async_valid_params = get_supported_params(self.async_client.chat.completions.create)

        extra_body = params.get('extra_body', {})
        for key in list(params.keys()):
            if key not in self._async_valid_params:
                extra_body[key] = params.pop(key)
        if extra_body:
            params['extra_body'] = extra_body

    # --------------------------------------------------------------------- #
    # Sync wrappers (called by DefaultEvaluator worker threads)
    # --------------------------------------------------------------------- #

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        return AsyncioLoopRunner.run(self._agenerate(input, tools, tool_choice, config))

    def batch_generate(
        self,
        inputs: List[List[ChatMessage]],
        tools: List[List[ToolInfo]],
        tool_choices: List[ToolChoice],
        configs: List[GenerateConfig],
    ) -> Generator[ModelOutput, None, None]:
        async def _gather():
            coros = [
                self._agenerate(inp, t, tc, cfg)
                for inp, t, tc, cfg in zip(inputs, tools, tool_choices, configs)
            ]
            return await asyncio.gather(*coros)

        results = AsyncioLoopRunner.run(_gather())
        yield from results

    def supports_batch(self) -> bool:
        return True
