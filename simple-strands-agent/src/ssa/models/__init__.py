import logging
from omegaconf import DictConfig, OmegaConf
from strands.models import Model
from strands.models.model import CacheConfig
from ssa.models.litellm import SRLiteLLMModel
from ssa.models.gemini import SRGeminiModel
from ssa.models.bedrock import SRBedrockModel
from ssa.models.openai import SROpenAIModel
from ssa.models.xai import XAIModel
from ssa.models.anthropic import AnthropicModel
from anthropic import AsyncAnthropicBedrockMantle
from botocore.config import Config as BotocoreConfig

LOG = logging.getLogger(__name__)


def sr_model(cfg: DictConfig) -> Model:
    model_id = cfg.agent.model
    invoker_params = OmegaConf.to_container(cfg.agent.invoker_params, resolve=True)
    match cfg.agent.invoker:
        case "bedrock":
            cache_kwargs = {}
            if cfg.agent.get("prompt_caching", False):
                caching_params = cfg.agent.get("prompt_caching_params", {})
                _strategy = caching_params.get("strategy", "auto")
                if _strategy == "auto":
                    cache_kwargs["cache_config"] = CacheConfig(strategy="auto")
                cache_kwargs["cache_tools"] = "default"
                LOG.info(f"Prompt caching enabled (cache_config={_strategy}, cache_tools=default)")
            # TODO: (vatshank) make read_timeout part of the config with a default?
            return SRBedrockModel(
                model_id=model_id,
                region_name=cfg.aws.region,
                boto_client_config=BotocoreConfig(read_timeout=300),
                **cache_kwargs,
                **invoker_params,
            )
        # TODO: (vatshank) also add caching for litellm models.
        case "litellm":
            return SRLiteLLMModel(
                model_id=model_id, params=invoker_params,
            )
        case "gemini":
            return SRGeminiModel(
                model_id=model_id,
                params=invoker_params,
            )
        case "openai":
            use_previous_id = invoker_params.pop("use_previous_id", False)
            use_responses_api = invoker_params.pop("use_responses_api", True)
            refresh_bedrock_token = invoker_params.pop("refresh_bedrock_token", False)
            refresh_gcloud_token = invoker_params.pop("refresh_gcloud_token", False)
            include_reasoning_in_history = invoker_params.pop("include_reasoning_in_history", False)
            cache_client = invoker_params.pop("cache_client", True)
            provide_session_id = invoker_params.pop("provide_session_id", False)
            request_log = invoker_params.pop("request_log", False)
            client_args = invoker_params.pop("client_args", {})
            return SROpenAIModel(
                model_id=model_id.removeprefix("openai/"), # make compatible with litellm
                params=invoker_params,
                client_args=client_args,
                use_previous_id=use_previous_id,
                use_responses_api=use_responses_api,
                refresh_bedrock_token = refresh_bedrock_token,
                refresh_gcloud_token = refresh_gcloud_token,
                include_reasoning_in_history=include_reasoning_in_history,
                cache_client=cache_client,
                provide_session_id=provide_session_id,
                request_log=request_log,
            )
        case "xai":
            conv_id = invoker_params.pop("conv_id", None)
            timeout = invoker_params.pop("timeout", 3600)
            use_previous_id = invoker_params.pop("use_previous_id", True)
            return XAIModel(
                use_previous_id=use_previous_id,
                model_id=model_id,
                params=invoker_params,
                conv_id=conv_id,
                timeout=timeout,
            )
        case "anthropic":
            client_args = invoker_params.pop("client_args", {})
            max_tokens = invoker_params.pop("max_tokens")
            return AnthropicModel(
                model_id=model_id,
                max_tokens=max_tokens,
                params=invoker_params,
                client_args=client_args,
            )
        case "bedrock_mantle":
            # AWS Bedrock mantle endpoint — native Anthropic Messages API shape.
            # client_args is forwarded to AsyncAnthropicBedrockMantle (aws_region,
            # base_url, api_key, aws_profile, etc.). Falls back to cfg.aws.region
            # when aws_region isn't in client_args.
            client_args = invoker_params.pop("client_args", {}) or {}
            if "aws_region" not in client_args:
                aws_region = OmegaConf.select(cfg, "aws.region")
                if aws_region:
                    client_args["aws_region"] = aws_region
            max_tokens = invoker_params.pop("max_tokens")
            model = AnthropicModel(
                model_id=model_id,
                max_tokens=max_tokens,
                params=invoker_params,
                client_args={},
            )
            model.client = AsyncAnthropicBedrockMantle(**client_args)
            return model

    raise ValueError(f"{cfg.agent.invoker} not supported.")
