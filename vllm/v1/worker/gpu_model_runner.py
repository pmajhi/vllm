# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import gc
import itertools
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from tqdm import tqdm

import vllm.envs as envs
from vllm.attention import Attention, AttentionType
from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.layers.chunked_local_attention import ChunkedLocalAttention
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (CompilationLevel, CUDAGraphMode, VllmConfig,
                         get_layers_from_vllm_config, update_config)
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (
    get_pp_group, get_tp_group, graph_capture, is_global_first_rank,
    prepare_communication_buffer_for_model)
from vllm.forward_context import (BatchDescriptor, DPMetadata,
                                  set_forward_context)
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaBase
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import TensorizerLoader, get_model_loader
from vllm.model_executor.models.interfaces import (is_mixture_of_experts,
                                                   supports_eagle3,
                                                   supports_transcription)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling, is_pooling_model, is_text_generation_model)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (BatchedTensorInputs, MultiModalKwargsItem,
                                    PlaceholderRange)
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors, PoolerOutput
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        GiB_bytes, LazyLoader, cdiv, check_use_alibi,
                        get_dtype_size, is_pin_memory_available, round_up,
                        supports_dynamo)
from vllm.v1.attention.backends.mamba_selectors import get_mamba_attn_backend
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport, AttentionMetadataBuilder, CommonAttentionMetadata,
    make_kv_sharing_fast_prefill_attention_metadata,
    reorder_batch_to_split_decodes_and_prefills)
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import (AttentionSpec,
                                        ChunkedLocalAttentionSpec,
                                        FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec, MambaSpec,
                                        SlidingWindowSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors,
                             ModelRunnerOutput)
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.experimental.cachegen_int8_byte_page_adapter import (
    CacheGenInt8FixedBytePageAdapter,
    write_cachegen_int8_mapped_tokens,
)
from vllm.v1.worker.experimental.cachegen_int8_layer_page_pool import (
    CacheGenInt8LayerPagePool,
)
from vllm.v1.worker.experimental.cachegen_int8_page_adapter_config import (
    is_cachegen_int8_page_adapter_enabled,
)
from vllm.v1.worker.experimental.cachegen_kv_quantizer_assignment_table import (
    CacheGenKVQuantizerAssignmentTable,
)
from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
)

from vllm.v1.worker.experimental.cachegen_int8_attention_config import (
    get_cachegen_int8_attention_layer,
    get_cachegen_int8_attention_mode,
    get_cachegen_int8_attention_max_pages,
    is_cachegen_int8_attention_enabled,
)
from vllm.v1.worker.experimental.cachegen_int8_calibration_config import (
    get_cachegen_int8_calibration_layer,
)

from vllm.v1.worker.experimental.cachegen_int8_decode_route import (
    get_cachegen_int8_decode_route_decision,
)
from vllm.v1.worker.experimental.cachegen_int8_dynamic_selected_layer_cache import (
    CacheGenInt8DynamicSelectedLayerCache,
)
from vllm.v1.worker.experimental.cachegen_int8_common_page_decode_triton import (
    cachegen_int8_common_page_decode_attention_triton,
)

from vllm.v1.worker.experimental.cachegen_int8_paged_decode_attention import (
    cachegen_int8_paged_decode_attention_triton,
)

from vllm.v1.worker.experimental.cachegen_int8_shadow_all_layers_config import (
    is_cachegen_int8_shadow_all_layers_enabled,
)
from vllm.v1.worker.experimental.cachegen_int8_shadow_diagnostics_config import (
    is_cachegen_int8_shadow_diagnostics_enabled,
)
from vllm.v1.worker.experimental.cachegen_int8_shadow_layer_config import (
    get_cachegen_int8_shadow_layer,
)
from vllm.v1.worker.experimental.cachegen_int8_shadow_sample_config import (
    is_cachegen_int8_shadow_sample_enabled,
)
from vllm.v1.worker.experimental.cachegen_int8_shadow_write_config import (
    is_cachegen_int8_shadow_write_enabled,
)
from vllm.v1.worker.experimental.hetero_kv_page_config import (
    get_hetero_kv_page_bytes,
    is_hetero_kv_page_planning_enabled,
)
from vllm.v1.worker.experimental.hetero_kv_page_pool_config import (
    get_hetero_kv_page_pool_pages,
    is_hetero_kv_page_pool_enabled,
)
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin, KVConnectorOutput)
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

from .utils import (AttentionGroup, MultiModalBudget, bind_kv_cache,
                    gather_mm_placeholders, initialize_kv_cache_for_kv_sharing,
                    sanity_check_mm_encoder_outputs, scatter_mm_placeholders)

if TYPE_CHECKING:
    import xgrammar as xgr
    import xgrammar.kernels.apply_token_bitmask_inplace_torch_compile as xgr_torch_compile  # noqa: E501

    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")
    xgr_torch_compile = LazyLoader(
        "xgr_torch_compile", globals(),
        "xgrammar.kernels.apply_token_bitmask_inplace_torch_compile")

logger = init_logger(__name__)


def _require_baseline_quantizer_ids(
    quantizer_ids: np.ndarray,
) -> None:
    """Reject nonbaseline execution except the explicit CacheGen shadow mode."""
    if np.all(quantizer_ids == 0):
        return

    if is_cachegen_int8_shadow_write_enabled() and np.all(
            quantizer_ids == 2):
        return

    raise NotImplementedError(
        "Non-default quantizer_id requires a quantized KV-cache "
        "attention backend"
    )


class GPUModelRunner(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        from vllm.model_executor.models.utils import set_cpu_offload_max_bytes
        set_cpu_offload_max_bytes(
            int(self.cache_config.cpu_offload_gb * 1024**3))

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.is_pooling_model = model_config.pooler_config is not None
        self.is_encoder_only_model = False
        self.is_multimodal_raw_input_supported = (
            model_config.is_multimodal_raw_input_supported)
        self.max_model_len = model_config.max_model_len
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.hidden_size = model_config.get_hidden_size()
        self.attention_chunk_size = model_config.attention_chunk_size

        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config)

        # Sampler
        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)

        self.eplb_state: Optional[EplbState] = None
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []
        # Separate experimental byte-page storage. This must not be passed to
        # a standard V1 attention backend until a matching backend exists.
        self.hetero_kv_page_pool: torch.Tensor | None = None
        self.cachegen_int8_layer_page_pool: CacheGenInt8LayerPagePool | None = None
        self.cachegen_int8_pages_per_layer: int | None = None
        self.cachegen_int8_page_adapter: (
            CacheGenInt8FixedBytePageAdapter | None
        ) = None
        self.cachegen_int8_page_adapters: dict[
            str, CacheGenInt8FixedBytePageAdapter
        ] = {}
        self.cachegen_int8_shadow_callback_calls: torch.Tensor | None = None
        self.cachegen_int8_shadow_tokens_written: torch.Tensor | None = None
        self.cachegen_int8_shadow_layer_counters: dict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.cachegen_int8_shadow_sample: dict[str, float | int] | None = None

        # Selected-layer fused INT8 decode validation state. Native
        # FlashAttention remains authoritative until the validation route is
        # explicitly promoted to the model-output path.
        self.cachegen_int8_selected_layer_cache: (
            CacheGenInt8DynamicSelectedLayerCache | None
        ) = None
        self.cachegen_int8_resolved_attention_layer: str | None = None
        self.cachegen_kv_quantizer_assignments = (
            CacheGenKVQuantizerAssignmentTable()
        )

        self.cachegen_int8_decode_validation_stats: dict[str, float | int] = {
            "callback_calls": 0,
            "route_eligible_calls": 0,
            "route_capacity_fallbacks": 0,
            "route_unsupported_fallbacks": 0,
            "route_fallback_reasons": {},
            "native_output_calls": 0,
            "int8_output_calls": 0,
            "common_page_int8_output_calls": 0,
            "reset_calls": 0,
            "common_page_fused_calls": 0,
            "common_page_missing_mirror_fallbacks": 0,
            "common_page_max_abs_error_vs_typed": 0.0,
            "common_page_mean_abs_error_vs_typed_sum": 0.0,
            "common_page_mean_abs_error_vs_typed_count": 0,
            "max_abs_error": 0.0,
            "mean_abs_error_sum": 0.0,
            "mean_abs_error_count": 0,
        }
        self.cachegen_int8_calibration_stats: dict[str, object] | None = None
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []
        # self.kv_cache_config: KVCacheConfig

        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}

        self.use_aux_hidden_state_outputs = False
        # Set up speculative decoding.
        # NOTE(Jiayi): currently we put the entire draft model on
        # the last PP rank. This is not ideal if there are many
        # layers in the draft model.
        if self.speculative_config and get_pp_group().is_last_rank:
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device,
                                             self)  # type: ignore
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = True
            elif self.speculative_config.method == "medusa":
                self.drafter = MedusaProposer(
                    vllm_config=self.vllm_config,
                    device=self.device)  # type: ignore
            else:
                raise ValueError("Unknown speculative decoding method: "
                                 f"{self.speculative_config.method}")
            self.rejection_sampler = RejectionSampler()

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config, self.device, self.pin_memory,
                self.is_pooling_model,
                self.vllm_config.model_config.logits_processors),
            is_pooling_model=self.is_pooling_model,
        )

        # TODO(woosuk): Provide an option to tune the max cudagraph batch size.
        # The convention is different.
        # self.cudagraph_batch_sizes sorts in ascending order.
        # The batch sizes in the config are in descending order.
        if self.compilation_config.cudagraph_capture_sizes and \
                self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            self.cudagraph_batch_sizes = list(
                reversed(self.compilation_config.cudagraph_capture_sizes))

        # Cache the device properties.
        self._init_device_properties()

        # Persistent buffers for CUDA graphs.
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        self.query_start_loc = torch.zeros(self.max_num_reqs + 1,
                                           dtype=torch.int32,
                                           device=self.device)
        self.seq_lens = torch.zeros(self.max_num_reqs,
                                    dtype=torch.int32,
                                    device=self.device)
        self.slot_mapping = torch.zeros(self.max_num_tokens,
                                        dtype=torch.int64,
                                        device=self.device)

        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: Optional[IntermediateTensors] = None

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros((3, self.max_num_tokens + 1),
                                               dtype=torch.int64,
                                               device=self.device)
            self.mrope_positions_cpu = torch.zeros(
                (3, self.max_num_tokens + 1),
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory)
            self.mrope_positions_np = self.mrope_positions_cpu.numpy()

        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = check_use_alibi(model_config)

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device)

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(max(self.max_num_reqs + 1,
                                       self.max_model_len,
                                       self.max_num_tokens),
                                   dtype=np.int64)
        # NOTE(woosuk): These tensors are "stateless", i.e., they are literally
        # a faster version of creating a new tensor every time. Thus, we should
        # not make any assumptions about the values in these tensors.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.positions_np = self.positions_cpu.numpy()
        self.query_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=self.pin_memory)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()
        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=self.pin_memory)
        self.seq_lens_np = self.seq_lens_cpu.numpy()

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device)

        self.uniform_decode_query_len = 1 if not self.speculative_config else \
            1 + self.speculative_config.num_speculative_tokens

        # Cudagraph dispatcher for runtime cudagraph dispatching.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        self.mm_budget = (MultiModalBudget(
            self.model_config,
            self.scheduler_config,
            self.mm_registry,
            max_model_len=self.max_model_len,
            max_num_reqs=self.max_num_reqs,
        ) if self.supports_mm_inputs \
            else None)

        self.reorder_batch_threshold: Optional[int] = None

    def _init_model_kwargs(self, num_tokens: int):
        model_kwargs = dict[str, Any]()
        num_reqs = self.input_batch.num_reqs

        num_pooling_reqs = len(self.input_batch.pooling_params)

        if num_pooling_reqs == 0:
            return model_kwargs

        pooling_params = self.input_batch.pooling_metadata.pooling_params

        assert num_pooling_reqs == num_reqs

        token_type_id_requests = dict[int, Any]()
        for i, param in enumerate(pooling_params):
            if param.extra_kwargs is not None and \
            (token_types := param.extra_kwargs.get(
                "compressed_token_type_ids")) is not None:
                token_type_id_requests[i] = token_types

        if len(token_type_id_requests) == 0:
            return model_kwargs

        seq_lens = self.seq_lens[:num_reqs]
        token_type_ids = []

        for i in range(num_reqs):
            pos = token_type_id_requests.get(i, seq_lens[i])
            ids = (torch.arange(seq_lens[i]) >= pos).int()
            token_type_ids.append(ids)

        model_kwargs["token_type_ids"] = torch.concat(token_type_ids).to(
            device=self.device)
        return model_kwargs

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        # Attention free models have zero kv_cache_goups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if self.reorder_batch_threshold is not None:
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold)

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        """Initialize attributes from torch.cuda.get_device_properties
        """
        self.device_properties = torch.cuda.get_device_properties(self.device)
        self.num_sms = self.device_properties.multi_processor_count

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        torch.cuda.synchronize()

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if sampling_params and \
                sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if pooling_params:
                assert (task := pooling_params.task) is not None, (
                    "You did not set `task` in the API")

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_kwargs=new_req_data.mm_kwargs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
                quantizer_id=new_req_data.quantizer_id,
            )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                second_per_grid_ts = []
                audio_feature_lengths = []
                use_audio_in_video = False
                for mm_item in self.requests[req_id].mm_kwargs:
                    mm_input = mm_item.get_data()
                    if mm_input.get("image_grid_thw") is not None:
                        image_grid_thw.append(
                            mm_input["image_grid_thw"].tolist())
                    if mm_input.get("video_grid_thw") is not None:
                        video_grid_thw.append(
                            mm_input["video_grid_thw"].tolist())
                    if mm_input.get("second_per_grid_ts") is not None:
                        second_per_grid_ts.append(
                            mm_input["second_per_grid_ts"])
                    if mm_input.get("audio_feature_lengths") is not None:
                        audio_feature_lengths.append(
                            mm_input["audio_feature_lengths"])
                    if mm_input.get("use_audio_in_video") is True:
                        use_audio_in_video = True

                hf_config = self.model_config.hf_config

                self.requests[req_id].mrope_positions, \
                    self.requests[req_id].mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions_tensor(
                        self.requests[req_id].prompt_token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        audio_feature_lengths=audio_feature_lengths,
                        use_audio_in_video=use_audio_in_video,
                    )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (num_computed_tokens + len(new_token_ids) -
                                  req_state.num_tokens)
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(
                        new_token_ids[-num_new_tokens:])

            # Update the block IDs.
            if not resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_ids in zip(req_state.block_ids,
                                              new_block_ids):
                    block_ids.extend(new_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            self.input_batch.block_table.append_row(new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index,
                    start_token_index:end_token_index] = new_token_ids
                self.input_batch.num_tokens_no_spec[
                    req_index] = end_token_index
                self.input_batch.num_tokens[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
            if spec_token_ids:
                num_spec_tokens = len(spec_token_ids)
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec tokens.
                self.input_batch.num_tokens[req_index] += num_spec_tokens

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            self.input_batch.add_request(req_state)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    def _extract_mm_kwargs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> BatchedTensorInputs:
        if self.is_multimodal_raw_input_supported:  # noqa: SIM102
            if scheduler_output:
                mm_kwargs = list[MultiModalKwargsItem]()
                for req in scheduler_output.scheduled_new_reqs:
                    req_mm_kwargs = req.mm_kwargs
                    if not isinstance(req_mm_kwargs, list):
                        req_mm_kwargs = list(req_mm_kwargs)
                    mm_kwargs.extend(req_mm_kwargs)

                # Input all modalities at once
                mm_kwargs_combined: BatchedTensorInputs = {}
                for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
                        mm_kwargs,
                        device=self.device,
                        pin_memory=self.pin_memory,
                ):
                    mm_kwargs_combined.update(mm_kwargs_group)

                return mm_kwargs_combined

        return {}

    def _dummy_mm_kwargs(self, num_seqs: int) -> BatchedTensorInputs:
        if self.is_multimodal_raw_input_supported:
            mm_budget = self.mm_budget
            assert mm_budget is not None

            dummy_modality, _ = mm_budget.get_modality_with_max_tokens()

            return self._get_mm_dummy_batch(dummy_modality, num_seqs)

        return {}

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: Optional[np.dtype] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[dict[str, Any], torch.Tensor, Optional[SpecDecodeMetadata],
               np.ndarray, Optional[CommonAttentionMetadata], int]:
        """
        :return: tuple[
            attn_metadata: layer-to-attention_metadata mapping,
            logits_indices, spec_decode_metadata
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)
        self.input_batch.commit_quantized_page_metadata(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = max(tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # The current slot-mapping and KV-cache kernels address physical
        # block_size-token pages. Non-default fixed-byte layouts require a
        # dedicated cache representation and attention backend.
        _require_baseline_quantizer_ids(
            self.input_batch.quantizer_id_cpu[:num_reqs]
        )

        self.input_batch.block_table.compute_slot_mapping(
            req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(
            total_num_scheduled_tokens)

        self.input_batch.block_table.compute_and_set_quantized_mapping(
            req_indices,
            positions_np,
            self.input_batch.tokens_per_page_cpu,
        )
        self.input_batch.block_table.commit_quantized_mapping(
            total_num_scheduled_tokens)

        self.input_batch.commit_quantized_page_metadata(num_reqs)

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)

        # Copy the tensors to the GPU.
        self.input_ids[:total_num_scheduled_tokens].copy_(
            self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)
        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)
        else:
            # Common case (1D positions)
            self.positions[:total_num_scheduled_tokens].copy_(
                self.positions_cpu[:total_num_scheduled_tokens],
                non_blocking=True)

        self.query_start_loc[:num_reqs + 1].copy_(
            self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)

        # Fill unused with 0 for full cuda graph mode.
        self.seq_lens[num_reqs:].fill_(0)
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc[num_reqs + 1:].fill_(
            self.query_start_loc_cpu[num_reqs].item())

        query_start_loc = self.query_start_loc[:num_reqs + 1]

        spec_decode_common_attn_metadata = None

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)
            logits_indices = spec_decode_metadata.logits_indices

        logits_indices_padded = None
        if self.cache_config.kv_sharing_fast_prefill:
            assert self.kv_sharing_fast_prefill_logits_indices is not None
            num_logits = logits_indices.shape[0]
            assert num_logits > 0
            self.kv_sharing_fast_prefill_logits_indices[:num_logits].copy_(
                logits_indices)
            # There might have leftover indices in logits_indices[num_logits:]
            # from previous iterations, whose values may be greater than the
            # batch size in the current iteration. To ensure indices are always
            # valid, we fill the padded indices with the last index.
            self.kv_sharing_fast_prefill_logits_indices[num_logits:].fill_(
                logits_indices[-1].item())
            if (self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
                    and num_logits <= self.cudagraph_batch_sizes[-1]):
                # Use piecewise CUDA graphs.
                # Add padding to the batch size.
                num_logits_padded = self.vllm_config.pad_for_cudagraph(
                    num_logits)
            else:
                num_logits_padded = num_logits
            logits_indices_padded = (
                self.kv_sharing_fast_prefill_logits_indices[:num_logits_padded]
            )

        attn_metadata: dict[str, Any] = {}

        # Prepare encoder attention metadata separately
        # (encoder layers are not in KV cache groups)
        if self.is_encoder_only_model:

            per_layer_metadata = \
                self._build_encoder_only_attn_metadata(
                scheduler_output)

            # Add encoder attention metadata for all encoder layers
            attention_layers = get_layers_from_vllm_config(
                self.vllm_config, Attention)
            for layer_name, attn_module in attention_layers.items():
                if attn_module.attn_type == AttentionType.ENCODER_ONLY:
                    common_attn_metadata, encoder_attn_metadata =\
                        per_layer_metadata[layer_name]
                    attn_metadata[layer_name] = encoder_attn_metadata

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):

            blk_table = self.input_batch.block_table[kv_cache_group_id]
            blk_table_tensor = blk_table.get_device_tensor()[:num_reqs]
            slot_mapping = blk_table.slot_mapping[:total_num_scheduled_tokens]

            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode.
            blk_table.slot_mapping[total_num_scheduled_tokens:].fill_(-1)

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=self.query_start_loc[:num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc_cpu[:num_reqs + 1],
                seq_lens=self.seq_lens[:num_reqs],
                seq_lens_cpu=self.seq_lens_cpu[:num_reqs],
                num_computed_tokens_cpu=self.input_batch.
                num_computed_tokens_cpu_tensor[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                block_table_tensor=blk_table_tensor,
                slot_mapping=slot_mapping,
                quantizer_id=self.input_batch.quantizer_id_gpu[:num_reqs],
                tokens_per_page=self.input_batch.tokens_per_page_gpu[:num_reqs],
                quantized_page_ids=blk_table.quantized_page_ids[
                    :total_num_scheduled_tokens],
                quantized_page_offsets=blk_table.quantized_page_offsets[
                    :total_num_scheduled_tokens],
                causal=True,
            )

            if self.speculative_config and \
                spec_decode_common_attn_metadata is None:
                spec_decode_common_attn_metadata = common_attn_metadata

            for attn_group in self.attn_groups[kv_cache_group_id]:
                # Prepare for cascade attention if enabled & beneficial.
                common_prefix_len = 0
                builder = attn_group.metadata_builder
                if self.cascade_attn_enabled:
                    common_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        scheduler_output.
                        num_common_prefix_blocks[kv_cache_group_id],
                        kv_cache_group_spec.kv_cache_spec,
                        builder,
                    )

                attn_metadata_i = (builder.build(
                    common_prefix_len=common_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                ))

                active_request_ids = list(
                    scheduler_output.num_scheduled_tokens.keys()
                )
                attn_metadata_i.cachegen_kv_quantizer = (
                    self._get_cachegen_kv_quantizer_for_metadata(
                        request_ids=active_request_ids,
                    )
                )

                all_layers_enabled = (
                    is_cachegen_int8_shadow_all_layers_enabled()
                )
                selected_layer = (
                    None
                    if all_layers_enabled
                    else get_cachegen_int8_shadow_layer()
                )
                layer_metadata_overrides: dict[str, Any] = {}
                for layer_name in attn_group.layer_names:
                    needs_calibration = (
                        get_cachegen_int8_calibration_layer() == layer_name
                    )
                    needs_decode_validation = (
                        is_cachegen_int8_attention_enabled()
                        and attn_metadata_i.cachegen_kv_quantizer
                        == CacheGenKVQuantizer.INT8_ADAPTIVE.value
                        and self.cachegen_int8_resolved_attention_layer
                        == layer_name
                    )
                    needs_shadow_write = (
                        is_cachegen_int8_shadow_write_enabled()
                        and (
                            all_layers_enabled
                            or layer_name == selected_layer
                        )
                    )
                    if (
                        not needs_calibration
                        and not needs_decode_validation
                        and not needs_shadow_write
                    ):
                        continue

                    layer_metadata = dataclasses.replace(attn_metadata_i)
                    if needs_calibration:
                        self._attach_cachegen_int8_calibration(
                            layer_name=layer_name,
                            attn_metadata=layer_metadata,
                        )
                    if needs_decode_validation:
                        self._attach_cachegen_int8_decode_validation(
                            layer_name=layer_name,
                            attn_metadata=layer_metadata,
                            kv_cache_group_id=kv_cache_group_id,
                        )
                    if needs_shadow_write:
                        self._attach_cachegen_int8_shadow_write(
                            layer_name=layer_name,
                            attn_metadata=layer_metadata,
                        )
                    layer_metadata_overrides[layer_name] = layer_metadata

                fast_prefill_metadata = attn_metadata_i
                if (self.cache_config.kv_sharing_fast_prefill
                        and self.kv_sharing_fast_prefill_eligible_layers):
                    # Dynamically create a a dataclass type that inherits
                    # from attention metadata type but includes additional
                    # fields logits_indices_padded and num_logits_indices
                    # which are required for prefill truncation
                    fast_prefill_metadata_type = (
                        make_kv_sharing_fast_prefill_attention_metadata(
                            metadata_cls=type(attn_metadata_i), ))
                    fast_prefill_metadata = fast_prefill_metadata_type(
                        **dataclasses.asdict(attn_metadata_i),
                        logits_indices_padded=logits_indices_padded,
                        num_logits_indices=logits_indices.size(0),
                    )

                self._last_cachegen_int8_attn_metadata = {
                    layer_name: {
                        "num_actual_tokens": int(attn_metadata_i.num_actual_tokens),
                        "max_query_len": int(attn_metadata_i.max_query_len),
                        "max_seq_len": int(attn_metadata_i.max_seq_len),
                        "seq_lens": attn_metadata_i.seq_lens.tolist(),
                        "query_start_loc": (
                            attn_metadata_i.query_start_loc.tolist()
                        ),
                        "block_table_shape": list(
                            attn_metadata_i.block_table.shape
                        ),
                        "active_block_table": (
                            attn_metadata_i.block_table[
                                :1,
                                :(
                                    int(attn_metadata_i.max_seq_len)
                                    + self.input_batch.block_table[
                                        kv_cache_group_id
                                    ].block_size
                                    - 1
                                )
                                // self.input_batch.block_table[
                                    kv_cache_group_id
                                ].block_size
                            ].tolist()
                        ),
                        "quantized_page_ids": (
                            attn_metadata_i.quantized_page_ids.tolist()
                            if attn_metadata_i.quantized_page_ids is not None
                            else None
                        ),
                        "quantized_page_offsets": (
                            attn_metadata_i.quantized_page_offsets.tolist()
                            if attn_metadata_i.quantized_page_offsets is not None
                            else None
                        ),
                    }
                    for layer_name in attn_group.layer_names
                }

                for layer_name in attn_group.layer_names:
                    if layer_name in layer_metadata_overrides:
                        attn_metadata[layer_name] = (
                            layer_metadata_overrides[layer_name]
                        )
                    elif (self.cache_config.kv_sharing_fast_prefill
                            and layer_name
                            in self.kv_sharing_fast_prefill_eligible_layers):
                        attn_metadata[layer_name] = fast_prefill_metadata
                    else:
                        attn_metadata[layer_name] = attn_metadata_i

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        return (attn_metadata, logits_indices, spec_decode_metadata,
                num_scheduled_tokens, spec_decode_common_attn_metadata,
                max_num_scheduled_tokens)

    def _compute_cascade_attn_prefix_len(
        self,
        num_scheduled_tokens: np.ndarray,
        num_common_prefix_blocks: int,
        kv_cache_spec: KVCacheSpec,
        attn_metadata_builder: AttentionMetadataBuilder,
    ) -> int:
        """Compute the length of the common prefix for cascade attention.

        NOTE(woosuk): The common prefix length returned by this function
        represents the length used specifically for cascade attention, not the
        actual number of tokens shared between requests. When cascade attention
        is disabled (use_cascade=False), this function returns 0 even if
        requests share common tokens. Additionally, the common prefix length is
        truncated to a multiple of the block size and may be further truncated
        due to implementation details explained below.

        Args:
            num_scheduled_tokens: Number of tokens scheduled per request.
            num_common_prefix_blocks: Number of shared KV cache blocks.

        Returns:
            int: Length of common prefix in tokens.
        """
        common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size
        if common_prefix_len == 0:
            # Common case.
            return 0

        # NOTE(woosuk): Cascade attention uses two attention kernels: one
        # for the common prefix and the other for the rest. For the first
        # kernel, we concatenate all the query tokens (possibly from
        # different requests) and treat them as if they are from the same
        # request. Then, we use bi-directional attention to process the
        # common prefix in the KV cache. Importantly, this means that the
        # first kernel does not do any masking.

        # Consider the following example:
        # Request 1's input query: [D, E, X]
        # Request 1's kv cache: [A, B, C, D, E, X]
        # Request 1's num_computed_tokens: 3 (i.e., [A, B, C])
        # Request 2's input query: [E, Y]
        # Request 2's kv cache: [A, B, C, D, E, Y]
        # Request 2's num_computed_tokens: 4 (i.e., [A, B, C, D])

        # If we use [A, B, C, D, E] as the common prefix, then the
        # first kernel will compute the bi-directional attention between
        # input query [D, E, X, E, Y] and common prefix [A, B, C, D, E].
        # However, this is wrong because D in Request 1 should not attend to
        # E in the common prefix (i.e., we need masking).
        # To avoid this, [A, B, C, D] should be the common prefix.
        # That is, the common prefix should be capped by the minimum
        # num_computed_tokens among the requests, and plus one to include
        # the first token of the query.

        # In practice, we use [A, B, C] as the common prefix, instead of
        # [A, B, C, D] (i.e., the common prefix is capped by the minimum
        # num_computed_tokens, without plus one).
        # This is because of an implementation detail: We want to always
        # use two kernels for cascade attention. Let's imagine:
        # Request 3's input query: [D]
        # Request 3's kv cache: [A, B, C, D]
        # Request 3's num_computed_tokens: 3 (i.e., [A, B, C])
        # If we use [A, B, C, D] as the common prefix for Request 1-3,
        # then Request 3 will be processed only by the first kernel,
        # and the second kernel will get an empty input. While this is not
        # a fundamental problem, our current implementation does not support
        # this case.
        num_reqs = len(num_scheduled_tokens)
        common_prefix_len = min(
            common_prefix_len,
            self.input_batch.num_computed_tokens_cpu[:num_reqs].min())
        # common_prefix_len should be a multiple of the block size.
        common_prefix_len = (common_prefix_len // kv_cache_spec.block_size *
                             kv_cache_spec.block_size)
        use_sliding_window = (isinstance(kv_cache_spec, SlidingWindowSpec) or
                              (isinstance(kv_cache_spec, FullAttentionSpec)
                               and kv_cache_spec.sliding_window is not None))
        use_local_attention = (
            isinstance(kv_cache_spec, ChunkedLocalAttentionSpec)
            or (isinstance(kv_cache_spec, FullAttentionSpec)
                and kv_cache_spec.attention_chunk_size is not None))
        assert isinstance(kv_cache_spec, AttentionSpec)
        use_cascade = attn_metadata_builder.use_cascade_attention(
            common_prefix_len=common_prefix_len,
            query_lens=num_scheduled_tokens,
            num_query_heads=self.num_query_heads,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            use_alibi=self.use_alibi,
            use_sliding_window=use_sliding_window,
            use_local_attention=use_local_attention,
            num_sms=self.num_sms,
        )
        return common_prefix_len if use_cascade else 0

    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = \
                self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = \
                scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = len(req.prompt_token_ids)

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0,
                                      num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(
                    0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions_cpu[:, dst_start:dst_end] = \
                    req.mrope_positions[:,src_start:src_end]

                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.mrope_positions_np,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                mrope_pos_ptr += completion_part_len

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int32)
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32)
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True)
        logits_indices = torch.from_numpy(logits_indices).to(self.device,
                                                             non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
        return metadata

    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_kwargs = list[MultiModalKwargsItem]()
        req_ids_pos = list[tuple[str, int, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_kwargs.append(req_state.mm_kwargs[mm_input_id])
                req_ids_pos.append(
                    (req_id, mm_input_id, req_state.mm_positions[mm_input_id]))

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        encoder_outputs = []
        for _, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
                mm_kwargs,
                device=self.device,
                pin_memory=self.pin_memory,
        ):
            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            curr_group_outputs = self.model.get_multimodal_embeddings(
                **mm_kwargs_group)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )

            for output in curr_group_outputs:
                encoder_outputs.append(output)

        # Cache the encoder outputs.
        for (req_id, input_id, pos_info), output in zip(
                req_ids_pos,
                encoder_outputs,
        ):
            if req_id not in self.encoder_cache:
                self.encoder_cache[req_id] = {}

            self.encoder_cache[req_id][input_id] = scatter_mm_placeholders(
                output,
                is_embed=pos_info.is_embed,
            )

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
        shift_computed_tokens: int = 0,
    ) -> list[torch.Tensor]:
        mm_embeds: list[torch.Tensor] = []
        for req_id in self.input_batch.req_ids:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = \
                req_state.num_computed_tokens + shift_computed_tokens
            mm_positions = req_state.mm_positions
            for i, pos_info in enumerate(mm_positions):
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens)
                assert start_idx < end_idx
                assert req_id in self.encoder_cache
                assert i in self.encoder_cache[req_id]
                encoder_output = self.encoder_cache[req_id][i]

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]

                mm_embeds_item = gather_mm_placeholders(
                    encoder_output[start_idx:end_idx],
                    is_embed=is_embed,
                )
                mm_embeds.append(mm_embeds_item)
        return mm_embeds

    def get_model(self) -> nn.Module:
        # get raw model out of the cudagraph wrapper.
        if isinstance(self.model, CUDAGraphWrapper):
            return self.model.unwrap()
        return self.model

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        supported_tasks = list(model.pooler.get_supported_tasks())

        if (self.scheduler_config.chunked_prefill_enabled
                and "encode" in supported_tasks):
            supported_tasks.remove("encode")

            logger.info_once("Chunked prefill is not supported with "
                             "encode task which using ALL pooling. "
                             "Please turn off chunked prefill by "
                             "`--no-enable-chunked-prefill` before using it.")

        return supported_tasks

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def apply_grammar_bitmask(
        self,
        scheduler_output: "SchedulerOutput",
        logits: torch.Tensor,
    ):
        grammar_bitmask = scheduler_output.grammar_bitmask
        if grammar_bitmask is None:
            return

        # We receive the structured output bitmask from the scheduler,
        # compacted to contain bitmasks only for structured output requests.
        # The order of the requests in the bitmask is not guaranteed to be the
        # same as the order of the requests in the gpu runner's batch. We need
        # to sort the bitmask to match the order of the requests used here.

        # Get the batch indices of the structured output requests.
        # Keep track of the number of speculative tokens scheduled for every
        # request in the batch, as the logit indices are offset by this amount.
        struct_out_req_batch_indices: dict[str, int] = {}
        cumulative_offset = 0
        seq = sorted(self.input_batch.req_id_to_index.items(),
                     key=lambda x: x[1])
        for req_id, batch_index in seq:
            logit_index = batch_index + cumulative_offset
            cumulative_offset += len(
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            if req_id in scheduler_output.structured_output_request_ids:
                struct_out_req_batch_indices[req_id] = logit_index

        out_indices = []

        # Reorder the bitmask to match the order of the requests in the batch.
        sorted_bitmask = np.full(shape=(logits.shape[0],
                                        grammar_bitmask.shape[1]),
                                 fill_value=-1,
                                 dtype=grammar_bitmask.dtype)
        cumulative_index = 0
        seq = sorted(scheduler_output.structured_output_request_ids.items(),
                     key=lambda x: x[1])
        for req_id, _ in seq:
            logit_index = struct_out_req_batch_indices[req_id]
            num_spec_tokens = len(
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            for i in range(1 + num_spec_tokens):
                sorted_bitmask[logit_index + i] = \
                    grammar_bitmask[cumulative_index + i]
                out_indices.append(logit_index + i)
            cumulative_index += 1 + num_spec_tokens
        grammar_bitmask = sorted_bitmask

        # If the length of out indices and the logits have the same shape
        # we don't need to pass indices to the kernel,
        # since the bitmask is already aligned with the logits.
        skip_out_indices = len(out_indices) == logits.shape[0]

        # Serialization of np.ndarray is much more efficient than a tensor,
        # so we receive it in that format.
        grammar_bitmask = torch.from_numpy(grammar_bitmask).contiguous()

        # Force use of the torch.compile implementation from xgrammar to work
        # around issues with the Triton kernel in concurrent structured output
        # scenarios. See PR #19565 and issues #19493, #18376 for details.
        xgr_torch_compile.apply_token_bitmask_inplace_torch_compile(
            logits,
            grammar_bitmask.to(self.device, non_blocking=True),
            indices=out_indices if not skip_out_indices else None,
        )

    def sync_and_slice_intermediate_tensors(
            self, num_tokens: int, intermediate_tensors: IntermediateTensors,
            sync_self: bool) -> IntermediateTensors:

        assert self.intermediate_tensors is not None

        tp = self.vllm_config.parallel_config.tensor_parallel_size
        enabled_sp = self.compilation_config.pass_config. \
            enable_sequence_parallelism
        if enabled_sp:
            # When sequence parallelism is enabled, we always pad num_tokens
            # to be a multiple of tensor_parallel_size (tp) earlier
            assert num_tokens % tp == 0
        is_residual_scattered = tp > 1 and enabled_sp \
            and num_tokens % tp == 0

        # When sequence parallelism is enabled, the "residual" tensor is sharded
        # across tensor parallel ranks, so each rank only needs its own slice.
        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                is_scattered = k == "residual" and is_residual_scattered
                copy_len = num_tokens // tp if is_scattered else \
                    num_tokens
                self.intermediate_tensors[k][:copy_len].copy_(
                    v[:copy_len], non_blocking=True)

        return IntermediateTensors({
            k:
            v[:num_tokens // tp]
            if k == "residual" and is_residual_scattered else v[:num_tokens]
            for k, v in self.intermediate_tensors.items()
        })

    def eplb_step(self,
                  is_dummy: bool = False,
                  is_profile: bool = False) -> None:
        """
        Step for the EPLB (Expert Parallelism Load Balancing) state.
        """
        if not self.parallel_config.enable_eplb:
            return

        assert self.eplb_state is not None
        model = self.get_model()
        assert is_mixture_of_experts(model)
        self.eplb_state.step(
            model,
            is_dummy,
            is_profile,
            log_stats=self.parallel_config.eplb_log_balancedness,
        )

    def get_dp_padding(self,
                       num_tokens: int) -> tuple[int, Optional[torch.Tensor]]:
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank

        # For DP: Don't pad when setting enforce_eager.
        # This lets us set enforce_eager on the prefiller in a P/D setup and
        # still use CUDA graphs (enabled by this padding) on the decoder.
        #
        # TODO(tms) : There are many cases where padding is enabled for
        # prefills, causing unnecessary and excessive padding of activations.

        if dp_size == 1 or self.vllm_config.model_config.enforce_eager:
            # Early exit.
            return 0, None

        num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
            num_tokens, dp_size, dp_rank)
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp).item()
        num_tokens_after_padding = torch.tensor([max_tokens_across_dp_cpu] *
                                                dp_size,
                                                device="cpu",
                                                dtype=torch.int32)
        return max_tokens_across_dp_cpu - num_tokens, num_tokens_after_padding

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        kv_connector_output: Optional[KVConnectorOutput],
    ) -> ModelRunnerOutput:
        assert self.input_batch.num_reqs ==\
            len(self.input_batch.pooling_params), \
        "Either all or none of the requests in" \
        " a batch must be pooling request"

        extracted_hidden_states = list(
            torch.split(hidden_states[:num_scheduled_tokens],
                        num_scheduled_tokens_np.tolist()))

        pooling_metadata = self.input_batch.pooling_metadata

        raw_pooler_output = self.model.pooler(
            hidden_states=extracted_hidden_states,
            pooling_metadata=pooling_metadata)

        pooler_output: list[Optional[torch.Tensor]] = []
        seq_lens = self.seq_lens[:self.input_batch.num_reqs]
        for raw_output, seq_len, prompt_len in zip(
                raw_pooler_output, seq_lens, pooling_metadata.prompt_lens):

            if seq_len == prompt_len:
                pooler_output.append(raw_output.data.cpu())
            else:
                pooler_output.append(None)

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
            kv_connector_output=kv_connector_output,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output,
                                                self.vllm_config)

        # Prepare the decoder inputs.
        (attn_metadata, logits_indices, spec_decode_metadata,
         num_scheduled_tokens_np, spec_decode_common_attn_metadata,
         max_query_len) = (self._prepare_inputs(scheduler_output))

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if (self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            # Eager mode.
            # Pad tokens to multiple of tensor_parallel_size when
            # enabled collective fusion for SP
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if self.compilation_config.pass_config. \
                enable_sequence_parallelism and tp_size > 1:
                num_input_tokens = round_up(num_scheduled_tokens, tp_size)
            else:
                num_input_tokens = num_scheduled_tokens

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if self.supports_mm_inputs:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        if self.supports_mm_inputs and get_pp_group().is_first_rank:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds_scheduled = self.model.get_input_embeddings(
                input_ids=self.input_ids[:num_scheduled_tokens],
                multimodal_embeddings=mm_embeds or None,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:num_scheduled_tokens].copy_(
                inputs_embeds_scheduled)

            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            model_kwargs = {
                **self._init_model_kwargs(num_scheduled_tokens),
                **self._extract_mm_kwargs(scheduler_output),
            }
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs(num_input_tokens)
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)

        uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
            num_scheduled_tokens == self.input_batch.num_reqs * max_query_len)
        batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                           uniform_decode=uniform_decode)
        cudagraph_runtime_mode, batch_descriptor = \
            self.cudagraph_dispatcher.dispatch(batch_descriptor)

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
        ), self.maybe_get_kv_connector_output(
                scheduler_output) as kv_connector_output:
            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
            aux_hidden_states = None

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        broadcast_pp_output = \
            self.parallel_config.distributed_executor_backend \
            == "external_launcher" and len(get_pp_group().ranks) > 0
        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            assert isinstance(hidden_states, IntermediateTensors)
            if not broadcast_pp_output:
                hidden_states.kv_connector_output = kv_connector_output
                return hidden_states
            get_pp_group().send_tensor_dict(hidden_states.tensors,
                                            all_gather_group=get_tp_group())
            logits = None
        else:
            if self.input_batch.pooling_params:
                return self._pool(hidden_states, num_scheduled_tokens,
                                  num_scheduled_tokens_np, kv_connector_output)

            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)
        if broadcast_pp_output:
            model_output_broadcast_data = {
                "logits": logits.contiguous(),
            } if logits is not None else {}
            model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1)
            assert model_output_broadcast_data is not None
            logits = model_output_broadcast_data["logits"]

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            # When indexing with a tensor (bonus_logits_indices), PyTorch
            # creates a new tensor with separate storage from the original
            # logits tensor. This means any in-place operations on bonus_logits
            # won't affect the original logits tensor.
            assert logits is not None
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids

            # Just like `bonus_logits`, `target_logits` is a new tensor with
            # separate storage from the original `logits` tensor. Therefore,
            # it is safe to update `target_logits` in place.
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            output_token_ids = self.rejection_sampler(
                spec_decode_metadata,
                None,  # draft_probs
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
            sampler_output.sampled_token_ids = output_token_ids

        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token for partial prefills.
                # Rewind the generator state as if the token was not sampled.
                # This relies on cuda-specific torch-internal impl details
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)

        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output,
        )

        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            # Includes spec decode tokens.
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
            )
        # Mask out the sampled tokens that should not be sampled.
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        if not self.speculative_config:
            # Speculative decoding is not enabled.
            spec_token_ids = None
        else:
            assert spec_decode_common_attn_metadata is not None
            spec_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                valid_sampled_token_ids,
                sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
            )

        self.eplb_step()

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=kv_connector_output,
            num_nans_in_logits=num_nans_in_logits,
        )

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: Optional[torch.Tensor],
        spec_decode_metadata: Optional[SpecDecodeMetadata],
        common_attn_metadata: CommonAttentionMetadata,
    ) -> list[list[int]]:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if self.speculative_config.method == "ngram":
            assert isinstance(self.drafter, NgramProposer)
            spec_token_ids = self.propose_ngram_draft_token_ids(
                sampled_token_ids)
        elif self.speculative_config.method == "medusa":
            assert isinstance(self.drafter, MedusaProposer)
            if sample_hidden_states.shape[0] == len(sampled_token_ids):
                # The input to the target model does not include draft tokens.
                hidden_states = sample_hidden_states
            else:
                indices = []
                offset = 0
                for num_draft, tokens in zip(
                        spec_decode_metadata.num_draft_tokens,
                        sampled_token_ids):
                    indices.append(offset + len(tokens) - 1)
                    offset += num_draft + 1
                indices = torch.tensor(indices, device=self.device)
                hidden_states = sample_hidden_states[indices]

            spec_token_ids = self.drafter.propose(
                target_hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
            )
        elif self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # TODO(woosuk): Refactor the loop.
            next_token_ids: list[int] = []
            for i, token_ids in enumerate(sampled_token_ids):
                if token_ids:
                    # Common case.
                    next_token_id = token_ids[-1]
                else:
                    # Partial prefill (rare case).
                    # Get the next token id from the request state.
                    req_id = self.input_batch.req_ids[i]
                    req_state = self.requests[req_id]
                    seq_len = (req_state.num_computed_tokens +
                               scheduler_output.num_scheduled_tokens[req_id])
                    next_token_id = req_state.get_token_id(seq_len)
                next_token_ids.append(next_token_id)
            next_token_ids = torch.tensor(next_token_ids,
                                          dtype=torch.int32,
                                          device=self.device)

            if spec_decode_metadata is None:
                # input_ids can be None for multimodal models.
                target_token_ids = self.input_ids[:num_scheduled_tokens]
                # TODO(woosuk): Support M-RoPE.
                target_positions = self.positions[:num_scheduled_tokens]
                if self.use_aux_hidden_state_outputs:
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states],
                        dim=-1)
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
            else:
                # TODO(woosuk): Refactor this.
                num_draft_tokens = spec_decode_metadata.num_draft_tokens
                num_rejected_tokens = [
                    n + 1 - len(sampled_token_ids[i]) if n > 0 else 0
                    for i, n in enumerate(num_draft_tokens)
                ]
                num_rejected_tokens_cpu = torch.tensor(num_rejected_tokens,
                                                       dtype=torch.int32)
                common_attn_metadata, token_indices =\
                    self.drafter.prepare_inputs(
                    common_attn_metadata, num_rejected_tokens_cpu)

                target_token_ids = self.input_ids[token_indices]
                # TODO(woosuk): Support M-RoPE.
                target_positions = self.positions[token_indices]
                if self.use_aux_hidden_state_outputs:
                    target_hidden_states = torch.cat(
                        [h[token_indices] for h in aux_hidden_states], dim=-1)
                else:
                    target_hidden_states = hidden_states[token_indices]
            mm_embeds = None
            if self.supports_mm_inputs:
                mm_embeds = self._gather_mm_embeddings(scheduler_output,
                                                       shift_computed_tokens=1)

            draft_token_ids = self.drafter.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
                mm_embeds=mm_embeds,
            )
            spec_token_ids = draft_token_ids.tolist()
        return spec_token_ids

    def propose_ngram_draft_token_ids(
        self,
        sampled_token_ids: list[list[int]],
    ) -> list[list[int]]:
        # TODO(woosuk): Optimize.
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                draft_token_ids.append([])
                continue

            # Skip requests that require sampling parameters that are not
            # supported with speculative decoding.
            req_id = self.input_batch.req_ids[i]
            if req_id in self.input_batch.spec_decode_unsupported_reqs:
                draft_token_ids.append([])
                continue

            num_tokens = self.input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                draft_token_ids.append([])
                continue

            drafter_output = self.drafter.propose(
                self.input_batch.token_ids_cpu[i, :num_tokens])
            if drafter_output is None or len(drafter_output) == 0:
                draft_token_ids.append([])
            else:
                draft_token_ids.append(drafter_output.tolist())
        return draft_token_ids

    def update_config(self, overrides: dict[str, Any]) -> None:
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, \
                f"Config `{config_name}` not supported. " \
                f"Allowed configs: {allowed_config_names}"
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    def load_model(self, eep_scale_up: bool = False) -> None:
        """
        Args:
            eep_scale_up: the model loading is for elastic EP scale up.
        """
        logger.info("Starting to load model %s...", self.model_config.model)
        if eep_scale_up:
            from vllm.distributed.parallel_state import get_ep_group
            num_local_physical_experts = torch.empty(1,
                                                     dtype=torch.int32,
                                                     device="cpu")
            torch.distributed.broadcast(num_local_physical_experts,
                                        group=get_ep_group().cpu_group,
                                        group_src=0)
            num_local_physical_experts = int(num_local_physical_experts.item())
            new_ep_size = get_ep_group().world_size
            global_expert_load, old_global_expert_indices = (
                EplbState.recv_state())
            num_logical_experts = global_expert_load.shape[1]
            self.parallel_config.num_redundant_experts = (
                num_local_physical_experts * new_ep_size - num_logical_experts)
            assert old_global_expert_indices.shape[
                1] % num_local_physical_experts == 0
            old_ep_size = old_global_expert_indices.shape[
                1] // num_local_physical_experts
            rank_mapping = {
                old_ep_rank: old_ep_rank
                for old_ep_rank in range(old_ep_size)
            }
        else:
            global_expert_load = None
            old_global_expert_indices = None
            rank_mapping = None

        with DeviceMemoryProfiler() as m:
            time_before_load = time.perf_counter()
            model_loader = get_model_loader(self.load_config)
            logger.info("Loading model from scratch...")
            self.model = model_loader.load_model(
                vllm_config=self.vllm_config, model_config=self.model_config)
            if self.lora_config:
                self.model = self.load_lora_model(self.model,
                                                  self.model_config,
                                                  self.scheduler_config,
                                                  self.lora_config,
                                                  self.device)
            if hasattr(self, "drafter"):
                logger.info("Loading drafter model...")
                self.drafter.load_model(self.model)
            if self.use_aux_hidden_state_outputs:
                if supports_eagle3(self.model):
                    self.model.set_aux_hidden_state_layers(
                        self.model.get_eagle3_aux_hidden_state_layers())
                else:
                    raise RuntimeError(
                        "Model does not support EAGLE3 interface but "
                        "aux_hidden_state_outputs was requested")
            time_after_load = time.perf_counter()
        self.model_memory_usage = m.consumed_memory
        logger.info("Model loading took %.4f GiB and %.6f seconds",
                    self.model_memory_usage / GiB_bytes,
                    time_after_load - time_before_load)
        prepare_communication_buffer_for_model(self.model)

        if is_mixture_of_experts(
                self.model) and self.parallel_config.enable_eplb:
            logger.info("EPLB is enabled for model %s.",
                        self.model_config.model)
            self.eplb_state = EplbState.build(
                self.model,
                self.device,
                self.parallel_config,
                global_expert_load,
                old_global_expert_indices,
                rank_mapping,
            )

        if (
            self.vllm_config.compilation_config.level == \
                CompilationLevel.DYNAMO_AS_IS and supports_dynamo()
        ):
            backend = self.vllm_config.compilation_config.init_backend(
                self.vllm_config)
            compilation_counter.dynamo_as_is_count += 1
            self.model.compile(
                fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=backend)
            return
        # for other compilation levels, cudagraph behavior is controlled by
        # CudagraphWraper and CudagraphDispatcher of vllm.

        # wrap the model with full cudagraph wrapper if needed.
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.model = CUDAGraphWrapper(self.model,
                                          self.vllm_config,
                                          runtime_mode=CUDAGraphMode.FULL)

    def reload_weights(self) -> None:
        assert getattr(self, "model", None) is not None, \
            "Cannot reload weights before model is loaded."
        model_loader = get_model_loader(self.load_config)
        logger.info("Reloading weights inplace...")
        model = self.get_model()
        model_loader.load_weights(model, model_config=self.model_config)

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        model = self.get_model()
        TensorizerLoader.save_model(
            model,
            tensorizer_config=tensorizer_config,
            model_config=self.model_config,
        )

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        scheduler_output: "SchedulerOutput",
    ) -> dict[str, Optional[LogprobsTensors]]:
        num_prompt_logprobs_dict = self.input_batch.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():

            num_tokens = scheduler_output.num_scheduled_tokens[req_id]

            # Get metadata for this request.
            request = self.requests[req_id]
            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True)

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1)
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc_np[req_idx].item()
            prompt_hidden_states = hidden_states[offset:offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states, None)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok:start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids)

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True)
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs,
                                                         non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True)

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_nans_in_logits(
        self,
        logits: Optional[torch.Tensor],
    ) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])
                    if num_nans_for_index is not None
                    and req_index < logits.shape[0] else 0)
            return num_nans_in_logits
        except IndexError:
            return {}

    @contextmanager
    def maybe_randomize_inputs(self, input_ids: torch.Tensor):
        """
        Randomize input_ids if VLLM_RANDOMIZE_DP_DUMMY_INPUTS is set.
        This is to help balance expert-selection
         - during profile_run
         - during DP rank dummy run
        """
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1
        if not randomize_inputs:
            yield
        else:
            import functools

            @functools.cache
            def rand_input_ids() -> torch.Tensor:
                return torch.randint_like(
                    self.input_ids,
                    low=0,
                    high=self.model_config.get_vocab_size(),
                    dtype=input_ids.dtype)

            logger.debug_once("Randomizing dummy data for DP Rank")
            input_ids.copy_(rand_input_ids()[:input_ids.size(0)],
                            non_blocking=True)
            yield
            input_ids.fill_(0)

    def _get_mm_dummy_batch(
        self,
        modality: str,
        max_items_per_batch: int,
    ) -> BatchedTensorInputs:
        """Dummy data for profiling and precompiling multimodal models."""
        dummy_decoder_data = self.mm_registry.get_decoder_dummy_data(
            model_config=self.model_config,
            seq_len=self.max_num_tokens,
            mm_counts={modality: 1},
        )
        dummy_mm_data = dummy_decoder_data.multi_modal_data

        # Result in the maximum GPU consumption of the model
        dummy_mm_item = dummy_mm_data.get_item(modality=modality, item_index=0)

        return next(mm_kwargs_group
                    for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
                        [dummy_mm_item] * max_items_per_batch,
                        device=self.device,
                        pin_memory=self.pin_memory,
                    ))

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        force_attention: bool = False,
        uniform_decode: bool = False,
        skip_eplb: bool = False,
        is_profile: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to 
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
        """
        assert cudagraph_runtime_mode in {
            CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
        }

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
        num_tokens += num_pad

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.seperate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else \
                                                                num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if uniform_decode:
            num_reqs = cdiv(num_tokens, max_query_len)
            assert num_reqs <= max_num_reqs, \
                "Do not capture num_reqs > max_num_reqs for uniform batch"
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)

        attn_metadata: Optional[dict[str, Any]] = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == \
                CUDAGraphMode.FULL:
            attn_metadata = {}

            # Make sure max_model_len is used at the graph capture time.
            self.seq_lens_np[:num_reqs] = self.max_model_len
            self.seq_lens_np[num_reqs:] = 0
            self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                           non_blocking=True)

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                common_attn_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc[:num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc_cpu[:num_reqs +
                                                                 1],
                    seq_lens=self.seq_lens[:num_reqs],
                    seq_lens_cpu=self.seq_lens_cpu[:num_reqs],
                    num_computed_tokens_cpu=self.input_batch.
                    num_computed_tokens_cpu_tensor[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    max_query_len=max_query_len,
                    block_table_tensor=self.input_batch.block_table[
                        kv_cache_group_id].get_device_tensor()[:num_reqs],
                    slot_mapping=self.input_batch.
                    block_table[kv_cache_group_id].slot_mapping[:num_tokens],
                    quantizer_id=self.input_batch.quantizer_id_gpu[:num_reqs],
                    tokens_per_page=self.input_batch.tokens_per_page_gpu[:num_reqs],
                    quantized_page_ids=self.input_batch.block_table[
                        kv_cache_group_id].quantized_page_ids[:num_tokens],
                    quantized_page_offsets=self.input_batch.block_table[
                        kv_cache_group_id].quantized_page_offsets[:num_tokens],
                    causal=True)

                for attn_group in self.attn_groups[kv_cache_group_id]:
                    attn_metadata_i = attn_group.metadata_builder\
                        .build_for_cudagraph_capture(common_attn_metadata)
                    for layer_name in kv_cache_group_spec.layer_names:
                        attn_metadata[layer_name] = attn_metadata_i

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens):
            if self.supports_mm_inputs:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
                model_kwargs = {
                    **self._init_model_kwargs(num_tokens),
                    **self._dummy_mm_kwargs(num_reqs),
                }
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None
                model_kwargs = self._init_model_kwargs(num_tokens)

            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_tokens]
            else:
                positions = self.positions[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device))

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens, None, False)
            if cudagraph_runtime_mode == CUDAGraphMode.NONE:
                batch_descriptor = None
            else:
                # filter out the valid batch descriptor
                _cg_mode, batch_descriptor = \
                    self.cudagraph_dispatcher.dispatch(
                        BatchDescriptor(num_tokens=num_tokens,
                                        uniform_decode=uniform_decode))
                # sanity check
                assert cudagraph_runtime_mode == _cg_mode, (
                    f"Cudagraph runtime mode mismatch at dummy_run. "
                    f"Expected {_cg_mode}, but got {cudagraph_runtime_mode}.")

            with self.maybe_randomize_inputs(input_ids), set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                self.drafter.dummy_run(num_tokens)

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        return hidden_states, hidden_states[logit_indices]

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # The dummy hidden states may contain special values,
        # like `inf` or `nan`.
        # To avoid breaking the sampler, we use a random tensor here instead.
        hidden_states = torch.rand_like(hidden_states)

        logits = self.model.compute_logits(hidden_states, None)
        num_reqs = logits.size(0)

        dummy_tensors = lambda v: torch.full(
            (num_reqs, ), v, device=self.device)

        dummy_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.5),
            all_greedy=False,
            all_random=False,
            top_p=dummy_tensors(0.9),
            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        try:
            sampler_output = self.sampler(logits=logits,
                                          sampling_metadata=dummy_metadata)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine.") from e
            else:
                raise e
        if self.speculative_config:
            draft_token_ids = [[0] for _ in range(num_reqs)]
            dummy_spec_decode_metadata = SpecDecodeMetadata.make_dummy(
                draft_token_ids, self.device)

            num_tokens = sum(len(ids) for ids in draft_token_ids)
            # draft_probs = torch.randn(
            #     num_tokens, logits.shape[-1], device=self.device,
            #     dtype=logits.dtype)
            draft_probs = None
            target_logits = torch.randn(num_tokens,
                                        logits.shape[-1],
                                        device=self.device,
                                        dtype=logits.dtype)
            # NOTE(woosuk): Here, we should use int32 because the sampler uses
            # int32 for bonus_token_ids. If the dtype mismatches, re-compilation
            # will occur at runtime.
            bonus_token_ids = torch.zeros(num_reqs,
                                          device=self.device,
                                          dtype=torch.int32)
            self.rejection_sampler(
                dummy_spec_decode_metadata,
                draft_probs,
                target_logits,
                bonus_token_ids,
                dummy_metadata,
            )
        return sampler_output

    def _dummy_pooler_run_task(
        self,
        hidden_states: torch.Tensor,
        task: PoolingTask,
    ) -> PoolerOutput:
        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs

        hidden_states_list = list(
            torch.split(hidden_states, num_scheduled_tokens_list))
        req_num_tokens = num_tokens // num_reqs

        dummy_prompt_lens = torch.tensor(
            [h.shape[0] for h in hidden_states_list],
            device=self.device,
        )
        dummy_token_ids = torch.zeros((num_reqs, req_num_tokens),
                                      dtype=torch.int32,
                                      device=self.device)

        model = cast(VllmModelForPooling, self.get_model())
        dummy_pooling_params = PoolingParams(task=task)
        to_update = model.pooler.get_pooling_updates(task)
        to_update.apply(dummy_pooling_params)

        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            pooling_params=[dummy_pooling_params] * num_reqs,
        )

        try:
            return model.pooler(hidden_states=hidden_states_list,
                                pooling_metadata=dummy_metadata)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler "
                    f"({task=}) with {num_reqs} dummy requests. Please try "
                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine.") from e
            else:
                raise e

    @torch.inference_mode()
    def _dummy_pooler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> PoolerOutput:
        # Find the task that has the largest output for subsequent steps
        output_size = dict[PoolingTask, float]()
        for task in self.get_supported_pooling_tasks():
            # Run a full batch with each task to ensure none of them OOMs
            output = self._dummy_pooler_run_task(hidden_states, task)
            output_size[task] = output.get_data_nbytes()
            del output  # Allow GC

        max_task = max(output_size.items(), key=lambda x: x[1])[0]
        return self._dummy_pooler_run_task(hidden_states, max_task)

    def profile_run(self) -> None:
        # Profile with multimodal encoder & encoder cache.
        if self.supports_mm_inputs:
            if self.model_config.multimodal_config.skip_mm_profiling:
                logger.info(
                    "Skipping memory profiling for multimodal encoder and "
                    "encoder cache.")
            else:
                mm_budget = self.mm_budget
                assert mm_budget is not None

                # TODO: handle encoder-decoder models once we support them.
                if (encoder_budget := mm_budget.get_encoder_budget()) > 0:
                    # NOTE: Currently model is profiled with a single non-text
                    # modality with the max possible input tokens even when
                    # it supports multiple.
                    (
                        dummy_modality,
                        max_tokens,
                    ) = mm_budget.get_modality_with_max_tokens()
                    (
                        max_mm_items_per_prompt,
                        max_mm_items_per_batch,
                    ) = mm_budget.get_max_items(dummy_modality, max_tokens)

                    logger.info(
                        "Encoder cache will be initialized with a budget of "
                        "%s tokens, and profiled with %s %s items of the "
                        "maximum feature size.",
                        encoder_budget,
                        max_mm_items_per_batch,
                        dummy_modality,
                    )

                    # Create dummy batch of multimodal inputs.
                    batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                        dummy_modality,
                        max_mm_items_per_batch,
                    )

                    # Run multimodal encoder.
                    dummy_encoder_outputs = \
                        self.model.get_multimodal_embeddings(
                        **batched_dummy_mm_inputs)

                    sanity_check_mm_encoder_outputs(
                        dummy_encoder_outputs,
                        expected_num_items=max_mm_items_per_batch,
                    )

                    # Cache the dummy encoder outputs.
                    self.encoder_cache["tmp"] = dict(
                        enumerate(dummy_encoder_outputs))

        # Add `is_profile` here to pre-allocate communication buffers
        hidden_states, last_hidden_states \
            = self._dummy_run(self.max_num_tokens, is_profile=True)
        if get_pp_group().is_last_rank:
            if self.is_pooling_model:
                output = self._dummy_pooler_run(hidden_states)
            else:
                output = self._dummy_sampler_run(last_hidden_states)
        else:
            output = None
        self._sync_device()
        del hidden_states, output
        self.encoder_cache.clear()
        gc.collect()

    def capture_model(self) -> None:
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`")
            return
        else:
            self.initialize_cudagraph_capture()

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        @contextmanager
        def freeze_gc():
            # Optimize garbage collection during CUDA graph capture.
            # Clean up, then freeze all remaining objects from being included
            # in future collections.
            gc.collect()
            should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
            if should_freeze:
                gc.freeze()
            try:
                yield
            finally:
                if should_freeze:
                    gc.unfreeze()

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        set_cudagraph_capturing_enabled(True)
        with freeze_gc(), graph_capture(device=self.device):
            cudagraph_mode = self.compilation_config.cudagraph_mode
            if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
                cudagraph_runtime_mode = cudagraph_mode.mixed_mode()

                compilation_cases = list(reversed(self.cudagraph_batch_sizes))
                self._capture_cudagraphs(
                    compilation_cases,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    uniform_decode=False)

            # Capture full cudagraph for uniform decode batches if we have
            # dont already have full mixed prefill-decode cudagraphs
            if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL and \
                cudagraph_mode.separate_routine():
                max_num_tokens = self.scheduler_config.max_num_seqs * \
                        self.uniform_decode_query_len
                decode_cudagraph_batch_sizes = [
                    x for x in self.cudagraph_batch_sizes if
                    x <= max_num_tokens and x >= self.uniform_decode_query_len
                ]
                compilation_cases_decode = list(
                    reversed(decode_cudagraph_batch_sizes))
                self._capture_cudagraphs(
                    compilation_cases=compilation_cases_decode,
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    uniform_decode=True)

        # Disable cudagraph capturing globally, so any unexpected cudagraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may doing lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))

    def _capture_cudagraphs(self, compilation_cases: list[int],
                            cudagraph_runtime_mode: CUDAGraphMode,
                            uniform_decode: bool):
        assert cudagraph_runtime_mode != CUDAGraphMode.NONE and \
            cudagraph_runtime_mode in [CUDAGraphMode.FULL,
                                        CUDAGraphMode.PIECEWISE]

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():
            compilation_cases = tqdm(
                compilation_cases,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing CUDA graphs ({}, {})".format(
                    "decode" if uniform_decode else "mixed prefill-decode",
                    cudagraph_runtime_mode.name))
        # We skip EPLB here since we don't want to record dummy metrics
        for num_tokens in compilation_cases:
            for _ in range(self.compilation_config.cudagraph_num_of_warmups):
                # Use CUDAGraphRuntimeStyle.NONE (default) for warmup.
                # But be careful, warm up with `NONE`is orthogonal to
                # if we want to warm up attention or not. This is
                # different from the case where `FULL` implies capture
                # attention while `PIECEWISE` implies no attention.
                force_attention = (
                    cudagraph_runtime_mode == CUDAGraphMode.FULL)
                self._dummy_run(num_tokens,
                                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                force_attention=force_attention,
                                uniform_decode=uniform_decode,
                                skip_eplb=True)
            self._dummy_run(num_tokens,
                            cudagraph_runtime_mode=cudagraph_runtime_mode,
                            uniform_decode=uniform_decode,
                            skip_eplb=True)

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, \
            "Attention backends are already initialized"
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)

        def get_attn_backends_for_layers(
                layer_names: list[str]
        ) -> dict[type[AttentionBackend], list[str]]:
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than using
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in layer_names:
                attn_backend = attn_layers[layer_name].get_attn_backend()
                key = attn_backend.full_cls_name()
                attn_backends[key] = attn_backend
                attn_backend_layers[key].append(layer_name)
            return {
                attn_backends[k]: v
                for k, v in attn_backend_layers.items()
            }

        def create_attn_groups(
            attn_backends_map: dict[AttentionBackend, list[str]],
            kv_cache_spec: KVCacheSpec,
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for attn_backend, layer_names in attn_backends_map.items():
                attn_metadata_builder_i = attn_backend.get_builder_cls()(
                    kv_cache_spec,
                    layer_names,
                    self.vllm_config,
                    self.device,
                )
                attn_group = AttentionGroup(attn_backend,
                                            attn_metadata_builder_i,
                                            layer_names)
                attn_groups.append(attn_group)
            return attn_groups

        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group_spec.kv_cache_spec
            if isinstance(kv_cache_spec, AttentionSpec):
                attn_backends = get_attn_backends_for_layers(
                    kv_cache_group_spec.layer_names)
            # TODO(lucas): move `get_mamba_attn_backend` into the mamba
            # layers like above
            elif isinstance(kv_cache_spec, MambaSpec):
                attn_backends = {
                    get_mamba_attn_backend(kv_cache_spec.mamba_type):
                    kv_cache_group_spec.layer_names
                }
            else:
                raise ValueError(
                    f"Unknown KV cache spec type: {type(kv_cache_spec)}")

            self.attn_groups.append(
                create_attn_groups(attn_backends, kv_cache_spec))

        # Calculate reorder batch threshold (if neeeded)
        self.calculate_reorder_batch_threshold()

        if len(self.attn_groups) > 0:
            return

        # Check if model is encoder-only
        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        attn_specs: dict[AttentionSpec, list[str]] = defaultdict(list)
        for layer_name, attn_module in attn_layers.items():

            if attn_module.attn_type == AttentionType.ENCODER_ONLY:
                if attn_module.sliding_window is None:
                    attn_spec: AttentionSpec = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        use_mla=use_mla)
                else:
                    attn_spec = SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        sliding_window=attn_module.sliding_window,
                        use_mla=use_mla)
                attn_specs[attn_spec].append(layer_name)

            else:
                raise ValueError("Expected only encoder-only layers")

        if len(attn_specs) > 0:
            total_layers = 0
            for attn_spec, layer_names in attn_specs.items():

                attn_backends = get_attn_backends_for_layers(layer_names)
                total_layers += len(layer_names)

                self.attn_groups.append(
                    create_attn_groups(attn_backends, attn_spec))
            assert total_layers == len(attn_layers), \
                "All or none of the layers are expected to be encoder-only"
            self.is_encoder_only_model = True

    def initialize_cudagraph_capture(self) -> None:
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_builder_name = None

        for attn_group in self._attn_group_iterator():
            builder = attn_group.metadata_builder
            if builder.cudagraph_support.value < min_cg_support.value:
                min_cg_support = builder.cudagraph_support
                min_cg_builder_name = builder.__class__.__name__

        # Flexible resolve the cudagraph mode
        cudagraph_mode = self.compilation_config.cudagraph_mode
        # check cudagraph for mixed batch is supported
        if cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL \
            and min_cg_support != AttentionCGSupport.ALWAYS:
            msg = (f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                   f"with {min_cg_builder_name} backend (support: "
                   f"{min_cg_support})")
            if min_cg_support == AttentionCGSupport.NEVER:
                # if not supported any full cudagraphs, just raise it.
                msg += "; please try cudagraph_mode=PIECEWISE, and "\
                    "make sure compilation level is piecewise"
                raise ValueError(msg)

            # attempt to resolve the full cudagraph related mode
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=FULL_AND_PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = \
                    CUDAGraphMode.FULL_AND_PIECEWISE
            else:
                msg += "; setting cudagraph_mode=FULL_DECODE_ONLY"
                cudagraph_mode = self.compilation_config.cudagraph_mode = \
                    CUDAGraphMode.FULL_DECODE_ONLY
            logger.warning(msg)

        # check that if we are doing spec-decode + decode full-cudagraphs it is
        # supported
        if (cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
                and self.uniform_decode_query_len > 1 and min_cg_support.value
                < AttentionCGSupport.UNIFORM_BATCH.value):
            msg = (f"CUDAGraphMode.{cudagraph_mode.name} is not supported"
                   f" with spec-decode for attention backend "
                   f"{min_cg_builder_name} (support: {min_cg_support})")
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = \
                    CUDAGraphMode.PIECEWISE
            else:
                msg += "; setting cudagraph_mode=NONE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = \
                    CUDAGraphMode.NONE
            logger.warning(msg)

        # double check that we can support full cudagraph if they are requested
        # even after automatic downgrades
        if cudagraph_mode.has_full_cudagraphs() \
            and min_cg_support == AttentionCGSupport.NEVER:
            raise ValueError(f"CUDAGraphMode.{cudagraph_mode.name} is not "
                             f"supported with {min_cg_builder_name} backend ("
                             f"support:{min_cg_support}) "
                             "; please try cudagraph_mode=PIECEWISE, "
                             "and make sure compilation level is piecewise")

        # Trigger cudagraph dispatching keys initialization here (after
        # initializing attn backends).
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            self.compilation_config.cudagraph_mode,
            self.uniform_decode_query_len)

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Check that if any backends reorder batches; that the reordering
        is compatible (e.g., decode threshold is the same)
        """
        for group in self._attn_group_iterator():
            attn_metadata_builder_i = group.metadata_builder

            # check that if any backends reorder batches; that the reordering
            # is compatible (e.g., decode threshold is the same)
            reorder_batch_threshold_i = (
                attn_metadata_builder_i.reorder_batch_threshold)
            if reorder_batch_threshold_i is not None:
                if self.reorder_batch_threshold is not None:
                    if reorder_batch_threshold_i != \
                        self.reorder_batch_threshold:
                        raise ValueError(
                            f"Attention backend reorders decodes with "
                            f"threshold {reorder_batch_threshold_i} but other "
                            f"backend uses threshold "
                            f"{self.reorder_batch_threshold}")
                else:
                    self.reorder_batch_threshold = reorder_batch_threshold_i

    def may_reinitialize_input_batch(self,
                                     kv_cache_config: KVCacheConfig) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
        ]
        if block_sizes != [self.cache_config.block_size]:
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details.")
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                is_pooling_model=self.is_pooling_model,
            )

    @staticmethod
    def allocate_fixed_byte_kv_page_pool(
        *,
        num_pages: int,
        page_bytes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Allocate a common fixed-byte KV-page pool for one cache group.

        The pool is deliberately uint8: an experimental heterogeneous
        quantized attention backend owns the interpretation of page bytes.
        Standard V1 K/V cache backends must not consume this tensor.
        """
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")
        if page_bytes <= 0:
            raise ValueError("page_bytes must be positive")

        return torch.zeros(
            (num_pages, page_bytes),
            dtype=torch.uint8,
            device=device,
        )

    def get_cachegen_int8_page_adapter(
        self,
        *,
        layer_name: str,
    ) -> CacheGenInt8FixedBytePageAdapter:
        """Return the CacheGen byte-page adapter for one attention layer."""
        layer_adapters = getattr(self, "cachegen_int8_page_adapters", {})
        if layer_adapters:
            try:
                return layer_adapters[layer_name]
            except KeyError as exc:
                raise ValueError(
                    "No CacheGen-style byte-page adapter is configured for "
                    f"layer {layer_name!r}"
                ) from exc

        adapter = getattr(self, "cachegen_int8_page_adapter", None)
        if adapter is not None:
            return adapter

        raise ValueError(
            "No CacheGen-style byte-page adapter is configured for "
            f"layer {layer_name!r}"
        )

    def _remap_cachegen_int8_shadow_page_ids(
        self,
        *,
        page_ids: torch.Tensor,
        adapter: CacheGenInt8FixedBytePageAdapter,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map normal KV page IDs into the bounded shadow-pool ID space.

        The mapping is runner-global rather than layer-local so every captured
        layer writes the same normal KV locations to the same local page IDs.
        Entries that cannot be assigned a bounded local page are omitted.
        """
        if page_ids.ndim != 1:
            raise ValueError(
                f"page_ids must be rank 1, got shape {tuple(page_ids.shape)}"
            )

        remapped_ids = torch.empty_like(page_ids)
        keep_mask = torch.zeros_like(page_ids, dtype=torch.bool)

        for index, normal_page_id in enumerate(page_ids.tolist()):
            shadow_page_id = self.cachegen_int8_shadow_page_id_map.get(
                normal_page_id
            )
            if shadow_page_id is None:
                if len(self.cachegen_int8_shadow_page_id_map) >= adapter.num_pages:
                    self.cachegen_int8_shadow_dropped_page_ids.add(
                        normal_page_id
                    )
                    continue
                shadow_page_id = len(self.cachegen_int8_shadow_page_id_map)
                self.cachegen_int8_shadow_page_id_map[normal_page_id] = (
                    shadow_page_id
                )
            remapped_ids[index] = shadow_page_id
            keep_mask[index] = True

        return remapped_ids, keep_mask

    def _get_cachegen_kv_quantizer_for_metadata(
        self,
        *,
        request_ids: list[str],
    ) -> str:
        """Return one safe quantizer selection for an attention metadata batch.

        Explicit per-request assignment has priority. For backward-compatible
        single-request experiments, the enabled CacheGen INT8 environment
        route supplies an INT8 default when no assignment exists. Mixed
        batches always remain native because the initial fused route supports
        exactly one sequence.
        """
        if len(request_ids) != 1:
            return CacheGenKVQuantizer.NATIVE.value

        request_id = request_ids[0]
        assignment = self.cachegen_kv_quantizer_assignments.get(request_id)
        if assignment is not None:
            return assignment.quantizer_id

        if is_cachegen_int8_attention_enabled():
            return CacheGenKVQuantizer.INT8_ADAPTIVE.value
        return CacheGenKVQuantizer.NATIVE.value

    def _attach_cachegen_int8_decode_validation(
        self,
        *,
        layer_name: str,
        attn_metadata: Any,
        kv_cache_group_id: int,
    ) -> None:
        """Attach selected-layer dynamic INT8 fused-decode validation.

        This callback writes actual K/V values to compact INT8 pages and
        computes a fused candidate. Native FlashAttention remains responsible
        for the model output; its output is compared after native attention.
        """
        if not is_cachegen_int8_attention_enabled():
            return

        selected_layer = self.cachegen_int8_resolved_attention_layer
        if layer_name != selected_layer:
            return

        if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            raise ValueError(
                "CacheGen INT8 decode validation requires cudagraph_mode=NONE"
            )
        if not hasattr(attn_metadata, "cachegen_int8_decode_validation"):
            return

        selected_cache = self.cachegen_int8_selected_layer_cache
        if selected_cache is None:
            raise AssertionError(
                "CacheGen INT8 selected-layer cache is not initialized"
            )

        native_block_size = self.input_batch.block_table[
            kv_cache_group_id
        ].block_size
        attention_layers = get_layers_from_vllm_config(
            self.vllm_config,
            Attention,
        )
        try:
            attention_layer = attention_layers[layer_name]
        except KeyError as exc:
            raise ValueError(
                "No Attention module found for CacheGen INT8 decode validation "
                f"layer {layer_name!r}"
            ) from exc

        def validate(
            query: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            metadata: Any,
        ) -> None:
            stats = self.cachegen_int8_decode_validation_stats
            stats["callback_calls"] = int(stats["callback_calls"]) + 1

            native_slots = metadata.slot_mapping[:metadata.num_actual_tokens]
            native_page_ids = torch.div(
                native_slots,
                native_block_size,
                rounding_mode="floor",
            ).to(torch.int32)
            native_page_offsets = torch.remainder(
                native_slots,
                native_block_size,
            ).to(torch.int32)

            wrote_pages = selected_cache.write_native_mapped_tokens(
                keys=keys,
                values=values,
                native_page_ids=native_page_ids,
                page_offsets=native_page_offsets,
            )
            if not wrote_pages:
                stats["route_capacity_fallbacks"] = (
                    int(stats["route_capacity_fallbacks"]) + 1
                )
                return
            stats["tokens_written"] = (
                int(stats["tokens_written"]) + keys.shape[0]
            )

            decision = get_cachegen_int8_decode_route_decision(
                num_actual_tokens=metadata.num_actual_tokens,
                max_query_len=metadata.max_query_len,
                seq_lens=metadata.seq_lens,
                query_start_loc=metadata.query_start_loc,
                block_table=metadata.block_table,
                block_size=native_block_size,
                use_cascade=metadata.use_cascade,
                sliding_window=attention_layer.impl.sliding_window,
                has_kv_sharing=(
                    attention_layer.kv_sharing_target_layer_name is not None
                ),
            )
            if not decision.is_eligible:
                stats["route_unsupported_fallbacks"] = (
                    int(stats["route_unsupported_fallbacks"]) + 1
                )
                reason = decision.reason
                assert reason is not None
                fallback_reasons = stats.setdefault(
                    "route_fallback_reasons",
                    {},
                )
                assert isinstance(fallback_reasons, dict)
                fallback_reasons[reason] = (
                    int(fallback_reasons.get(reason, 0)) + 1
                )
                return

            plan = decision.plan
            assert plan is not None
            compact_block_table = selected_cache.map_active_block_table(
                plan.block_table
            )
            if compact_block_table is None:
                stats["route_capacity_fallbacks"] = (
                    int(stats["route_capacity_fallbacks"]) + 1
                )
                return

            metadata.cachegen_int8_fused_decode_output = (
                cachegen_int8_paged_decode_attention_triton(
                    query=query[0],
                    cache=selected_cache.page_store.as_paged_kv(),
                    block_table=compact_block_table,
                    seq_len=plan.sequence_length,
                )
            )

            if selected_cache.has_common_page_ids(
                compact_page_ids=compact_block_table,
            ):
                common_page_ids = selected_cache.get_common_page_ids(
                    compact_page_ids=compact_block_table,
                )
                common_page_format = (
                    selected_cache.common_page_codec.page_format
                )
                key_scales = selected_cache.common_page_codec.key_scales_by_page
                value_scales = (
                    selected_cache.common_page_codec.value_scales_by_page
                )
                assert key_scales is not None
                assert value_scales is not None
                metadata.cachegen_int8_common_page_decode_output = (
                    cachegen_int8_common_page_decode_attention_triton(
                        query=query[0],
                        page_bytes=selected_cache.common_page_pool.page_bytes,
                        key_scales=key_scales,
                        value_scales=value_scales,
                        page_ids=common_page_ids,
                        sequence_length=plan.sequence_length,
                        num_kv_heads=selected_cache.common_page_codec.
                        layout.num_kv_heads,
                        tokens_per_page=selected_cache.common_page_codec.
                        layout.tokens_per_page,
                        head_size=selected_cache.common_page_codec.layout.
                        head_size,
                        metadata_nbytes=common_page_format.metadata_nbytes,
                        key_payload_offset=common_page_format.key_payload_offset,
                        value_payload_offset=(
                            common_page_format.value_payload_offset
                        ),
                    )
                )
                stats["common_page_fused_calls"] = (
                    int(stats.setdefault("common_page_fused_calls", 0)) + 1
                )

                if get_cachegen_int8_attention_mode() == "common_page_int8":
                    metadata.cachegen_int8_fused_decode_output = (
                        metadata.cachegen_int8_common_page_decode_output
                    )

            else:
                stats["common_page_missing_mirror_fallbacks"] = (
                    int(
                        stats.setdefault(
                            "common_page_missing_mirror_fallbacks",
                            0,
                        )
                    )
                    + 1
                )
            stats["route_eligible_calls"] = (
                int(stats["route_eligible_calls"]) + 1
            )

        attn_metadata.cachegen_int8_decode_validation = validate

        def record_result(result: dict[str, float]) -> None:
            stats = self.cachegen_int8_decode_validation_stats
            stats["max_abs_error"] = max(
                float(stats["max_abs_error"]),
                result["max_abs_error"],
            )
            stats["mean_abs_error_sum"] = (
                float(stats["mean_abs_error_sum"])
                + result["mean_abs_error"]
            )
            stats["mean_abs_error_count"] = (
                int(stats["mean_abs_error_count"]) + 1
            )

            common_page_max_error = result.get(
                "common_page_max_abs_error_vs_typed"
            )
            common_page_mean_error = result.get(
                "common_page_mean_abs_error_vs_typed"
            )
            if (
                common_page_max_error is not None
                and common_page_mean_error is not None
            ):
                stats["common_page_max_abs_error_vs_typed"] = max(
                    float(
                        stats.setdefault(
                            "common_page_max_abs_error_vs_typed",
                            0.0,
                        )
                    ),
                    float(common_page_max_error),
                )
                stats["common_page_mean_abs_error_vs_typed_sum"] = (
                    float(
                        stats.setdefault(
                            "common_page_mean_abs_error_vs_typed_sum",
                            0.0,
                        )
                    )
                    + float(common_page_mean_error)
                )
                stats["common_page_mean_abs_error_vs_typed_count"] = (
                    int(
                        stats.setdefault(
                            "common_page_mean_abs_error_vs_typed_count",
                            0,
                        )
                    )
                    + 1
                )

            mode = get_cachegen_int8_attention_mode()
            if mode == "common_page_int8":
                stats["common_page_int8_output_calls"] = (
                    int(
                        stats.setdefault(
                            "common_page_int8_output_calls",
                            0,
                        )
                    )
                    + 1
                )
            elif mode == "int8":
                stats["int8_output_calls"] = (
                    int(stats.setdefault("int8_output_calls", 0)) + 1
                )
            else:
                stats["native_output_calls"] = (
                    int(stats.setdefault("native_output_calls", 0)) + 1
                )

        attn_metadata.cachegen_int8_decode_validation_result = record_result

    def _attach_cachegen_int8_calibration(
        self,
        *,
        layer_name: str,
        attn_metadata: Any,
    ) -> None:
        """Attach per-KV-head K/V range collection for one selected layer."""
        selected_layer = get_cachegen_int8_calibration_layer()
        if selected_layer is None or layer_name != selected_layer:
            return

        if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            raise ValueError(
                "CacheGen INT8 calibration requires cudagraph_mode=NONE"
            )
        if not hasattr(attn_metadata, "cachegen_int8_calibration"):
            return

        if self.cachegen_int8_calibration_stats is None:
            attention_layers = get_layers_from_vllm_config(
                self.vllm_config,
                Attention,
            )
            try:
                attention_layer = attention_layers[layer_name]
            except KeyError as exc:
                raise ValueError(
                    "No Attention module found for CacheGen INT8 calibration "
                    f"layer {layer_name!r}"
                ) from exc

            self.cachegen_int8_calibration_stats = {
                "layer_name": layer_name,
                "callback_calls": 0,
                "tokens_seen": 0,
                "key_abs_max_per_head": torch.zeros(
                    attention_layer.num_kv_heads,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "value_abs_max_per_head": torch.zeros(
                    attention_layer.num_kv_heads,
                    dtype=torch.float32,
                    device=self.device,
                ),
            }

        def calibrate(
            keys: torch.Tensor,
            values: torch.Tensor,
            metadata: Any,
        ) -> None:
            del metadata
            assert self.cachegen_int8_calibration_stats is not None

            key_abs_max = keys.float().abs().amax(dim=(0, 2))
            value_abs_max = values.float().abs().amax(dim=(0, 2))

            stored_key_abs_max = self.cachegen_int8_calibration_stats[
                "key_abs_max_per_head"
            ]
            stored_value_abs_max = self.cachegen_int8_calibration_stats[
                "value_abs_max_per_head"
            ]
            assert isinstance(stored_key_abs_max, torch.Tensor)
            assert isinstance(stored_value_abs_max, torch.Tensor)

            torch.maximum(
                stored_key_abs_max,
                key_abs_max,
                out=stored_key_abs_max,
            )
            torch.maximum(
                stored_value_abs_max,
                value_abs_max,
                out=stored_value_abs_max,
            )

            self.cachegen_int8_calibration_stats["callback_calls"] = (
                int(self.cachegen_int8_calibration_stats["callback_calls"]) + 1
            )
            self.cachegen_int8_calibration_stats["tokens_seen"] = (
                int(self.cachegen_int8_calibration_stats["tokens_seen"])
                + keys.shape[0]
            )

        attn_metadata.cachegen_int8_calibration = calibrate

    def _attach_cachegen_int8_shadow_write(
        self,
        *,
        layer_name: str,
        attn_metadata: Any,
    ) -> None:
        """Attach an eager-only CacheGen-style K/V shadow-write callback."""
        if not is_cachegen_int8_shadow_write_enabled():
            return

        all_layers_enabled = is_cachegen_int8_shadow_all_layers_enabled()
        selected_layer = (
            None
            if all_layers_enabled
            else get_cachegen_int8_shadow_layer()
        )
        if not all_layers_enabled and layer_name != selected_layer:
            return

        if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            raise ValueError(
                "CacheGen-style KV shadow write requires "
                "cudagraph_mode=NONE"
            )

        try:
            adapter = self.get_cachegen_int8_page_adapter(
                layer_name=layer_name,
            )
        except ValueError as exc:
            if all_layers_enabled:
                raise AssertionError(
                    "CacheGen-style all-layer shadow write requires a "
                    f"layer-local adapter for {layer_name!r}"
                ) from exc
            raise AssertionError(
                "CacheGen-style KV shadow write requires the page adapter"
            ) from exc

        if not hasattr(attn_metadata, "cachegen_int8_shadow_write"):
            return
        if attn_metadata.quantizer_id is None:
            raise AssertionError(
                "CacheGen-style KV shadow write requires quantizer IDs"
            )
        if torch.any(attn_metadata.quantizer_id != 2):
            raise ValueError(
                "CacheGen-style KV shadow write currently requires every "
                "active request to use quantizer_id=2"
            )

        def shadow_write(
            keys: torch.Tensor,
            values: torch.Tensor,
            metadata: Any,
        ) -> None:
            if metadata.quantized_page_ids is None:
                raise AssertionError(
                    "CacheGen-style KV shadow write requires page IDs"
                )
            if metadata.quantized_page_offsets is None:
                raise AssertionError(
                    "CacheGen-style KV shadow write requires page offsets"
                )

            quantized_page_ids = metadata.quantized_page_ids[
                :keys.shape[0]
            ]
            quantized_page_offsets = metadata.quantized_page_offsets[
                :keys.shape[0]
            ]
            capture_mask = quantized_page_ids < adapter.num_pages

            if bool(capture_mask.any()):
                write_cachegen_int8_mapped_tokens(
                    adapter=adapter,
                    keys=keys[capture_mask],
                    values=values[capture_mask],
                    quantized_page_ids=quantized_page_ids[capture_mask],
                    quantized_page_offsets=quantized_page_offsets[capture_mask],
                )

            captured_tokens = int(capture_mask.sum().item())

            callback_calls = getattr(
                self,
                "cachegen_int8_shadow_callback_calls",
                None,
            )
            if callback_calls is not None:
                tokens_written = getattr(
                    self,
                    "cachegen_int8_shadow_tokens_written",
                    None,
                )
                assert tokens_written is not None
                callback_calls.add_(1)
                tokens_written.add_(captured_tokens)

            layer_counters = getattr(
                self,
                "cachegen_int8_shadow_layer_counters",
                {},
            )
            layer_counter = layer_counters.get(layer_name)
            if layer_counter is not None:
                layer_callback_calls, layer_tokens_written = layer_counter
                layer_callback_calls.add_(1)
                layer_tokens_written.add_(captured_tokens)

            if (
                captured_tokens > 0
                and is_cachegen_int8_shadow_sample_enabled()
                and self.cachegen_int8_shadow_sample is None
            ):
                first_captured_index = int(
                    torch.nonzero(capture_mask, as_tuple=False)[0].item()
                )
                page_id = int(quantized_page_ids[first_captured_index].item())
                page_offset = int(
                    quantized_page_offsets[first_captured_index].item()
                )
                decoded_key, decoded_value = adapter.read_token(
                    page_id=page_id,
                    page_offset=page_offset,
                    dtype=torch.float32,
                )
                key_error = decoded_key - keys[0].to(torch.float32)
                value_error = decoded_value - values[0].to(torch.float32)
                self.cachegen_int8_shadow_sample = {
                    "layer_name": layer_name,
                    "page_id": page_id,
                    "page_offset": page_offset,
                    "key_max_abs_error": float(key_error.abs().amax().item()),
                    "value_max_abs_error": float(
                        value_error.abs().amax().item()
                    ),
                    "key_rmse": float(key_error.square().mean().sqrt().item()),
                    "value_rmse": float(
                        value_error.square().mean().sqrt().item()
                    ),
                }

        attn_metadata.cachegen_int8_shadow_write = shadow_write

    def write_cachegen_int8_shadow_kv(
        self,
        *,
        layer_name: str,
        kv_cache_group_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Reference-write mapped K/V tokens into one layer's byte pages.

        This helper is intentionally not called by the normal model forward
        path. It validates runner-owned block-table mapping and byte-page
        layout before a future cache-read integration path is introduced.
        """
        try:
            adapter = self.get_cachegen_int8_page_adapter(layer_name=layer_name)
        except ValueError as exc:
            raise RuntimeError(
                "CacheGen-style byte-page adapter is not initialized"
            ) from exc

        num_kv_cache_groups = len(self.input_batch.block_table.block_tables)
        if not 0 <= kv_cache_group_id < num_kv_cache_groups:
            raise ValueError(
                f"kv_cache_group_id must be in "
                f"[0, {num_kv_cache_groups}), got {kv_cache_group_id}"
            )

        num_tokens = keys.shape[0]
        if values.shape != keys.shape:
            raise ValueError(
                f"values must have shape {tuple(keys.shape)}, "
                f"got {tuple(values.shape)}"
            )

        block_table = self.input_batch.block_table[kv_cache_group_id]
        write_cachegen_int8_mapped_tokens(
            adapter=adapter,
            keys=keys,
            values=values,
            quantized_page_ids=block_table.quantized_page_ids[:num_tokens],
            quantized_page_offsets=block_table.quantized_page_offsets[
                :num_tokens
            ],
        )

    def read_cachegen_int8_shadow_kv(
        self,
        *,
        layer_name: str,
        quantized_page_ids: torch.Tensor,
        quantized_page_offsets: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference-read mapped K/V tokens from one layer's byte pages."""
        if quantized_page_ids.ndim != 1:
            raise ValueError("quantized_page_ids must be one-dimensional")
        if quantized_page_offsets.shape != quantized_page_ids.shape:
            raise ValueError(
                "quantized_page_offsets must have the same shape as "
                "quantized_page_ids"
            )
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating point")

        adapter = self.get_cachegen_int8_page_adapter(layer_name=layer_name)
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for page_id, page_offset in zip(
            quantized_page_ids.tolist(),
            quantized_page_offsets.tolist(),
        ):
            key, value = adapter.read_token(
                page_id=page_id,
                page_offset=page_offset,
                dtype=dtype,
            )
            keys.append(key)
            values.append(value)

        if not keys:
            empty = torch.empty(
                (0, adapter.layout.num_kv_heads, adapter.layout.head_size),
                device=adapter.page_pool.device,
                dtype=dtype,
            )
            return empty, empty.clone()

        return torch.stack(keys), torch.stack(values)

    def assign_cachegen_kv_quantizer(
        self,
        request_id: str,
        quantizer: str,
    ) -> dict[str, str]:
        """Assign an experimental paged-KV quantizer to one request."""
        assignment = self.cachegen_kv_quantizer_assignments.assign(
            request_id=request_id,
            quantizer=quantizer,
        )
        return {
            "request_id": assignment.request_id,
            "quantizer": assignment.quantizer_id,
        }

    def remove_cachegen_kv_quantizer(
        self,
        request_id: str,
    ) -> dict[str, str] | None:
        """Remove one request's experimental paged-KV quantizer assignment."""
        assignment = self.cachegen_kv_quantizer_assignments.remove(request_id)
        if assignment is None:
            return None
        return {
            "request_id": assignment.request_id,
            "quantizer": assignment.quantizer_id,
        }

    def get_cachegen_kv_quantizer(
        self,
        request_id: str,
    ) -> str:
        """Return one request's quantizer, defaulting safely to native."""
        return self.cachegen_kv_quantizer_assignments.quantizer_for(
            request_id
        ).value

    def snapshot_cachegen_kv_quantizers(self) -> dict[str, str]:
        """Return all active request-to-quantizer assignments."""
        return self.cachegen_kv_quantizer_assignments.snapshot()

    def get_cachegen_kv_quantizer_assignment_stats(
        self,
    ) -> dict[str, object]:
        """Return active per-request quantizer assignment counts."""
        snapshot = self.cachegen_kv_quantizer_assignments.snapshot()
        counts = {
            quantizer.value: 0
            for quantizer in CacheGenKVQuantizer
        }
        for quantizer in snapshot.values():
            counts[quantizer] += 1

        return {
            "num_assignments": len(snapshot),
            "assignments": snapshot,
            "counts": counts,
        }

    def reset_cachegen_int8_decode_validation(self) -> None:
        """Reset selected-layer compact INT8 validation pages in place.

        This explicit diagnostic RPC is intentionally separate from native
        vLLM request/block lifecycle. It is used to start an independent
        experiment with empty compact pages and mappings.
        """
        selected_cache = self.cachegen_int8_selected_layer_cache
        if selected_cache is None:
            return

        selected_cache.reset()
        stats = self.cachegen_int8_decode_validation_stats
        stats["reset_calls"] = int(stats.setdefault("reset_calls", 0)) + 1

    def get_cachegen_int8_decode_validation_stats(
        self,
    ) -> dict[str, object] | None:
        """Return selected-layer dynamic INT8 fused-decode validation stats."""
        if not is_cachegen_int8_attention_enabled():
            return None

        stats = dict(self.cachegen_int8_decode_validation_stats)
        stats.setdefault("route_fallback_reasons", {})
        stats.setdefault("native_output_calls", 0)
        stats.setdefault("int8_output_calls", 0)
        stats.setdefault("common_page_int8_output_calls", 0)
        stats.setdefault("reset_calls", 0)
        stats.setdefault("common_page_fused_calls", 0)
        stats.setdefault("common_page_missing_mirror_fallbacks", 0)
        stats.setdefault("common_page_max_abs_error_vs_typed", 0.0)
        stats.setdefault("common_page_mean_abs_error_vs_typed_sum", 0.0)
        stats.setdefault("common_page_mean_abs_error_vs_typed_count", 0)
        stats["layer_name"] = self.cachegen_int8_resolved_attention_layer
        count = int(stats.pop("mean_abs_error_count"))
        total = float(stats.pop("mean_abs_error_sum"))
        stats["mean_abs_error"] = total / count if count else 0.0

        common_count = int(
            stats.pop("common_page_mean_abs_error_vs_typed_count", 0)
        )
        common_total = float(
            stats.pop("common_page_mean_abs_error_vs_typed_sum", 0.0)
        )
        stats["common_page_mean_abs_error_vs_typed"] = (
            common_total / common_count if common_count else 0.0
        )

        selected_cache = self.cachegen_int8_selected_layer_cache
        if selected_cache is not None:
            stats["mapped_pages"] = selected_cache.page_map.num_mapped_pages
            stats["remaining_pages"] = selected_cache.page_map.remaining_pages
            stats["int8_page_store_bytes"] = (
                selected_cache.page_store.persistent_nbytes
            )
        return stats

    def get_cachegen_int8_calibration_stats(
        self,
    ) -> dict[str, object] | None:
        """Return selected-layer K/V calibration ranges from live attention."""
        layer_name = get_cachegen_int8_calibration_layer()
        if layer_name is None:
            return None

        attention_layers = get_layers_from_vllm_config(
            self.vllm_config,
            Attention,
        )
        try:
            attention_layer = attention_layers[layer_name]
        except KeyError as exc:
            raise ValueError(
                "No Attention module found for CacheGen INT8 calibration "
                f"layer {layer_name!r}"
            ) from exc

        stats = getattr(
            attention_layer,
            "_cachegen_int8_calibration_stats",
            None,
        )
        if stats is None:
            return {
                "layer_name": layer_name,
                "callback_calls": 0,
                "tokens_seen": 0,
                "key_abs_max_per_head": [],
                "value_abs_max_per_head": [],
            }

        return {
            "layer_name": layer_name,
            "callback_calls": int(stats["callback_calls"]),
            "tokens_seen": int(stats["tokens_seen"]),
            "key_abs_max_per_head": (
                stats["key_abs_max_per_head"].float().cpu().tolist()
            ),
            "value_abs_max_per_head": (
                stats["value_abs_max_per_head"].float().cpu().tolist()
            ),
        }

    def get_cachegen_int8_decode_route_debug_state(
        self,
    ) -> dict[str, object]:
        """Return read-only metadata needed to validate decode-route inputs."""
        metadata_by_layer = getattr(
            self,
            "_last_cachegen_int8_attn_metadata",
            {},
        )
        return dict(metadata_by_layer)

    def get_cachegen_int8_layer_page_audit(
        self,
    ) -> dict[str, dict[str, float | int]]:
        """Read one mapped token from each configured layer-local byte pool.

        This eager diagnostic is intentionally bounded to one token per layer.
        It is not part of the attention read path and must not be used as a
        batched decode implementation.
        """
        layer_adapters = getattr(self, "cachegen_int8_page_adapters", {})
        if not layer_adapters:
            return {}

        block_table = self.input_batch.block_table[0]
        page_ids = block_table.quantized_page_ids
        page_offsets = block_table.quantized_page_offsets
        if page_ids.numel() == 0 or page_offsets.numel() == 0:
            return {}

        first_adapter = next(iter(layer_adapters.values()))
        valid_indices = torch.nonzero(
            (page_ids >= 0) & (page_ids < first_adapter.num_pages),
            as_tuple=False,
        )
        if valid_indices.numel() == 0:
            return {}

        index = int(valid_indices[0].item())
        page_id = int(page_ids[index].item())
        page_offset = int(page_offsets[index].item())
        audit: dict[str, dict[str, float | int]] = {}
        for layer_name, adapter in layer_adapters.items():
            key, value = adapter.read_token(
                page_id=page_id,
                page_offset=page_offset,
                dtype=torch.float32,
            )
            audit[layer_name] = {
                "page_id": page_id,
                "page_offset": page_offset,
                "key_abs_max": float(key.abs().amax().item()),
                "value_abs_max": float(value.abs().amax().item()),
            }
        return audit

    def _resolve_cachegen_int8_attention_layer(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> str:
        """Resolve a user selector to one live decoder AttentionSpec layer.

        Accepted selectors:
        - ``auto``: first compatible registered decoder attention layer.
        - ``last``: last compatible registered decoder attention layer.
        - Non-negative integer: ordinal among compatible decoder layers.
        - Exact vLLM cache-layer name.
        - An unambiguous architecture-path prefix, such as
          ``model.layers.0.self_attn`` for a registered
          ``model.layers.0.self_attn.attn`` leaf.
        """
        selector = get_cachegen_int8_attention_layer()
        if selector is None:
            selector = "auto"

        compatible_layers = [
            layer_name
            for group in kv_cache_config.kv_cache_groups
            if isinstance(group.kv_cache_spec, AttentionSpec)
            for layer_name in group.layer_names
        ]
        if not compatible_layers:
            raise ValueError(
                "CacheGen INT8 attention validation found no decoder "
                "AttentionSpec layers in the live KV-cache configuration."
            )

        if selector == "auto":
            resolved_layer = compatible_layers[0]
        elif selector == "last":
            resolved_layer = compatible_layers[-1]
        elif selector.isdecimal():
            layer_index = int(selector)
            if layer_index >= len(compatible_layers):
                raise ValueError(
                    "CacheGen INT8 attention layer index is out of range: "
                    f"selector={selector!r}, compatible_layer_count="
                    f"{len(compatible_layers)}"
                )
            resolved_layer = compatible_layers[layer_index]
        elif selector in compatible_layers:
            resolved_layer = selector
        else:
            prefix_matches = [
                layer_name
                for layer_name in compatible_layers
                if layer_name.startswith(selector + ".")
            ]
            if len(prefix_matches) == 1:
                resolved_layer = prefix_matches[0]
            elif len(prefix_matches) > 1:
                raise ValueError(
                    "CacheGen INT8 attention selector is ambiguous: "
                    f"selector={selector!r}, matches={prefix_matches}"
                )
            else:
                raise ValueError(
                    "CacheGen INT8 attention selector did not match a live "
                    "decoder cache layer: "
                    f"selector={selector!r}, "
                    f"compatible_layers={compatible_layers}"
                )

        logger.info(
            "CacheGen INT8 resolved attention selector %r to live layer %r",
            selector,
            resolved_layer,
        )
        return resolved_layer

    def _initialize_cachegen_int8_selected_layer_cache(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """Allocate dynamic INT8 validation pages for one selected layer."""
        self.cachegen_int8_selected_layer_cache = None
        self.cachegen_int8_decode_validation_stats = {
            "callback_calls": 0,
            "route_eligible_calls": 0,
            "route_capacity_fallbacks": 0,
            "route_unsupported_fallbacks": 0,
            "route_fallback_reasons": {},
            "native_output_calls": 0,
            "int8_output_calls": 0,
            "common_page_int8_output_calls": 0,
            "max_abs_error": 0.0,
            "mean_abs_error_sum": 0.0,
            "mean_abs_error_count": 0,
            "tokens_written": 0,
        }

        if not is_cachegen_int8_attention_enabled():
            return

        layer_name = self._resolve_cachegen_int8_attention_layer(
            kv_cache_config
        )
        self.cachegen_int8_resolved_attention_layer = layer_name
        max_pages = get_cachegen_int8_attention_max_pages()
        assert max_pages is not None

        decoder_specs = [
            group.kv_cache_spec
            for group in kv_cache_config.kv_cache_groups
            if isinstance(group.kv_cache_spec, AttentionSpec)
            and layer_name in group.layer_names
        ]
        if len(decoder_specs) != 1:
            raise ValueError(
                "Selected CacheGen INT8 attention layer must belong to exactly "
                f"one decoder AttentionSpec; got {layer_name!r}"
            )

        spec = decoder_specs[0]
        self.cachegen_int8_selected_layer_cache = (
            CacheGenInt8DynamicSelectedLayerCache.allocate(
                max_pages=max_pages,
                tokens_per_page=spec.block_size,
                num_kv_heads=spec.num_kv_heads,
                head_size=spec.head_size,
                device=self.device,
            )
        )


    def _initialize_cachegen_int8_page_adapter(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """Optionally bind CacheGen-style layout to experimental byte pages."""
        if not is_cachegen_int8_page_adapter_enabled():
            self.cachegen_int8_page_adapter = None
            self.cachegen_int8_page_adapters = {}
            self.cachegen_int8_shadow_layer_counters = {}
            self.cachegen_int8_shadow_callback_calls = None
            self.cachegen_int8_shadow_tokens_written = None
            self.cachegen_int8_shadow_sample = None
            return

        if self.hetero_kv_page_pool is None:
            raise AssertionError(
                "CacheGen-style page adapter requires byte-page storage"
            )
        if (
            is_cachegen_int8_shadow_all_layers_enabled()
            and getattr(self, "cachegen_int8_layer_page_pool", None) is None
        ):
            raise AssertionError(
                "CacheGen-style all-layer shadow write requires layer-isolated "
                "byte-page storage"
            )

        decoder_specs = [
            group.kv_cache_spec
            for group in kv_cache_config.kv_cache_groups
            if isinstance(group.kv_cache_spec, AttentionSpec)
        ]
        if not decoder_specs:
            raise ValueError(
                "CacheGen-style page adapter requires at least one "
                "decoder AttentionSpec."
            )

        geometries = {
            (spec.num_kv_heads, spec.head_size)
            for spec in decoder_specs
        }
        if len(geometries) != 1:
            raise ValueError(
                "CacheGen-style page adapter requires all decoder attention "
                "groups to share num_kv_heads and head_size."
            )

        num_kv_heads, head_size = geometries.pop()
        self.cachegen_int8_page_adapters = {}
        layer_page_pool = getattr(
            self,
            "cachegen_int8_layer_page_pool",
            None,
        )
        if layer_page_pool is not None:
            for layer_name in layer_page_pool.layer_names:
                self.cachegen_int8_page_adapters[layer_name] = (
                    CacheGenInt8FixedBytePageAdapter(
                        page_pool=layer_page_pool.layer_page_pool(layer_name),
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                    )
                )
            self.cachegen_int8_page_adapter = None
        else:
            self.cachegen_int8_page_adapter = CacheGenInt8FixedBytePageAdapter(
                page_pool=self.hetero_kv_page_pool,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
            )

        self.cachegen_int8_shadow_layer_counters = {}
        if is_cachegen_int8_shadow_diagnostics_enabled():
            diagnostics_device = self.hetero_kv_page_pool.device
            self.cachegen_int8_shadow_callback_calls = torch.zeros(
                (), dtype=torch.int64, device=diagnostics_device
            )
            self.cachegen_int8_shadow_tokens_written = torch.zeros(
                (), dtype=torch.int64, device=diagnostics_device
            )
            for layer_name in self.cachegen_int8_page_adapters:
                self.cachegen_int8_shadow_layer_counters[layer_name] = (
                    torch.zeros((), dtype=torch.int64, device=diagnostics_device),
                    torch.zeros((), dtype=torch.int64, device=diagnostics_device),
                )
        else:
            self.cachegen_int8_shadow_callback_calls = None
            self.cachegen_int8_shadow_tokens_written = None
        self.cachegen_int8_shadow_sample = None

    def get_cachegen_int8_shadow_diagnostics(
        self,
    ) -> dict[str, float | int] | None:
        """Return explicitly enabled CacheGen shadow-write counters."""
        callback_calls = getattr(
            self,
            "cachegen_int8_shadow_callback_calls",
            None,
        )
        if callback_calls is None:
            return None

        tokens_written = getattr(
            self,
            "cachegen_int8_shadow_tokens_written",
            None,
        )
        assert tokens_written is not None
        diagnostics: dict[str, float | int] = {
            "callback_calls": int(callback_calls.item()),
            "tokens_written": int(tokens_written.item()),
        }
        shadow_sample = getattr(
            self,
            "cachegen_int8_shadow_sample",
            None,
        )
        if shadow_sample is not None:
            diagnostics["sample_count"] = 1
            diagnostics.update(shadow_sample)
        else:
            diagnostics["sample_count"] = 0

        layer_counters = getattr(
            self,
            "cachegen_int8_shadow_layer_counters",
            {},
        )
        if layer_counters:
            diagnostics["layers_with_writes"] = sum(
                int(callback_calls.item() > 0)
                for callback_calls, _ in layer_counters.values()
            )
            diagnostics["layer_counters"] = {
                layer_name: {
                    "callback_calls": int(callback_calls.item()),
                    "tokens_written": int(tokens_written.item()),
                }
                for layer_name, (
                    callback_calls,
                    tokens_written,
                ) in layer_counters.items()
            }
        return diagnostics

    @staticmethod
    def _get_cachegen_int8_decoder_layer_names(
        kv_cache_config: KVCacheConfig,
    ) -> list[str]:
        """Return unique decoder-attention layer names in cache-group order."""
        layer_names: list[str] = []
        for group in kv_cache_config.kv_cache_groups:
            if not isinstance(group.kv_cache_spec, AttentionSpec):
                continue
            layer_names.extend(group.layer_names)

        if len(set(layer_names)) != len(layer_names):
            raise ValueError(
                "CacheGen-style layer page pools require unique decoder "
                "attention layer names."
            )
        return layer_names

    def _initialize_hetero_kv_page_pool(self) -> None:
        """Optionally allocate experimental fixed-byte GPU pages.

        Storage remains separate from standard V1 KV-cache tensors. In
        selected-layer mode this is one common pool. In all-layer shadow mode,
        the same flat allocation is split into disjoint contiguous slices, one
        for each decoder attention layer.
        """
        self.cachegen_int8_layer_page_pool = None
        self.cachegen_int8_pages_per_layer = None
        if not is_hetero_kv_page_pool_enabled():
            self.hetero_kv_page_pool = None
            return

        pages_per_layer = get_hetero_kv_page_pool_pages()
        page_bytes = get_hetero_kv_page_bytes()
        if (
            is_cachegen_int8_shadow_all_layers_enabled()
            and hasattr(self, "kv_cache_config")
        ):
            layer_names = self._get_cachegen_int8_decoder_layer_names(
                self.kv_cache_config
            )
            self.cachegen_int8_layer_page_pool = (
                CacheGenInt8LayerPagePool.allocate(
                    layer_names=layer_names,
                    pages_per_layer=pages_per_layer,
                    page_bytes=page_bytes,
                    device=self.device,
                )
            )
            self.hetero_kv_page_pool = (
                self.cachegen_int8_layer_page_pool.page_pool
            )
            self.cachegen_int8_pages_per_layer = pages_per_layer
            return

        self.hetero_kv_page_pool = self.allocate_fixed_byte_kv_page_pool(
            num_pages=pages_per_layer,
            page_bytes=page_bytes,
            device=self.device,
        )

    def _allocate_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
         """
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            tensor = torch.zeros(kv_cache_tensor.size,
                                 dtype=torch.int8,
                                 device=self.device)
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            layer_names.update(group.layer_names)
        assert layer_names == set(kv_cache_raw_tensors.keys(
        )), "Some layers are not correctly initialized"
        return kv_cache_raw_tensors

    def _attn_group_iterator(self) -> Iterator[AttentionGroup]:
        return itertools.chain.from_iterable(self.attn_groups)

    def _kv_cache_spec_attn_group_iterator(
            self) -> Iterator[tuple[KVCacheSpec, AttentionGroup]]:
        if not self.kv_cache_config.kv_cache_groups:
            return
        for kv_cache_spec_id, attn_groups in enumerate(self.attn_groups):
            for attn_group in attn_groups:
                yield self.kv_cache_config.kv_cache_groups[
                    kv_cache_spec_id].kv_cache_spec, attn_group

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
            correct size but uninitialized shape.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False
        for kv_cache_spec, group in self._kv_cache_spec_attn_group_iterator():
            attn_backend = group.backend
            for layer_name in group.layer_names:
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = (raw_tensor.numel() //
                              kv_cache_spec.page_size_bytes)
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    try:
                        kv_cache_stride_order = \
                            attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(
                            kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(
                            range(len(kv_cache_shape)))
                    # The allocation respects the backend-defined stride order
                    # to ensure the semantic remains consistent for each
                    # backend. We first obtain the generic kv cache shape and
                    # then permute it according to the stride order which could
                    # result in a non-contiguous tensor.
                    kv_cache_shape = tuple(kv_cache_shape[i]
                                           for i in kv_cache_stride_order)
                    # Maintain original KV shape view.
                    inv_order = [
                        kv_cache_stride_order.index(i)
                        for i in range(len(kv_cache_stride_order))
                    ]
                    kv_caches[layer_name] = kv_cache_raw_tensors[
                        layer_name].view(dtype).view(kv_cache_shape).permute(
                            *inv_order)
                elif isinstance(kv_cache_spec, MambaSpec):
                    has_mamba = True
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    state_tensors = []
                    storage_offset_bytes = 0
                    for (shape, dtype) in zip(kv_cache_spec.shapes,
                                              kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        num_element_per_page = (
                            kv_cache_spec.page_size_bytes // dtype_size)
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        target_stride = (num_element_per_page, *stride[1:])
                        assert storage_offset_bytes % dtype_size == 0
                        tensor = torch.as_strided(
                            raw_tensor.view(dtype),
                            size=target_shape,
                            stride=target_stride,
                            storage_offset=storage_offset_bytes // dtype_size,
                        )
                        state_tensors.append(tensor)
                        storage_offset_bytes += stride[0] * dtype_size

                    kv_caches[layer_name] = state_tensors
                else:
                    raise NotImplementedError

        if has_attn and has_mamba:
            self._verify_hybrid_attention_mamba_layout(kv_cache_config,
                                                       kv_cache_raw_tensors)

        return kv_caches

    def _verify_hybrid_attention_mamba_layout(
            self, kv_cache_config: KVCacheConfig,
            kv_cache_raw_tensors: dict[str, torch.Tensor]) -> None:
        """
        Verify that the KV cache memory layout is compatible for
        models with both attention and mamba KV cache groups.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer.
        """

        for kv_cache_spec, group in self._kv_cache_spec_attn_group_iterator():
            for layer_name in group.layer_names:
                raw_tensor = kv_cache_raw_tensors[layer_name]
                num_blocks = (raw_tensor.numel() //
                              kv_cache_spec.page_size_bytes)
                if isinstance(kv_cache_spec, AttentionSpec):

                    kv_cache_shape = group.backend.get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    if kv_cache_shape[0] != num_blocks or kv_cache_shape[
                            1] != 2:
                        raise ValueError(
                            "Hybrid models in V1 require an attention "
                            "backend with kv_cache_shape="
                            "(num_blocks, 2, ...). Please try setting "
                            "VLLM_ATTENTION_BACKEND=FLASHINFER")

    def initialize_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        # Change the memory buffer to the desired shape
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config,
                                                   kv_cache_raw_tensors)

        # Setup `kv_cache_config` and `kv_caches` for models
        # with cross-layer KV sharing
        if self.shared_kv_cache_layers:
            initialize_kv_cache_for_kv_sharing(
                self.shared_kv_cache_layers,
                kv_cache_config.kv_cache_groups,
                kv_caches,
                self.attn_groups,
            )
            attn_layers = get_layers_from_vllm_config(self.vllm_config,
                                                      Attention)
            # Iterate in reversed order and add layers that re-use KV cache
            # e.g. in YOCO-like KV sharing setups (e.g. Gemma3n)
            for layer_name in reversed(attn_layers):
                if layer_name in self.shared_kv_cache_layers:
                    self.kv_sharing_fast_prefill_eligible_layers.add(
                        layer_name)
                else:
                    break

        bind_kv_cache(kv_caches,
                      self.compilation_config.static_forward_context,
                      self.kv_caches)
        return kv_caches

    def _configure_hetero_kv_page_planning(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """Install codec capacities after final decoder geometry is known."""
        if not is_hetero_kv_page_planning_enabled():
            return

        decoder_specs = [
            group.kv_cache_spec
            for group in kv_cache_config.kv_cache_groups
            if isinstance(group.kv_cache_spec, AttentionSpec)
        ]
        if not decoder_specs:
            raise ValueError(
                "Experimental heterogeneous KV page planning requires at "
                "least one decoder AttentionSpec."
            )

        geometries = {
            (spec.num_kv_heads, spec.head_size)
            for spec in decoder_specs
        }
        if len(geometries) != 1:
            raise ValueError(
                "Experimental heterogeneous KV page planning requires all "
                "decoder attention groups to share num_kv_heads and head_size."
            )

        num_kv_heads, head_size = geometries.pop()
        self.input_batch.configure_hetero_kv_page_planning(
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        self.kv_cache_config = kv_cache_config
        self.may_reinitialize_input_batch(kv_cache_config)
        self._configure_hetero_kv_page_planning(kv_cache_config)
        self._initialize_cachegen_int8_selected_layer_cache(kv_cache_config)
        self._initialize_hetero_kv_page_pool()
        self._initialize_cachegen_int8_page_adapter(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if (kv_tgt_layer :=
                    attn_module.kv_sharing_target_layer_name) is not None:
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue

            # TODO: Support other attention modules, e.g., cross-attention
            # TODO(lucas): move the attention specs into the model layers like
            # the attention backends
            if attn_module.attn_type == AttentionType.DECODER:
                if attn_module.sliding_window is not None:
                    kv_cache_spec[layer_name] = SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        sliding_window=attn_module.sliding_window,
                        use_mla=use_mla)
                elif self.attention_chunk_size is not None \
                        and isinstance(attn_module, ChunkedLocalAttention):
                    kv_cache_spec[layer_name] = ChunkedLocalAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        attention_chunk_size=self.attention_chunk_size,
                        use_mla=use_mla)
                else:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        use_mla=use_mla)
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        mamba_layers = get_layers_from_vllm_config(self.vllm_config, MambaBase)
        if len(mamba_layers) > 0:
            if self.vllm_config.speculative_config is not None:
                raise NotImplementedError(
                    "Mamba with speculative decoding is not supported yet.")
            if self.vllm_config.cache_config.enable_prefix_caching:
                raise NotImplementedError(
                    "Prefix caching is not supported for Mamba yet.")
            max_model_len = self.vllm_config.model_config.max_model_len

            page_size_padded = (
                self.vllm_config.cache_config.mamba_page_size_padded)

            # Set block_size to max_model_len, so that mamba model will always
            # have only one block in the KV cache.
            for layer_name, mamba_module in mamba_layers.items():
                kv_cache_spec[layer_name] = MambaSpec(
                    shapes=mamba_module.get_state_shape(),
                    dtypes=mamba_module.get_state_dtype(),
                    block_size=max_model_len,
                    page_size_padded=page_size_padded,
                    mamba_type=mamba_module.mamba_type)

        return kv_cache_spec

    def _build_encoder_only_attn_metadata(
            self, scheduler_output: "SchedulerOutput") -> \
                dict[str, tuple[CommonAttentionMetadata, Any]]:
        """Prepare encoder attention metadata for encoder-only models.

        Args:
            scheduler_output: Scheduler output

        Returns:
            dict[str, Any]: Encoder attention metadata
        """
        num_reqs = self.input_batch.num_reqs
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        max_num_scheduled_tokens = max(tokens)

        dummy_block_table = torch.zeros((num_reqs, 1),
                                        dtype=torch.int32,
                                        device=self.device)
        dummy_slot_mapping = torch.zeros((total_num_scheduled_tokens, ),
                                         dtype=torch.int32,
                                         device=self.device)

        group_metadata = dict[str, tuple[CommonAttentionMetadata, Any]]()

        for attn_group_list in self.attn_groups:

            assert len(attn_group_list) == 1
            attn_group = attn_group_list[0]

            # Use the first attention metadata builder
            # to create encoder attention metadata
            builder = attn_group.metadata_builder

            common_metadata = CommonAttentionMetadata(
                query_start_loc=self.query_start_loc[:num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc_cpu[:num_reqs + 1],
                seq_lens=self.seq_lens[:num_reqs],
                seq_lens_cpu=self.seq_lens_cpu[:num_reqs],
                num_computed_tokens_cpu=self.input_batch.
                num_computed_tokens_cpu_tensor[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                block_table_tensor=dummy_block_table,
                slot_mapping=dummy_slot_mapping,
                quantizer_id=torch.zeros(
                    num_reqs, dtype=torch.int32, device=self.device),
                tokens_per_page=torch.full(
                    (num_reqs,),
                    self.input_batch.block_table[0].block_size,
                    dtype=torch.int32,
                    device=self.device),
                quantized_page_ids=torch.zeros(
                    total_num_scheduled_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                quantized_page_offsets=torch.zeros(
                    total_num_scheduled_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                causal=False,
            )

            metadata = builder.build(
                common_prefix_len=0,  # No cascade for encoder
                common_attn_metadata=common_metadata,
            )

            for layer_name in attn_group.layer_names:
                group_metadata[layer_name] = (common_metadata, metadata)

        return group_metadata
