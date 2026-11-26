# Supported models

vLLM-GR runs OneRec recommendation workloads on CUDA and Ascend NPU. CUDA
also supports text decoders with eligible native recurrent and paged attention
cache groups. See the [Hybrid CUDA cache requirements](../design/feature/beam_batch_execution.md#hybrid-cuda-decode).
Use full-vocabulary Beam search or supply a constraint table to limit results
to your catalog.

## Model list

| Checkpoint | Task and output | CUDA | Ascend NPU |
| --- | --- | --- | --- |
| [OneRec-1.7B](https://huggingface.co/OpenOneRec/OneRec-1.7B) | Full-vocabulary Beam generation, ranked token sequences | [CUDA quickstart](../getting_started/quickstart.md#cuda) | [Ascend quickstart](../getting_started/quickstart.md#ascend) |
| OneRec-1.7B with a constraint table | Catalog-constrained recommendation, ranked semantic IDs (SIDs) | [Offline and HTTP examples](../examples/README.md) | [Ascend Beam guide](../user_guide/beam_search_npu.md) |
| [Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) | Hybrid Beam validation with KDA and MLA, including three-token catalog constraints | [Validated on ModelRunner V1 and V2](../../examples/offline_inference/beam_search/README.md#run-hybrid-beam-on-cuda) | Unavailable |

## Implementation and compatibility

The offline entry point is `GRLLM.beam_search()`. It uses the GR Beam worker
and scheduler. The [feature matrix](../user_guide/feature_compatibility.md)
covers differences in graph execution, candidate widths and constraint tables
across devices.
