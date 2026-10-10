"""Isolated LLM knowledge-distillation components for MAPPO."""

from .temporal_view import (
    TEMPORAL_SEMANTIC_VIEW_V1,
    serialize_temporal_semantic_view,
)
from .trajectory import (
    ACTOR_OBSERVATION_TRAJECTORY_V1,
    ActorObservationTrajectory,
)
from .dataset import (
    ASTAR_DIAGNOSTIC_DISTRIBUTION,
    MAIN_BASELINE_DISTRIBUTION,
    DistributionTrajectory,
    TemporalWindow,
    build_partition_windows,
    deterministic_stratified_sample,
    split_complete_episodes,
)
from .pretraining import (
    ACTOR_ARCHITECTURE_V1,
    HIDDEN_DIM,
    OBSERVATION_DIM,
    PRETRAINING_CHECKPOINT_V1,
    PRETRAINING_EXPORT_V1,
    PRETRAINING_RUN_V1,
    REQUIRED_PRETRAINING_IDENTITY_FIELDS,
    SEMANTIC_DIM,
    SEMANTIC_DIMENSIONS,
    SEMANTIC_ONTOLOGY_V1,
    PretrainingBatch,
    SemanticPretrainingModel,
    masked_semantic_loss,
    validate_pretraining_identity,
    validate_prediction_tensor,
)
from .pretraining_checkpoint import (
    load_pretraining_checkpoint,
    save_pretraining_checkpoint,
)
from .pretraining_run import (
    PRETRAINING_RUNNER_V1,
    PretrainingRunConfig,
    load_pretraining_run_config,
    run_semantic_pretraining,
)
from .representation_export import (
    export_actor_representation,
    import_actor_representation,
)
from .representation_embeddings import (
    EMBEDDING_INPUT_VARIANTS,
    FrozenRecurrentEmbeddingExtractor,
    RecurrentEmbeddings,
    embedding_input_variant,
)
from .representation_diagnostics import (
    DIAGNOSTIC_RUN_V1,
    REPRESENTATION_VARIANTS,
    TEST_ACCESS_LEDGER_V1,
    TEST_ACCESS_RECEIPT_V1,
    TestAccessGrant,
    TestAccessLedger,
    TestEvaluationResult,
    create_diagnostic_run_identity,
    validate_diagnostic_run_identity,
)

__all__ = [
    "ACTOR_ARCHITECTURE_V1",
    "ACTOR_OBSERVATION_TRAJECTORY_V1",
    "ActorObservationTrajectory",
    "ASTAR_DIAGNOSTIC_DISTRIBUTION",
    "DIAGNOSTIC_RUN_V1",
    "EMBEDDING_INPUT_VARIANTS",
    "FrozenRecurrentEmbeddingExtractor",
    "HIDDEN_DIM",
    "MAIN_BASELINE_DISTRIBUTION",
    "OBSERVATION_DIM",
    "PRETRAINING_CHECKPOINT_V1",
    "PRETRAINING_EXPORT_V1",
    "PRETRAINING_RUN_V1",
    "PRETRAINING_RUNNER_V1",
    "PretrainingBatch",
    "PretrainingRunConfig",
    "REQUIRED_PRETRAINING_IDENTITY_FIELDS",
    "REPRESENTATION_VARIANTS",
    "RecurrentEmbeddings",
    "SEMANTIC_DIM",
    "SEMANTIC_DIMENSIONS",
    "SEMANTIC_ONTOLOGY_V1",
    "SemanticPretrainingModel",
    "DistributionTrajectory",
    "TemporalWindow",
    "TEMPORAL_SEMANTIC_VIEW_V1",
    "TEST_ACCESS_LEDGER_V1",
    "TEST_ACCESS_RECEIPT_V1",
    "TestAccessGrant",
    "TestAccessLedger",
    "TestEvaluationResult",
    "build_partition_windows",
    "create_diagnostic_run_identity",
    "deterministic_stratified_sample",
    "embedding_input_variant",
    "masked_semantic_loss",
    "load_pretraining_run_config",
    "export_actor_representation",
    "import_actor_representation",
    "load_pretraining_checkpoint",
    "save_pretraining_checkpoint",
    "run_semantic_pretraining",
    "serialize_temporal_semantic_view",
    "split_complete_episodes",
    "validate_prediction_tensor",
    "validate_pretraining_identity",
    "validate_diagnostic_run_identity",
]
