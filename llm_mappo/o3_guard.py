"""Central fail-closed guard for evaluation-only O3 topology provenance."""

from collections.abc import Mapping


O3_ENVIRONMENT_IDS = (
    "llm-mappo-o3-unseen-narrow-passage-5ag-v2",
    "llm-mappo-o3-unseen-central-cross-5ag-v2",
)

O3_SOURCE_SHA256 = (
    "0120b104d61bb964baee39e21fcc95c2422ee67894b9c9b9a5c5de60edaff985",
    "4fc92da618def49e218abbcfaa46c118a65b4830d547dabe81d5b4792b333a14",
)

O3_EFFECTIVE_LAYOUT_HASHES = (
    "978751b66589003b10e493e1ba31590f732a4e2cd21b4896548d90d2e81d0132",
    "0f9b25f1ddfe42549d97b7426d5d60d0b73ba47833ab9d8d4f7c000a9f81ce8c",
)

O3_PROVENANCE_TOKENS = frozenset(
    O3_ENVIRONMENT_IDS
    + O3_SOURCE_SHA256
    + O3_EFFECTIVE_LAYOUT_HASHES
    + (
        "unseen_narrow_passage_v2.txt",
        "unseen_central_cross_v2.txt",
        "layouts/o3/unseen_narrow_passage_v2.txt",
        "layouts/o3/unseen_central_cross_v2.txt",
    )
)


def reject_o3_environment(environment_id: str, *, context: str) -> None:
    """Reject a held-out topology at a training or dataset entry point."""
    if environment_id in O3_ENVIRONMENT_IDS:
        raise ValueError(
            f"{context} cannot use evaluation-only O3 environment "
            f"{environment_id}."
        )


def reject_o3_provenance(payload, *, context: str) -> None:
    """Recursively reject exact O3 identifiers, hashes, or resource names."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            reject_o3_provenance(key, context=context)
            reject_o3_provenance(value, context=context)
        return
    if isinstance(payload, (list, tuple, set, frozenset)):
        for value in payload:
            reject_o3_provenance(value, context=context)
        return
    if isinstance(payload, str):
        normalized = payload.replace("\\", "/")
        if payload in O3_PROVENANCE_TOKENS or normalized in O3_PROVENANCE_TOKENS:
            raise ValueError(f"{context} contains forbidden O3 provenance.")
