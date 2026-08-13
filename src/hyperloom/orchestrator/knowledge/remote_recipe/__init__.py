# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KB Store Recipe read/write plus a config-only legacy warm-replay adapter."""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path
from typing import Any

from ._vendor.kb_store_client import KnowledgeSections, SectionContent
from .client import (
    KBStoreClient,
    KBStoreError,
    RemoteRecipeClient,
    RemoteRecipeConfigurationError,
)
from .models import (
    RemoteRecipeValidationError,
    RemoteWriteResult,
    validate_relative_path,
)
from .values import (
    KERNEL_AGENT_METRIC,
    build_kernel_agent_knowledge,
    build_remote_knowledge,
    convert_v1_recipe_to_knowledge,
    envelope_to_v1_recipe,
    has_new_keep,
    kernel_agent_canonical_id,
)

log = logging.getLogger(__name__)


def read_remote_recipe(
    canonical_id: str,
    destination: str | Path,
    *,
    client: RemoteRecipeClient | None = None,
) -> dict[str, Any] | None:
    """Download the direct best record as flattened recipe.json + files/."""
    resolved = client or RemoteRecipeClient.from_env_optional()
    if resolved is None:
        return None
    return resolved.read(canonical_id, destination)


# Kept as a standalone API compatibility alias.
read_remote_champion = read_remote_recipe


def write_kernel_agent_kb(
    state: Any,
    kernel_canonical_id: str,
    session_id: str,
    *,
    client: RemoteRecipeClient | None = None,
) -> RemoteWriteResult:
    """Write Hyperloom's independent kernel-agent KB record (``kernel:`` scheme).

    Unlike the recipe record, this write is NOT gated by end-to-end serving
    throughput: it stores this session's kernel optimizations (gemm/fusion/
    rewrite) into their own KB Store record and keeps whichever beats what the
    kernel-agent KB already holds, scored by the kernel's own gain. Overlap with
    KernelForge's ``kernel:`` records is by design and not consulted here.
    """
    resolved = client or RemoteRecipeClient.from_env_optional()
    if resolved is None:
        return RemoteWriteResult("disabled", "KB_STORE_URL/TOKEN not configured")
    with tempfile.TemporaryDirectory(prefix="hyperloom-kernel-agent-") as temporary:
        files_dir = Path(temporary) / "files"
        bundle, score = build_kernel_agent_knowledge(
            state, files_dir, sections=KnowledgeSections.from_env()
        )
        value = bundle.knowledge.get("value") or {}
        has_opt = bool(
            (value.get("gemm") or {}).get("optimizations")
            or (value.get("fusion") or {}).get("items")
            or (value.get("rewrite") or {}).get("items")
        )
        if not has_opt:
            return RemoteWriteResult(
                "skipped", "no_kernel_optimization", kernel_canonical_id, session_id
            )
        # Ensure a first kernel optimization always lands even if its gain field
        # is missing; real KEEPs carry a positive gain that drives keep-if-better.
        score_value = score if score > 0 else 1e-6
        return resolved.write_if_better(
            kernel_canonical_id,
            session_id,
            bundle,
            optimized_throughput=score_value,
            files_dir=files_dir,
            metric=KERNEL_AGENT_METRIC,
        )


def write_final_remote_recipe(
    state: Any,
    canonical_id: str,
    session_id: str,
    *,
    client: RemoteRecipeClient | None = None,
) -> RemoteWriteResult:
    """Build and conditionally write one final E2E/CLOSE session record."""
    resolved = client or RemoteRecipeClient.from_env_optional()
    if resolved is None:
        return RemoteWriteResult("disabled", "KB_STORE_URL/TOKEN not configured")
    if not has_new_keep(state):
        return RemoteWriteResult("skipped", "no_new_keep_or_pure_warm_replay", canonical_id, session_id)
    current_best = getattr(state, "current_best", {}) or {}
    try:
        throughput = float(current_best.get("tput") or 0.0) if isinstance(current_best, dict) else 0.0
    except (TypeError, ValueError):
        throughput = 0.0
    if not math.isfinite(throughput):
        return RemoteWriteResult(
            "skipped",
            "nonfinite_optimized_throughput",
            canonical_id,
            session_id,
        )
    if throughput <= 0:
        return RemoteWriteResult("skipped", "missing_optimized_throughput", canonical_id, session_id)
    with tempfile.TemporaryDirectory(prefix="hyperloom-remote-recipe-") as temporary:
        files_dir = Path(temporary) / "files"
        bundle = build_remote_knowledge(
            state, files_dir, sections=KnowledgeSections.from_env()
        )
        return resolved.write_if_better(
            canonical_id,
            session_id,
            bundle,
            optimized_throughput=throughput,
            files_dir=files_dir,
        )


class HyperloomRemoteKB:
    """Public facade for Hyperloom's remote inference knowledge."""

    def __init__(self, client: RemoteRecipeClient) -> None:
        self._client = client

    @classmethod
    def from_env(cls) -> "HyperloomRemoteKB":
        """Build a configured facade, requiring both KB Store variables."""
        client = RemoteRecipeClient.from_env_optional()
        if client is None:
            raise RemoteRecipeConfigurationError(
                "KB_STORE_URL and KB_STORE_TOKEN are required for HyperloomRemoteKB"
            )
        return cls(client)

    def read(
        self,
        identity: str,
        destination: str | Path,
    ) -> dict[str, Any] | None:
        """Download the direct best record for an inference canonical id."""
        return read_remote_recipe(identity, destination, client=self._client)

    def write(
        self,
        identity: str,
        state: Any,
        session_id: str | None = None,
    ) -> RemoteWriteResult:
        """Write final E2E knowledge, resolving the session id from state."""
        resolved_session_id = session_id
        if resolved_session_id is None:
            resolved_session_id = (
                str(getattr(state, "recipe_kb_session_id", "") or "").strip()
                or str(getattr(state, "session_id", "") or "").strip()
            )
        resolved_session_id = str(resolved_session_id or "").strip()
        if not resolved_session_id:
            raise RemoteRecipeValidationError(
                "session_id is required; set state.recipe_kb_session_id or state.session_id"
            )
        return write_final_remote_recipe(
            state,
            identity,
            resolved_session_id,
            client=self._client,
        )


class RemoteWarmRecipeAdapter:
    """Read-only RecipeKB-shaped adapter for the unchanged T0 replay pipeline."""

    enabled = True
    mode = "remote"
    backend_name = "kb-store"

    def __init__(
        self,
        remote_kb: HyperloomRemoteKB,
        destination: str | Path,
    ) -> None:
        self._remote_kb = remote_kb
        self._destination = Path(destination)
        self._cache: dict[str, dict[str, Any] | None] = {}
        self._search_notice_emitted = False

    def _read(self, canonical_id: str) -> dict[str, Any] | None:
        if canonical_id not in self._cache:
            document = self._remote_kb.read(canonical_id, self._destination)
            if document is None:
                self._cache[canonical_id] = None
            else:
                row = envelope_to_v1_recipe(document)
                # Schema 3 restores the exact flat timeline as required
                # ``prs_tested`` rows. Missing artifacts remain represented so
                # PRELUDE fails closed instead of silently shortening the set.
                if int(row.get("knowledge_schema_version") or 0) >= 3:
                    knowledge = document.get("knowledge")
                    value = (
                        (knowledge or {}).get("value")
                        if isinstance(knowledge, dict)
                        else document.get("value")
                    )
                    timeline = (
                        (value or {}).get("patch_timeline")
                        if isinstance(value, dict)
                        else None
                    )
                    gain = float(row.get("validated_gain_pct") or 0.0)
                    replayable: list[dict[str, Any]] = []
                    if isinstance(timeline, list):
                        files_root = self._destination / "files"
                        if files_root.is_symlink():
                            raise RemoteRecipeValidationError(
                                "schema-v3 files root must not be a symlink"
                            )
                        resolved_root = files_root.resolve()
                        for index, item in enumerate(timeline):
                            ref = validate_relative_path(
                                str(item or "").strip()
                            )
                            source = files_root / ref
                            cursor = files_root
                            for part in Path(ref).parts:
                                cursor = cursor / part
                                if cursor.is_symlink():
                                    raise RemoteRecipeValidationError(
                                        "schema-v3 timeline ref resolves through "
                                        f"a symlink: {ref!r}"
                                    )
                            try:
                                source.resolve().relative_to(resolved_root)
                            except ValueError as exc:
                                raise RemoteRecipeValidationError(
                                    "schema-v3 timeline ref escapes files root: "
                                    f"{ref!r}"
                                ) from exc
                            patch_content = ""
                            if (
                                ref
                                and source.is_file()
                                and not source.is_symlink()
                                and source.stat().st_size <= 50_000
                            ):
                                patch_content = source.read_text(
                                        encoding="utf-8",
                                        errors="replace",
                                    )
                            replayable.append(
                                {
                                    "outcome": "KEEP",
                                    "patch_file": ref,
                                    "patch_ref": str(source),
                                    "patch_content": patch_content,
                                    "measured_gain_pct": gain if gain > 0 else 1e-6,
                                    "required": True,
                                    "timeline_index": index,
                                }
                            )
                    row["prs_tested"] = replayable
                    row["patch_timeline"] = [
                        str(item or "") for item in (timeline or [])
                    ]
                    row["required_patch_timeline"] = True
                self._cache[canonical_id] = row
        return self._cache[canonical_id]

    def get_authoritative_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Return the exact best record projected into the legacy Recipe shape."""
        del version
        return self._read(canonical_id)

    def get_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
        prefer: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Return the cached exact best record; relative-tier search is deferred."""
        del version, prefer
        return self._read(canonical_id)

    def search(self, **_kwargs: Any) -> list[dict[str, Any]]:
        """Return no cross-identity donors in the config-only migration phase."""
        if not self._search_notice_emitted:
            log.info(
                "Remote Recipe KB cross-identity search is unsupported; "
                "warm start is limited to the direct identity best record"
            )
            self._search_notice_emitted = True
        return []

    def put_recipe(self, **kwargs: Any) -> dict[str, Any]:
        """No-op the legacy T0 anchor write; CLOSE owns remote publication."""
        log.debug("Remote Recipe KB T0 put_recipe is a no-op; CLOSE owns publication")
        return dict(kwargs)

    def close(self) -> None:
        """The wrapped blocking client has no explicit lifecycle."""


__all__ = [
    "HyperloomRemoteKB",
    "KBStoreClient",
    "KBStoreError",
    "KnowledgeSections",
    "RemoteRecipeClient",
    "RemoteRecipeConfigurationError",
    "RemoteWarmRecipeAdapter",
    "SectionContent",
    "KERNEL_AGENT_METRIC",
    "build_kernel_agent_knowledge",
    "build_remote_knowledge",
    "convert_v1_recipe_to_knowledge",
    "envelope_to_v1_recipe",
    "has_new_keep",
    "kernel_agent_canonical_id",
    "read_remote_champion",
    "read_remote_recipe",
    "write_final_remote_recipe",
    "write_kernel_agent_kb",
]
