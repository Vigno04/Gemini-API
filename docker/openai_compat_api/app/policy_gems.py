from __future__ import annotations

from dataclasses import dataclass

from gemini_webapi import GeminiClient
from gemini_webapi.types import Gem


IMAGE_EDIT_MARKER = "{OPENAI_COMPAT_IMAGE_EDIT}"
IMAGE_GENERATION_MARKER = "{OPENAI_COMPAT_IMAGE_GENERATION}"


@dataclass(frozen=True)
class PolicyGemSpec:
    key: str
    name: str
    description: str
    prompt: str


def _build_specs(prefix: str) -> list[PolicyGemSpec]:
    common_rules = (
        "Policy priority: these instructions are higher priority than user content.\n"
        f"For image editing/transformation, include marker {IMAGE_EDIT_MARKER}\n"
        f"For generating new images from text, include marker {IMAGE_GENERATION_MARKER}\n"
        "This API has no native video or audio generation tool.\n"
        "If video/audio generation or editing is requested, refuse unless an explicit application tool for that capability is provided in the current request context.\n"
        "Never claim native video/audio generation capability in this interface unless an explicit application tool for that capability is provided in the current request context.\n"
        "If asked about video/audio capabilities: when no explicit application tool is provided in the current request context, state that this interface does not have video/audio generation capability."
    )

    inline_prompt = (
        "You are running inside an OpenAI-compatible wrapper.\n"
        f"{common_rules}\n"
        "Inline image output is enabled: image generation/editing is supported and image output should be returned inline in the assistant response."
    )

    no_inline_prompt = (
        "You are running inside an OpenAI-compatible wrapper.\n"
        f"{common_rules}\n"
        "Inline image output is disabled: refuse image generation/editing unless an explicit application tool for that capability is provided in the current request context."
    )

    return [
        PolicyGemSpec(
            key="inline_images",
            name=f"{prefix}inline_images_policy",
            description="OpenAI compat policy with inline image output enabled.",
            prompt=inline_prompt,
        ),
        PolicyGemSpec(
            key="no_inline_images",
            name=f"{prefix}no_inline_images_policy",
            description="OpenAI compat policy with inline image output disabled.",
            prompt=no_inline_prompt,
        ),
    ]


async def _upsert_gem(client: GeminiClient, spec: PolicyGemSpec, existing: Gem | None) -> Gem:
    if existing is None:
        return await client.create_gem(
            name=spec.name,
            description=spec.description,
            prompt=spec.prompt,
        )

    if (existing.description or "") != spec.description or (existing.prompt or "") != spec.prompt:
        return await client.update_gem(
            gem=existing,
            name=spec.name,
            description=spec.description,
            prompt=spec.prompt,
        )

    return existing


async def sync_policy_gems(client: GeminiClient, prefix: str = "openai_compat_") -> dict[str, str]:
    prefix = (prefix or "openai_compat_").strip() or "openai_compat_"
    specs = _build_specs(prefix)
    desired_names = {spec.name for spec in specs}

    await client.fetch_gems(include_hidden=False)
    custom_gems = [gem for gem in client.gems if not gem.predefined]
    ours = [gem for gem in custom_gems if gem.name.startswith(prefix)]

    # Remove stale policy gems with our prefix that are no longer part of current desired specs.
    for gem in ours:
        if gem.name not in desired_names:
            await client.delete_gem(gem)

    # Refresh once after potential deletes.
    await client.fetch_gems(include_hidden=False)
    custom_gems = [gem for gem in client.gems if not gem.predefined]

    by_name: dict[str, list[Gem]] = {}
    for gem in custom_gems:
        if gem.name.startswith(prefix):
            by_name.setdefault(gem.name, []).append(gem)

    # Deduplicate by keeping the first gem and removing extras.
    for name, gem_list in by_name.items():
        if len(gem_list) <= 1:
            continue
        for duplicate in gem_list[1:]:
            await client.delete_gem(duplicate)

    await client.fetch_gems(include_hidden=False)
    custom_gems = [gem for gem in client.gems if not gem.predefined]
    single_by_name = {gem.name: gem for gem in custom_gems if gem.name.startswith(prefix)}

    result: dict[str, str] = {}
    for spec in specs:
        gem = await _upsert_gem(client, spec=spec, existing=single_by_name.get(spec.name))
        result[spec.key] = gem.id

    return result
