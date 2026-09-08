"""Shared canonical taxonomy and conservative legacy relation mappings."""
from typing import Dict, Tuple

# Canonical taxonomies — single source of truth for extraction prompt AND runtime validation
CANONICAL_FACT_CATEGORIES = frozenset({
    "infra", "hardware", "software", "contacts", "family", "health", "fitness",
    "finance", "insurance", "travel", "home", "media", "music", "work", "dev",
    "preferences", "communication", "cloud", "security", "architecture", "workflow", "general"
})

CANONICAL_LEARNING_CATEGORIES = frozenset({
    "workflow", "communication", "finance", "health", "shopping", "travel",
    "hardware", "safety", "architecture", "security", "automation", "preferences", "insurance", "general"
})

CANONICAL_EPISODE_TOPICS = frozenset({
    "family", "health", "travel", "finance", "home", "dev", "infra",
    "insurance", "music", "work", "realestate", "trading", "general"
})

CANONICAL_RELATIONS = frozenset({
    "hosted_on", "runs_on", "depends_on", "part_of", "member_of",
    "owned_by", "managed_by", "monitors", "treats", "prescribed_for",
    "insured_by", "finances", "communicates_via", "located_at", "uses",
    "stores", "connects_to", "related_to", "maintains", "created_by",
    "delivers_to", "advises", "works_at", "lives_at", "travels_to",
    "subscribed_to", "prescribes"
})


# Common LLM-generated category variants → canonical mapping
_CATEGORY_ALIASES = {
    "contact": "contacts", "kontakte": "contacts",
    "infrastructure": "infra", "system_architecture": "infra", "system_config": "infra",
    "admin": "infra", "config": "infra",
    "pref": "preferences", "preference": "preferences", "rule": "preferences", "user": "preferences",
    "pension": "finance", "trading": "finance", "stweg": "home",
    "gear": "hardware", "tesla": "hardware",
    "devsecops": "dev", "dev.cron": "automation",
    "heuristics": "general", "ai_tools": "software", "ai": "dev",
    "ui_ux": "architecture", "network": "infra",
    "realestate": "home", "calendar": "general",
}


def _normalize_category(category: str, allowed: frozenset) -> str:
    """Normalize a category string to the closest canonical category."""
    if not category:
        return "general"
    cat = category.strip().lower()
    if cat in allowed:
        return cat
    if cat in _CATEGORY_ALIASES:
        alias = _CATEGORY_ALIASES[cat]
        if alias in allowed:
            return alias
    return "general"


RELATION_MAPPINGS: Dict[str, Tuple[str, bool]] = {
    # Direct mappings
    "runs_in": ("runs_on", False),
    "executed_on": ("runs_on", False),
    "deployed_on": ("runs_on", False),
    "hosted_at": ("hosted_on", False),
    "installed_on": ("runs_on", False),
    "belongs_to": ("part_of", False),
    "is_member_of": ("member_of", False),
    "monitored_by": ("monitors", True),       # Inverted: A monitored_by B -> B monitors A
    "hosts": ("hosted_on", True),              # Inverted: A hosts B -> B hosted_on A
    "contains": ("part_of", True),             # Inverted: A contains B -> B part_of A
    "includes": ("part_of", True),             # Inverted: A includes B -> B part_of A
    "has_part": ("part_of", True),             # Inverted: A has_part B -> B part_of A
    "administers": ("managed_by", True),       # Inverted: A administers B -> B managed_by A
    "manages": ("managed_by", True),           # Inverted: A manages B -> B managed_by A
    "owns": ("owned_by", True),                # Inverted: A owns B -> B owned_by A

    "integrates_with": ("connects_to", False),
    "interfaces_with": ("connects_to", False),

    "synced_with": ("connects_to", False),
    "syncs_to": ("connects_to", False),
    "associates_with": ("related_to", False),
    "associated_with": ("related_to", False),
    "references": ("related_to", False),
    "subscribed": ("subscribed_to", False),
    "consults": ("advises", True),             # A consults B -> B advises A
    "treated_by": ("treats", True),            # A treated_by B -> B treats A
    "prescribed_by": ("prescribes", True), # Med prescribed_by Doc -> Doc prescribes Med
    "insured_at": ("insured_by", False),
    "stored_in": ("stores", True),             # Item stored_in Location -> Location stores Item
    "resides_in": ("located_at", False),
    "situated_at": ("located_at", False),
}

def map_relation(source: str, target: str, rel: str) -> Tuple[str, str, str]:
    """
    Map legacy relation string to canonical relation.
    Handles lowercase trimming, semantic mapping, and directional inversion.
    """
    cleaned_rel = rel.strip().lower().replace(" ", "_").replace("-", "_")
    
    if cleaned_rel in CANONICAL_RELATIONS:
        return source, target, cleaned_rel

    if cleaned_rel in RELATION_MAPPINGS:
        canonical_rel, inverted = RELATION_MAPPINGS[cleaned_rel]
        if inverted:
            return target, source, canonical_rel
        return source, target, canonical_rel

    raise ValueError(f"Unknown or ambiguous relation: {rel!r}")


CANONICAL_EPISODE_STATUSES = frozenset({'active','cooling','historic','resolved'})

def validate_category(value, allowed):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Category must be nonempty text')
    cleaned = value.strip().lower()
    canonical = cleaned if cleaned in allowed else _CATEGORY_ALIASES.get(cleaned)
    if canonical not in allowed:
        raise ValueError(f'Unknown category: {value!r}')
    return canonical

def require_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be nonempty text')
    return value.strip()
