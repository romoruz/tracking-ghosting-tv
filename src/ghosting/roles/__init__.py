"""Inferencia de roles por geometría (sin metadatos de formación privados)."""

from .infer import (
    infer_roles_by_geometry,
    infer_all_roles,
    role_at,
    ROLE_COLORS,
    ROLE_NAMES,
    ROLE_GROUP,
    ROLE_LABEL,
    ROLE_ORDER,
)

__all__ = [
    "infer_roles_by_geometry",
    "infer_all_roles",
    "role_at",
    "ROLE_COLORS",
    "ROLE_NAMES",
    "ROLE_GROUP",
    "ROLE_LABEL",
    "ROLE_ORDER",
]
