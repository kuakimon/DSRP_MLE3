# Estrategia de Git

Este proyecto (detección de fraude con CatBoost) usa **GitHub Flow** con dos ramas de larga vida.

| Rama | Rol | Reglas |
|---|---|---|
| `main` | Código estable, listo para entrega | Solo recibe merges vía Pull Request desde `development`. No se hace commit directo. |
| `development` | Integración | Las ramas de feature nacen aquí y regresan aquí vía PR. |

## Flujo de trabajo

1. Crear una rama desde `development`:
   ```bash
   git checkout development && git pull
   git checkout -b feature/nombre-corto
   ```
2. Hacer commits pequeños y descriptivos con prefijos convencionales:
   `feat:` (funcionalidad), `fix:` (corrección), `docs:` (documentación), `refactor:`, `chore:`.
3. Abrir un Pull Request `feature/nombre-corto` → `development` explicando qué cambia y por qué.
4. Revisar y mergear con **merge commit** (sin squash), para conservar el historial.
5. Cuando `development` esté estable y completa para una entrega, abrir un PR `development` → `main`.
6. Al mergear a `main`, crear un **tag/release** con versionado semántico.

### Nombres de rama sugeridos

- `feature/automl`, `feature/optuna-f1`, `feature/monitoreo-evidently`
- `fix/scoring-nan-destTrxAmount`
- `docs/readme-scorecard`

## Versionado

Versionado semántico `vMAYOR.MENOR.PARCHE`:

- **MAYOR:** una nueva entrega de curso (`v1.0.0` Curso II, `v2.0.0` esta entrega) o un cambio incompatible del modelo/pipeline.
- **MENOR:** nueva funcionalidad compatible (p. ej. un nuevo estudio de Optuna o un nuevo reporte de monitoreo).
- **PARCHE:** correcciones y documentación.

## Qué se versiona y qué no

- **Sí:** notebooks, `notebooks/sbp_modelo.py`, `src/`, `docs/`, `optuna/*.db` y `*.csv`, `reports/monitoreo/`, `pyproject.toml` y `uv.lock`.
- **No** (ver `.gitignore`): `data/raw`, `data/processed`, `data/logs`, `models/*`, `mlruns/*`, `.env`, `__pycache__`.
- Las carpetas ignoradas se conservan con `.gitkeep` (`data/`, `models/`, `mlruns/`).
- Antes de commitear notebooks, conviene revisar que las salidas no incluyan rutas o datos sensibles.

## Buenas prácticas

- Al menos un Pull Request cerrado exitosamente antes de la entrega final.
- Mantener `uv.lock` sincronizado con `pyproject.toml` en el mismo commit que cambie dependencias.
- Documentar en el PR cualquier cambio de modelo (hiperparámetros, variables, umbral) y actualizar el scorecard del [README](../README.md).
- Sin commits adicionales después de la fecha de entrega.
