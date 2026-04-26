# PID-RL PoC — Plan Completo

## Goal

Entrenar un LLM chico (Qwen 2.5 1.5B Instruct) con GRPO para que proponga ajustes de `Kp, Ki` al PID controller on-chain de GRIT, reaccionando a caídas/subidas de BTC antes de que el depeg materialice, respetando bounds del `ParameterGuard` y produciendo output JSON estructurado con rationale auditable.

## Stack

- **Model**: Qwen 2.5 1.5B Instruct (fits in Colab T4 free tier)
- **Training**: GRPO via Unsloth (fork de su notebook oficial)
- **Env**: Custom numpy sim (NOT cadCAD — too slow for RL)
- **Data**: Synthetic scenarios (training) + real BTC crashes via yfinance (validation)

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop (Colab + Unsloth GRPO)      │
│                                                              │
│  scenarios.py ──► prompt.py ──► Qwen 2.5 1.5B ──► JSON out   │
│       │                                              │        │
│       ▼                                         parse_output  │
│    sim.py (PID + mkt + collateral model) ◄───── action       │
│       │                                                       │
│       ▼                                                       │
│  reward.py ──► GRPO update ──► model weights                 │
└─────────────────────────────────────────────────────────────┘

Validation:
  real_btc_data.py ──► same pipeline (no update) ──► metrics
```

## Archivos a crear

```
/mnt/c/Users/henry/desktop/RL/pid_rl/
├── config.py           # bounds, pesos de reward, constantes PID, hiperparámetros
├── sim.py              # Simulador numpy: PID + mercado + colateralización
├── scenarios.py        # Generador synthetic + loader yfinance
├── prompt.py           # build_prompt + parse_output + JSON schema
├── reward.py           # reward_func(prompts, completions) → List[float]
├── baselines.py        # random, static RAI gains, heuristic (replica reasoning.ts)
├── eval.py             # Loop eval vs histórico real
└── train_grpo.ipynb    # Notebook Unsloth modificado
```

## Verificación onchain (deployed v10.1 — 2026-04-19)

**Bounds del ParameterGuard (source of truth)**:
- KP range: **[0.1, 10.0]** WAD (widened for demo; original was [1.4, 2.6])
- KI range: **[0.0, 0.1]** WAD (widened; original [0.001, 0.01])
- Max ΔKP per update: **2.0** WAD
- Max ΔKI per update: **0.2** WAD
- Cooldown normal: **5s**
- Cooldown emergency: **3s**
- Max updates: **1000**

**Baseline inicial en deploy** (`scripts/deploy_sepolia.sh`):
- KP = 2.0 WAD
- KI = 0.002 WAD
- Noise barrier, integral period, feedback bounds, per-second leak: ver script

**Anomalía actual**: `pid.get_controller_gains` = (10.0, 0.1) al 2026-04-19 — agente GLM-5.1 está monotónicamente incrementando. Ese es el bug que RL debería arreglar.

## Observation space (del `ProtocolState` de monitor.ts)

Estado inmediato:
- `marketPriceUsd` (GRIT/USD)
- `redemptionPriceUsd` (GRIT target)
- `collateralPriceUsd` (BTC/USD)
- `deviationPct` (peg deviation)
- `collateralDropPct` (BTC drop from baseline)
- `kp, ki` (current gains)
- `lastProportional, lastIntegral` (PID error terms)
- `guardUpdateCount` (budget usado)

Historial temporal (agregado para contexto):
- `btc_returns_last_24_samples`
- `btc_volatility_rolling`
- `btc_drawdown_from_window_high`
- `deviation_history_last_8_samples`
- `blocks_since_last_update`

## Action space

Output JSON estructurado:
```json
{
  "action": "hold | adjust | adjust_emergency",
  "new_kp": float,
  "new_ki": float,
  "is_emergency": bool,
  "reasoning": string
}
```

Validación:
- `new_kp ∈ [0.1, 10.0]`
- `new_ki ∈ [0.0, 0.1]`
- `|new_kp - current_kp| ≤ 2.0`
- `|new_ki - current_ki| ≤ 0.2`

## Reward function

```
reward = 
    - α * mean(|deviation_pct| over next N blocks)     # sistema sano
    - β * std(redemption_rate over next N blocks)      # rate predecible
    - γ * ||Δgains||                                    # sin movimientos gratuitos
    - δ * bounds_violation                              # guardrails
    - ε * monotonic_drift_penalty                       # no subir monotónicamente
    + bonus_justified_emergency                         # emergencias correctas
    - false_alarm_penalty                               # emergencias falsas
```

Pesos iniciales: `α=1.0, β=0.5, γ=0.2, δ=10.0, ε=0.3`. Iteramos según baselines.

## Data strategy

### Training: sintético

Generator parametrizado. Scenarios = paths de 200 pasos de BTC price.

| Tipo | Distribución | Peso |
|---|---|---|
| `stable` | GBM σ=0.5% | 15% |
| `crash` | GBM + jump -8% a -20% | 30% |
| `flash_crash` | Jump -15% + recovery 80% en 10 pasos | 15% |
| `pump` | GBM drift + + jumps +5% a +15% | 10% |
| `volatile` | σ=3%+ no drift | 20% |
| `recovery` | Start low + drift + progresivo | 10% |

### Validation: real crashes via yfinance

- LUNA collapse (May 2022)
- FTX collapse (Nov 2022)
- August 2024 yen carry trade
- March 2020 COVID
- Event reciente (~Ene 2025)

5 ventanas × 200 velas = 1000 pasos eval.

## Training hyperparameters

| Parámetro | Valor | Rationale |
|---|---|---|
| Base model | Qwen2.5-1.5B-Instruct | User choice |
| LoRA rank | 8 | Min viable, no overfit |
| Batch size | 4 | T4 memory |
| Num generations (G) | 4 | Grupo GRPO chico, T4 memory |
| Max prompt length | 1024 | State + history fit |
| Max completion length | 512 | JSON + reasoning |
| Learning rate | 5e-6 | Conservador, no romper Instruct prior |
| Max steps | 500 | Primer intento; extendemos si converge |
| Warmup ratio | 0.1 | Standard |
| KL coef (β) | 0.04 | TRL GRPO default |

ETA en T4 gratis: **~2.5-4 horas** para 500 steps.

## Evaluación — Definition of Done

✅ Valid JSON rate > 98%
✅ Mean reward > Static Gains baseline (synth + real)
✅ Mean reward ≥ Heuristic baseline (synth + real)
✅ % time `|dev| < 1%` > Heuristic baseline
✅ No monotonic drift (KP sube Y baja según corresponda)
✅ Emergency precision > 70% (true positives / declared emergencies)
✅ Funciona en validation set de real crashes

## Timeline

| Sesión | Trabajo | Horas humano | Horas GPU |
|---|---|---|---|
| 1 | `sim.py` + `config.py` + smoke test | 2-3h | - |
| 2 | `scenarios.py` + `prompt.py` + `reward.py` | 2-3h | - |
| 3 | `baselines.py` + validación sim vs baselines | 1-2h | - |
| 4 | Fork notebook Unsloth + primera corrida 50 steps | 1h | 0.5h |
| 5 | Training completo 500 steps | 1h | 3-4h |
| 6 | `eval.py` + comparación + iteración reward | 2-3h | 1h |
| **Total** | | **9-14h** | **5h** |

## Risks & mitigations

| Riesgo | Mitigación |
|---|---|
| Modelo no converge | SFT warmup con `decisions.jsonl` (75KB ya existe) |
| JSON malformado frecuente | Reward penaliza fuerte malformado |
| Reward hacking | Eval vs baselines lo detecta |
| Sim-to-real gap | Validation en real crashes es último test |
| Colab se corta | Checkpoint cada 50 steps |
| Sim muy lento | Profilear step 1; numba si hace falta |
| Bounds cambian | `config.py` fuente única |

## Decisiones pendientes (non-blocking)

- Velocidad de arbitraje (k) y sensibilidad BTC (β) del MarketModel
- Pesos α, β, γ, δ, ε finales (iteramos)
- Distribución final de scenario types

## Baselines obligatorias (antes de entrenar)

1. **RandomPolicy**: acciones random dentro de bounds
2. **StaticGainsPolicy**: siempre `(KP=2.0, KI=0.002)` — deploy baseline, nunca ajusta
3. **HeuristicPolicy**: replica del `reasoning.ts` actual (thresholds de deviation %)

Si el modelo entrenado no le gana a estos → hay bug en reward/sim, NO en training.

## Referencias

- Repos del proyecto: `/mnt/c/Users/henry/desktop/pid/` (protocol + agent actual)
- Deployed state: `deployed_v11.json`
- Agent actual: `agent/src/` (TypeScript, GLM-5.1 via CommonStack API)
- Decisions log: `agent/decisions.jsonl` (75KB — potencial SFT warmup data)
- Reflexer digital twin (futuro v2 validation): https://github.com/reflexer-labs/reflexer-digital-twin
