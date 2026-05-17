"""Наблюдатель за прогоном NM-1 на H200/H100.
   Парсит logs/*.txt → определяет метод по embedded коду → строит графики train/val + сводку.

Usage:
    python watch_speedrun.py --logs ~/Newton-Muon/logs --out ./speedrun_compare.png --interval 60
    # Ctrl-C чтобы остановить.
"""
import argparse
import os
import re
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # без X-сервера
import matplotlib.pyplot as plt

# numpy 1.x: trapz; numpy 2.x: trapezoid. Используем что доступно.
_trapz = getattr(np, "trapezoid", None) or np.trapz


METHOD_COLORS = {
    "AdamW":         "#1f77b4",
    "Muon":          "#7f7f7f",
    "NM-base":       "#9467bd",
    "NM-gamma":      "#ff7f0e",
    "NM-interval":   "#17becf",
    "NM-both":       "#2ca02c",
    "NM-v2-Woodbury":"#e377c2",
    "NM-v3-attn":    "#bcbd22",
    "unknown":       "#aaaaaa",
}

# Друзья прогоняли на чистых картах — будем рисовать сплошными.
DEFAULT_FRIEND_UUIDS = {
    "ddd3b352", "bb2a107b", "be3efb2f", "c83e8209",
}
# Прогоны, которые не учитываем при усреднении (Woodbury заметно хуже — иначе он сдвинет mean).
DEFAULT_EXCLUDE_FROM_MEAN = {
    "be3efb2f",
}


def identify_method(text: str) -> tuple[str, dict]:
    """Возвращает (имя_метода, словарь_параметров) по содержимому лога.

    В начале каждого лога авторский скрипт пишет весь свой код. Ищем там строку
    инстанциации оптимизатора и определяем тип.
    """
    params = {}

    # AdamW случай (train_gpt_adam_*.py) — нет Muon вообще
    if re.search(r"optimizer\s*=\s*torch\.optim\.AdamW\b", text) and "class Muon" not in text:
        return "AdamW", params

    # train_gpt_muon_*.py — Muon без preconditioner
    has_muon_class = "class Muon" in text
    has_attach = re.search(r"\.attach_preconditioner\s*\(", text)
    if has_muon_class and not has_attach:
        return "Muon", params

    # ниже — Newton-Muon разновидности. Ищем флаги в инстанциации.
    has_frob = re.search(r"use_frob_gamma\s*=\s*True", text) is not None
    has_path = re.search(r"use_path_interval\s*=\s*True", text) is not None

    m_ridge = re.search(r"precond_ridge_mult\s*=\s*([0-9.eE+-]+)", text)
    m_thr = re.search(r"path_threshold\s*=\s*([0-9.eE+-]+)", text)
    if m_ridge:
        try: params["ridge_mult"] = float(m_ridge.group(1))
        except: pass
    if m_thr:
        try: params["path_threshold"] = float(m_thr.group(1))
        except: pass

    # Друзья развили v2_8 (Woodbury+SVD) и v3 (attn-weighted covariance)
    has_woodbury = re.search(r"woodbury|Woodbury|randomized SVD|rank\s*=\s*256", text) is not None
    has_attn_avg = re.search(r"ATTN_AVG_MOMENTUM|attn_avg", text) is not None

    if has_woodbury:
        m_rank = re.search(r"rank\s*=\s*(\d+)", text)
        if m_rank:
            try: params["rank"] = int(m_rank.group(1))
            except: pass
        return "NM-v2-Woodbury", params
    if has_attn_avg:
        m_mom = re.search(r"ATTN_AVG_MOMENTUM\s*=\s*([0-9.]+)", text)
        if m_mom:
            try: params["attn_mom"] = float(m_mom.group(1))
            except: pass
        return "NM-v3-attn", params

    if has_frob and has_path:
        return "NM-both", params
    if has_frob:
        return "NM-gamma", params
    if has_path:
        return "NM-interval", params
    if has_muon_class and has_attach:
        return "NM-base", params

    return "unknown", params


RE_STEP_TRAIN = re.compile(r"^step:(\d+)/\d+\s+train_loss:([\d.eE+-]+|nan|inf)\s+train_time:([\d.]+)ms")
RE_STEP_VAL = re.compile(r"^step:(\d+)/\d+\s+val_loss:([\d.eE+-]+|nan|inf)\s+train_time:([\d.]+)ms")


def parse_log(path: str) -> dict:
    """Возвращает dict с train/val кривыми и метаинформацией."""
    with open(path) as f:
        text = f.read()
    method, params = identify_method(text)

    train_steps, train_loss, train_times = [], [], []
    val_steps, val_loss, val_times = [], [], []

    for line in text.splitlines():
        m = RE_STEP_TRAIN.match(line)
        if m:
            s = int(m.group(1)); v = float(m.group(2)); t = float(m.group(3))
            train_steps.append(s); train_loss.append(v); train_times.append(t)
            continue
        m = RE_STEP_VAL.match(line)
        if m:
            s = int(m.group(1)); v = float(m.group(2)); t = float(m.group(3))
            val_steps.append(s); val_loss.append(v); val_times.append(t)

    return {
        "path": path,
        "uuid": os.path.basename(path).replace(".txt", ""),
        "method": method,
        "params": params,
        "train_steps": np.array(train_steps),
        "train_loss": np.array(train_loss, dtype=float),
        "train_time_ms": np.array(train_times),
        "val_steps": np.array(val_steps),
        "val_loss": np.array(val_loss, dtype=float),
        "val_time_ms": np.array(val_times),
    }


def auc_val(d: dict, max_step: int | None = None) -> float:
    """Trapezoid-AUC val-кривой до max_step (включительно).
    Нормировано на длину окна. Чем меньше тем быстрее сходится."""
    if len(d["val_steps"]) < 2:
        return float("nan")
    finite = np.isfinite(d["val_loss"])
    x = d["val_steps"][finite]
    y = d["val_loss"][finite]
    if max_step is not None:
        mask = x <= max_step
        x = x[mask]; y = y[mask]
    if len(x) < 2:
        return float("nan")
    return float(_trapz(y, x) / max(x[-1] - x[0], 1))


def val_at(d: dict, max_step: int) -> float:
    """val_loss на последнем измеренном шаге ≤ max_step (без интерполяции)."""
    finite = np.isfinite(d["val_loss"])
    x = d["val_steps"][finite]
    y = d["val_loss"][finite]
    mask = x <= max_step
    if not mask.any():
        return float("nan")
    return float(y[mask][-1])


def train_at(d: dict, max_step: int) -> float:
    finite = np.isfinite(d["train_loss"])
    x = d["train_steps"][finite]
    y = d["train_loss"][finite]
    mask = x <= max_step
    if not mask.any():
        return float("nan")
    return float(y[mask][-1])


def train_window_mean(d: dict, max_step: int, window: int = 10) -> float:
    """Среднее train_loss по последним <window> точкам ≤ max_step.
       Сглаживает шум — train пишется на каждом шаге."""
    finite = np.isfinite(d["train_loss"])
    x = d["train_steps"][finite]
    y = d["train_loss"][finite]
    mask = x <= max_step
    if not mask.any():
        return float("nan")
    return float(y[mask][-window:].mean())


def label(d: dict) -> str:
    params_str = ""
    if d["params"]:
        kvs = ", ".join(f"{k}={v}" for k, v in d["params"].items())
        params_str = f" ({kvs})"
    return f"{d['method']}{params_str} [{d['uuid'][:8]}]"


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Скользящее среднее с честными краями: на границах используется expanding window,
    а не неполные окна (это убирает артефакты-«скачки» в начале/конце)."""
    if window <= 1 or len(y) < 2:
        return y
    win = min(window, len(y))
    cs = np.cumsum(np.insert(y, 0, 0.0))
    out = np.empty(len(y), dtype=float)
    for i in range(len(y)):
        lo = max(0, i - win + 1)
        out[i] = (cs[i + 1] - cs[lo]) / (i + 1 - lo)
    return out


def _is_friend(d: dict, friend_uuids: set[str]) -> bool:
    return d["uuid"][:8] in friend_uuids or d["uuid"] in friend_uuids


def _build_palette(logs: list[dict], friend_uuids: set[str]) -> dict[str, dict]:
    """Возвращает {uuid → {color, linestyle, linewidth, label}} с уникальными цветами на прогон.
       Group: 'F' (friend) — сплошной, 'M' (мой) — пунктир."""
    # tab20 даёт 20 различимых цветов
    cmap_friend = plt.get_cmap('tab10')   # для friend (солидные)
    cmap_mine   = plt.get_cmap('Set1')    # для моих (пунктиром)

    friend_i = mine_i = 0
    style_map = {}
    for d in logs:
        is_friend = _is_friend(d, friend_uuids)
        params_str = ""
        if d["params"]:
            kvs = ", ".join(f"{k}={v}" for k, v in d["params"].items())
            params_str = f" ({kvs})"
        prefix = "F" if is_friend else "M"
        lbl = f"{prefix}|{d['method']}{params_str} [{d['uuid'][:8]}]"

        if is_friend:
            color = cmap_friend(friend_i % 10); friend_i += 1
            style_map[d["uuid"]] = dict(color=color, linestyle='-', linewidth=2.0, alpha=0.95, label=lbl)
        else:
            color = cmap_mine(mine_i % 9); mine_i += 1
            style_map[d["uuid"]] = dict(color=color, linestyle='--', linewidth=1.6, alpha=0.9, label=lbl)
    return style_map


def _line_style(d: dict, palette: dict[str, dict]) -> dict:
    """Вытаскивает kwargs для plt.plot из palette (без label — label передаётся отдельно)."""
    st = palette[d["uuid"]]
    return {k: v for k, v in st.items() if k != "label"}


def _line_label(d: dict, palette: dict[str, dict]) -> str:
    return palette[d["uuid"]]["label"]


def _is_excluded(d: dict, exclude_uuids: set[str]) -> bool:
    return d["uuid"][:8] in exclude_uuids or d["uuid"] in exclude_uuids


def plot_all(logs: list[dict], out: str, cutoff: int, title_suffix: str = "",
             train_smooth: int = 50, friend_uuids: set[str] = set(),
             step_skip: int = 1000, exclude_from_mean: set[str] = set()):
    """2×2: верх — Δ vs mean (train/val), низ — абсолютные кривые (от step_skip).
       Свои = пунктир, чужие (clean) = сплошные. Каждому прогону — свой цвет.
       Прогоны из exclude_from_mean всё ещё рисуются, но не учитываются в mean
       и не влияют на y-автоскейл."""
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Newton-Muon-1 speedrun comparison{title_suffix}", fontsize=14, fontweight='bold')

    ax_tr_diff, ax_val_diff = axs[0, 0], axs[0, 1]
    ax_tr_abs,  ax_val_abs  = axs[1, 0], axs[1, 1]

    palette = _build_palette(logs, friend_uuids)

    train_logs = [d for d in logs if len(d["train_steps"])]
    val_logs   = [d for d in logs if len(d["val_steps"])]
    tr_max = int(min(d["train_steps"][-1] for d in train_logs)) if train_logs else 0
    val_max = int(min(d["val_steps"][-1] for d in val_logs)) if val_logs else 0

    # ---------- TRAIN ----------
    if train_logs and tr_max > 0:
        common = None; per_method_loss = []
        for d in train_logs:
            mask = d["train_steps"] <= tr_max
            steps = d["train_steps"][mask]; loss = d["train_loss"][mask]
            if common is None:
                common = steps
            else:
                L = min(len(common), len(steps))
                common = common[:L]
                per_method_loss = [pml[:L] for pml in per_method_loss]
                loss = loss[:L]
            per_method_loss.append(loss)

        # mean — только по тем что НЕ в exclude_from_mean
        mean_contrib = [l for d, l in zip(train_logs, per_method_loss)
                        if not _is_excluded(d, exclude_from_mean)]
        if mean_contrib:
            mean_train = np.stack(mean_contrib).mean(axis=0)
        else:
            mean_train = np.zeros(len(common))

        for d, loss in zip(train_logs, per_method_loss):
            style = _line_style(d, palette); lbl = _line_label(d, palette)
            diff_sm = _smooth(loss - mean_train, train_smooth)
            loss_sm = _smooth(loss, train_smooth)
            ax_tr_diff.plot(common, diff_sm, label=lbl, **style)
            ax_tr_abs.plot(common, loss_sm, label=lbl, **style)
        ax_tr_diff.axhline(0, color='black', linewidth=0.5)

    # ---------- VAL ----------
    if val_logs and val_max > 0:
        common_val = None; per_method_val = []
        for d in val_logs:
            mask = d["val_steps"] <= val_max
            steps = d["val_steps"][mask]; loss = d["val_loss"][mask]
            if common_val is None:
                common_val = steps
            else:
                L = min(len(common_val), len(steps))
                common_val = common_val[:L]
                per_method_val = [pmv[:L] for pmv in per_method_val]
                loss = loss[:L]
            per_method_val.append(loss)

        mean_contrib_val = [l for d, l in zip(val_logs, per_method_val)
                            if not _is_excluded(d, exclude_from_mean)]
        if mean_contrib_val:
            mean_val = np.stack(mean_contrib_val).mean(axis=0)
        else:
            mean_val = np.zeros(len(common_val))

        for d, loss in zip(val_logs, per_method_val):
            style = _line_style(d, palette); lbl = _line_label(d, palette)
            ax_val_diff.plot(common_val, loss - mean_val, label=lbl, **style)
            ax_val_abs.plot(common_val, loss, label=lbl, **style)
        ax_val_diff.axhline(0, color='black', linewidth=0.5)

    # обрезаем абсолютные графики со step_skip + перестраиваем y под видимый диапазон
    def _autoscale_y(ax, x_arrays, y_arrays, x_lo):
        ymin, ymax = float("inf"), float("-inf")
        for x, y in zip(x_arrays, y_arrays):
            mask = (np.asarray(x) >= x_lo) & np.isfinite(np.asarray(y))
            if mask.any():
                yvals = np.asarray(y)[mask]
                ymin = min(ymin, float(yvals.min()))
                ymax = max(ymax, float(yvals.max()))
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            pad = (ymax - ymin) * 0.05
            ax.set_ylim(ymin - pad, ymax + pad)

    # Y-autoscale: учитываем ТОЛЬКО прогоны не из exclude_from_mean
    # (чтобы Woodbury не утянул scale в небо).
    if train_logs and tr_max > step_skip:
        ax_tr_abs.set_xlim(step_skip, tr_max)
        kept_loss = [_smooth(l, train_smooth) for d, l in zip(train_logs, per_method_loss)
                     if not _is_excluded(d, exclude_from_mean)]
        _autoscale_y(ax_tr_abs, [common] * len(kept_loss), kept_loss, step_skip)
    if val_logs and val_max > step_skip:
        ax_val_abs.set_xlim(step_skip, val_max)
        kept_val = [l for d, l in zip(val_logs, per_method_val)
                    if not _is_excluded(d, exclude_from_mean)]
        _autoscale_y(ax_val_abs, [common_val] * len(kept_val), kept_val, step_skip)
    # То же на Δ-панелях (чтобы Woodbury не вытянул)
    if train_logs and tr_max > 0:
        kept_diff = [_smooth(l - mean_train, train_smooth)
                     for d, l in zip(train_logs, per_method_loss)
                     if not _is_excluded(d, exclude_from_mean)]
        _autoscale_y(ax_tr_diff, [common] * len(kept_diff), kept_diff, 0)
    if val_logs and val_max > 0:
        kept_dv = [l - mean_val for d, l in zip(val_logs, per_method_val)
                   if not _is_excluded(d, exclude_from_mean)]
        _autoscale_y(ax_val_diff, [common_val] * len(kept_dv), kept_dv, 0)

    for ax, t in [(ax_tr_diff, f"Δtrain vs mean (smooth={train_smooth})"),
                  (ax_val_diff, "Δval vs mean"),
                  (ax_tr_abs, f"train CE per run (smooth={train_smooth}, from step {step_skip})"),
                  (ax_val_abs, f"val CE per run (from step {step_skip})")]:
        ax.set_title(t); ax.set_xlabel("step"); ax.grid(alpha=0.4)
        ax.legend(fontsize=7, loc='best')
        if cutoff > 0:
            ax.axvline(cutoff, color='red', linestyle=':', alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_by_time(logs: list[dict], out: str, title_suffix: str = "",
                 friend_uuids: set[str] = set(), time_skip_s: float = 500.0):
    """val/train по wall-time. Только друзья (clean GPU прогоны). С time_skip_s секунд."""
    # фильтруем — только friend's (мои на shared GPU не показывают честный wall)
    logs = [d for d in logs if _is_friend(d, friend_uuids)]
    if not logs:
        return  # нечего показывать

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Newton-Muon-1 by wall-time (only clean-GPU runs){title_suffix}",
                 fontsize=14, fontweight='bold')
    ax_tr, ax_val = axs
    palette = _build_palette(logs, friend_uuids)

    max_t = 0.0
    for d in logs:
        style = _line_style(d, palette); lbl = _line_label(d, palette)
        if len(d["train_steps"]):
            x = d["train_time_ms"] / 1000.0
            y_sm = _smooth(d["train_loss"], 50)
            ax_tr.plot(x, y_sm, label=lbl, **style)
            if len(x): max_t = max(max_t, float(x[-1]))
        if len(d["val_steps"]):
            x = d["val_time_ms"] / 1000.0
            ax_val.plot(x, d["val_loss"], label=lbl, **style)
            if len(x): max_t = max(max_t, float(x[-1]))

    def _autoscale_y_time(ax, getx, gety, x_lo):
        ymin, ymax = float("inf"), float("-inf")
        for d in logs:
            x = getx(d); y = gety(d)
            if x is None or len(x) == 0: continue
            mask = (np.asarray(x) >= x_lo) & np.isfinite(np.asarray(y))
            if mask.any():
                yv = np.asarray(y)[mask]
                ymin = min(ymin, float(yv.min())); ymax = max(ymax, float(yv.max()))
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            pad = (ymax - ymin) * 0.05
            ax.set_ylim(ymin - pad, ymax + pad)

    for ax, t in [(ax_tr, "train CE vs wall_s (smooth=50)"),
                  (ax_val, "val CE vs wall_s")]:
        ax.set_title(t); ax.set_xlabel("wall time (s)"); ax.grid(alpha=0.4)
        ax.legend(fontsize=8, loc='best')
        if max_t > time_skip_s:
            ax.set_xlim(time_skip_s, max_t)

    if max_t > time_skip_s:
        _autoscale_y_time(ax_tr,
                          lambda d: d["train_time_ms"] / 1000.0 if len(d["train_steps"]) else None,
                          lambda d: _smooth(d["train_loss"], 50) if len(d["train_steps"]) else None,
                          time_skip_s)
        _autoscale_y_time(ax_val,
                          lambda d: d["val_time_ms"] / 1000.0 if len(d["val_steps"]) else None,
                          lambda d: d["val_loss"] if len(d["val_steps"]) else None,
                          time_skip_s)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _config_key(d: dict) -> str:
    """Уникальный ключ варианта: метод + значения параметров. Используем для группировки."""
    if not d["params"]:
        return d["method"]
    parts = [f"{k}={d['params'][k]}" for k in sorted(d["params"].keys())]
    return f"{d['method']} ({', '.join(parts)})"


def _build_row(group_logs: list[dict], cutoff: int) -> dict:
    """Считаем сводные числа для группы прогонов с одинаковым (методом+параметрами).
    Если в группе несколько прогонов — усредняем."""
    vals_at_cut, train_at_cut_vec, auc_vec, wall_cut_vec = [], [], [], []
    own_finals, own_walls, own_lasts, uuids = [], [], [], []
    for d in group_logs:
        last_step_self = int(max(
            d["train_steps"][-1] if len(d["train_steps"]) else 0,
            d["val_steps"][-1] if len(d["val_steps"]) else 0,
        ))
        own_lasts.append(last_step_self)
        uuids.append(d["uuid"][:8])

        if len(d["val_steps"]):
            mask = d["val_steps"] <= cutoff
            if mask.any():
                wall_cut_vec.append(float(d["val_time_ms"][mask][-1]) / 1000.0)
        train_at_cut_vec.append(train_at(d, cutoff))
        vals_at_cut.append(val_at(d, cutoff))
        auc_vec.append(auc_val(d, cutoff))

        if len(d["val_steps"]):
            finite = np.isfinite(d["val_loss"])
            if finite.any():
                own_finals.append(float(d["val_loss"][finite][-1]))
                own_walls.append(float(d["val_time_ms"][finite][-1]) / 1000.0)

    def _avg(arr):
        arr = [x for x in arr if np.isfinite(x)] if arr else []
        return float(np.mean(arr)) if arr else float("nan")

    return {
        "config": _config_key(group_logs[0]),
        "method": group_logs[0]["method"],
        "params": group_logs[0]["params"],
        "n": len(group_logs),
        "uuids": uuids,
        "self_last_min": min(own_lasts) if own_lasts else 0,
        "self_last_max": max(own_lasts) if own_lasts else 0,
        "train_at_cut": _avg(train_at_cut_vec),
        "val_at_cut": _avg(vals_at_cut),
        "auc_val_cut": _avg(auc_vec),
        "wall_at_cut_s": _avg(wall_cut_vec),
        "own_final_val": _avg(own_finals),
        "own_final_wall_s": _avg(own_walls),
    }


def summarize(logs: list[dict]) -> tuple[str, int]:
    """Сводка + общий cutoff_step (минимум last_val_step среди ВСЕХ прогонов)."""
    last_val_steps = [int(d["val_steps"][-1]) for d in logs if len(d["val_steps"])]
    cutoff = min(last_val_steps) if last_val_steps else 0

    # Отсортируем сырые логи: NM-base первым, остальные по val на cutoff
    def _sort_key(d):
        is_nmbase = d["method"] == "NM-base"
        v = val_at(d, cutoff)
        return (0 if is_nmbase else 1, d["uuid"] if is_nmbase else "", v if np.isfinite(v) else 1e9)
    logs = sorted(logs, key=_sort_key)

    # Каждый лог — отдельная строка
    rows = [_build_row([d], cutoff) for d in logs]
    # Прикрепляем сырой лог к строке для попарных diff'ов
    for r, d in zip(rows, logs):
        r["_log"] = d

    if not rows:
        return ("Нет лог-файлов.", 0)

    lines = []
    # Таблица 1 — все метрики (cutoff)
    lines.append(f"Все значения на cutoff_step = {cutoff}. NM-1 авторский ориентир: final_val=3.2611.")
    lines.append("")
    lines.append(f"{'#':>2}  {'config':<48} {'uuid':<9}  {'own_last':>8}  {'train':>7}  {'val':>7}  {'auc':>7}  {'wall_s':>8}  {'own_final_val':>13}")
    lines.append("-" * 134)
    for i, r in enumerate(rows, start=1):
        lines.append(
            f"{i:>2}  {r['config']:<48} {r['uuids'][0]:<9}  {r['self_last_max']:>8}  "
            f"{r['train_at_cut']:>7.4f}  {r['val_at_cut']:>7.4f}  "
            f"{r['auc_val_cut']:>7.4f}  {r['wall_at_cut_s']:>8.1f}  {r['own_final_val']:>13.4f}"
        )

    # Таблица 2 — попарные Δ.
    # Δtrain: pair_step берётся по train_steps (мелкая сетка — каждый step),
    #         Δ = mean(last TRAIN_WINDOW значений) у каждого, потом разница.
    # Δval:   pair_step берётся по val_steps (раз в 100), Δ точечно.
    TRAIN_WINDOW = 10
    if len(rows) >= 2:
        lines.append("")
        lines.append(f"Попарные Δ = (j − i). Δtrain: max общий train_step, mean(last {TRAIN_WINDOW}). Δval: max общий val_step.")
        lines.append(f"  {'i':>2}  {'j':>2}  {'tr_step':>7}  {'Δtrain':>9}  {'tr_win':>6}  {'val_step':>8}  {'Δval':>9}  {'val_win':>7}")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                log_i = rows[i]["_log"]; log_j = rows[j]["_log"]
                # train pair step — независимо, мелкая сетка
                last_tr_i = int(log_i["train_steps"][-1]) if len(log_i["train_steps"]) else 0
                last_tr_j = int(log_j["train_steps"][-1]) if len(log_j["train_steps"]) else 0
                tr_step = min(last_tr_i, last_tr_j)
                # val pair step — раз в 100
                last_val_i = int(log_i["val_steps"][-1]) if len(log_i["val_steps"]) else 0
                last_val_j = int(log_j["val_steps"][-1]) if len(log_j["val_steps"]) else 0
                val_step = min(last_val_i, last_val_j)

                ti = train_window_mean(log_i, tr_step, TRAIN_WINDOW)
                tj = train_window_mean(log_j, tr_step, TRAIN_WINDOW)
                vi = val_at(log_i, val_step); vj = val_at(log_j, val_step)
                dt = (tj - ti) if (np.isfinite(ti) and np.isfinite(tj)) else float("nan")
                dv = (vj - vi) if (np.isfinite(vi) and np.isfinite(vj)) else float("nan")
                tr_win = ("=" if not np.isfinite(dt) or abs(dt) < 1e-6
                          else (f"{j+1}>{i+1}" if dt < 0 else f"{i+1}>{j+1}"))
                val_win = ("=" if not np.isfinite(dv) or abs(dv) < 1e-6
                           else (f"{j+1}>{i+1}" if dv < 0 else f"{i+1}>{j+1}"))
                lines.append(f"  {i+1:>2}  {j+1:>2}  {tr_step:>7}  {dt:+9.4f}  {tr_win:>6}  {val_step:>8}  {dv:+9.4f}  {val_win:>7}")

    return ("\n".join(lines), cutoff)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default=os.path.expanduser("~/Newton-Muon/logs"),
                   help="директория с *.txt логами авторов")
    p.add_argument("--out", default="speedrun_compare.png",
                   help="куда сохранять PNG (перезаписывается)")
    p.add_argument("--interval", type=int, default=60,
                   help="секунд между обновлениями (0 = один раз и выход)")
    p.add_argument("--summary-out", default="speedrun_summary.txt",
                   help="куда писать текстовую сводку")
    p.add_argument("--time-plot", default="speedrun_by_time.png",
                   help="дополнительный график по wall-time (пунктир — friend's clean runs)")
    p.add_argument("--friend-uuids", type=str,
                   default=",".join(sorted(DEFAULT_FRIEND_UUIDS)),
                   help="comma-separated UUID prefixes (8 chars enough) clean-прогонов от друга")
    p.add_argument("--exclude-from-mean", type=str,
                   default=",".join(sorted(DEFAULT_EXCLUDE_FROM_MEAN)),
                   help="UUID prefixes которые рисуем но НЕ учитываем в mean/auto-scale")
    p.add_argument("--smooth", type=int, default=50, help="окно сглаживания train")
    args = p.parse_args()
    friend_uuids = set(s.strip() for s in args.friend_uuids.split(",") if s.strip())
    exclude_mean = set(s.strip() for s in args.exclude_from_mean.split(",") if s.strip())

    if not os.path.isdir(args.logs):
        print(f"[ERR] логи не найдены: {args.logs}")
        return

    iteration = 0
    while True:
        iteration += 1
        txt_files = sorted(p for p in (
            os.path.join(args.logs, f) for f in os.listdir(args.logs) if f.endswith(".txt")
        ))
        logs = []
        for tp in txt_files:
            try:
                d = parse_log(tp)
                if len(d["train_steps"]) or len(d["val_steps"]):
                    logs.append(d)
            except Exception as e:
                print(f"[WARN] не распарсил {tp}: {e}")

        if not logs:
            print(f"[#{iteration}] логов с данными пока нет в {args.logs}")
        else:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            summary, cutoff = summarize(logs)
            plot_all(logs, args.out, cutoff=cutoff,
                     title_suffix=f"  cut={cutoff}  ({ts})",
                     train_smooth=args.smooth, friend_uuids=friend_uuids,
                     exclude_from_mean=exclude_mean)
            plot_by_time(logs, args.time_plot,
                         title_suffix=f"  ({ts})", friend_uuids=friend_uuids)
            with open(args.summary_out, "w") as f:
                f.write(summary + "\n")
            print(f"\n[#{iteration} @ {ts}] {len(logs)} логов | step-plot → {args.out} | time-plot → {args.time_plot} | summary → {args.summary_out}")
            print(summary)

        if args.interval <= 0:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
