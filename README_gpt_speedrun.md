# Наши модификации NM-1 (для H200)

## Файлы

- `train_gpt_newton_muon_1.py` — оригинал авторов (NM-base).
- `train_gpt_newton_muon_1_gamma.py` — наш **NM-gamma**: damping по Frobenius норме K вместо trace.
- `train_gpt_newton_muon_1_interval.py` — наш **NM-interval**: refresh по накопленному пути lr вместо фикс. 32 шагов.
- `train_gpt_newton_muon_1_both.py` — **NM-both**: обе оптимизации сразу.

Каждый наш файл = оригинал + ~20 строк в классе `Muon` (+ строка инстанциации). Остальной код не тронут.

## Параметры (хардкод в скриптах)

Гиперпараметры теперь вписаны прямо в `optimizer2 = Muon(...)` около 765-й строки в каждом файле. Если хочешь другое значение — отредактируй там же.

| Файл | Параметры | Значения |
|---|---|---|
| `_gamma.py` | `precond_ridge_mult` | `0.02` |
| `_interval.py` | `path_threshold` | `0.013` |
| `_both.py` | оба | `0.02` + `0.013` |

### Как подобраны

**`precond_ridge_mult = 0.02` (Frob-gamma)** — авторский `ridge_mult = 0.2` рассчитан для trace-формулы; Frob `||K||_F/√d ≥ trace(K)/d` (RMS ≥ mean), поэтому при том же ridge_mult Frob-damping сильнее. На нашем мини-трансформере оптимум Frob был 0.005-0.03, что соответствует ~`base × (1/6 .. 1/10)`. Берём середину — `0.02 = base/10`.

**`path_threshold = 0.013` (path-based interval)** — у авторов scheduler **трапеция, не cosine**, поэтому повышать threshold сильно нельзя: плато даёт ровную lr, и нет «естественного» спада чтобы заработала разница path vs fixed. Подбираем так, чтобы:

- НА ПЛАТО refresh каждые **32 шага** (= оригинал авторов): `thr = 32 × lr_plateau = 32 × 0.0004 = 0.0128 ≈ 0.013`
- В warmdown lr падает → накопленный путь растёт медленнее → refresh **естественно реже**

Расчёт интеграла lr:
```
LR Muon = 0.1 × 0.004 = 0.0004 (на плато 4400 шагов), линейный warmdown 1800 шагов → 0
Интеграл lr:
  плато     4400 × 0.0004 = 1.760
  warmdown  0.0004 × 1800/2 = 0.360
  ИТОГО                     = 2.120
```
Сводная таблица режимов:

| threshold | refresh count за прогон | × реже fixed=32 | плато интервал | warmdown эффект |
|---|---|---|---|---|
| **0.013 (наш default)** | **160** | 1× (на плато) | **32 шага (= оригинал)** | автоматическая экономия ~17% |
| 0.022 | 97 | 1.7× | 55 шагов | |
| 0.044 | 48 | 3.4× | 110 шагов | (слишком агрессивно) |
| 0.085 | 25 | 6.6× | 210 шагов (мини-оптимум) | |

Безопасный профит: ~33 cholesky меньше (193 → 160) без потери частоты на плато. Если ещё хочется компромисс — можно поднять до 0.022.

## Что прогонять

Минимальный набор:

1. `train_gpt_newton_muon_1.py` — оригинальный NM-base, наш baseline на текущем железе.
2. `train_gpt_newton_muon_1_gamma.py` — Frob damping (ridge=0.02).
3. `train_gpt_newton_muon_1_interval.py` — path refresh (thr=0.05).
4. `train_gpt_newton_muon_1_both.py` — обе оптимизации.

## Запуск на удалённом сервере (выживает logout)

### tmux (рекомендуется)

```bash
mkdir -p logs

tmux new -s nm-base
python train_gpt_newton_muon_1.py 2>&1 | tee logs/nm_base.txt
# Ctrl-b d (detach) — можно безопасно отключаться

tmux new -s nm-gamma
python train_gpt_newton_muon_1_gamma.py 2>&1 | tee logs/nm_gamma.txt

tmux new -s nm-interval
python train_gpt_newton_muon_1_interval.py 2>&1 | tee logs/nm_interval.txt

tmux new -s nm-both
python train_gpt_newton_muon_1_both.py 2>&1 | tee logs/nm_both.txt
```

`tmux attach -t nm-base` чтобы вернуться.

### nohup (если tmux не доступен)

```bash
mkdir -p logs
nohup python train_gpt_newton_muon_1_gamma.py > logs/nm_gamma.txt 2>&1 &
disown
```

### Последовательно на одном GPU

```bash
mkdir -p logs
tmux new -s nm-seq
for f in train_gpt_newton_muon_1.py train_gpt_newton_muon_1_gamma.py \
         train_gpt_newton_muon_1_interval.py train_gpt_newton_muon_1_both.py; do
  python $f 2>&1 | tee logs/$(basename $f .py).txt
done
```

## Что искать в логах

Скрипт пишет каждые 100 шагов:
```
step:<i>/6200 val_loss:<v> train_time:<t>ms step_avg:<a>ms
```

Финальная метрика — `val_loss` после последних шагов. Авторские числа (H100):

| Метод | Final val_loss | Wall time (s) |
|---|---|---|
| AdamW | 3.3801 | 7228.4 |
| Muon | 3.2793 | 7314.1 |
| Newton-Muon | 3.2611 | 7443.3 |

На H200 ожидается ~1.4-1.6× быстрее.

## Контекст наших экспериментов

Сводка из нашей серии прогонов (GNN, маленький CNN, маленький трансформер):

- **Frob-gamma**: при правильном scaling (`ridge_mult /≈ 6-10` от их default) эквивалентен trace-gamma по качеству. На мини-трансформере дал параритет с NM-base.
- **Path interval**: refresh в 4-7× реже без потери качества. На мини-трансформере при ~7.5× реже refresh-а даже чуть обогнал NM-base.
- **Both**: на GNN — синергия (комбо лучше каждой по отдельности, t=3.6 на 150 сидах). На мини-трансформере — наоборот, комбо **хуже NM-base** (t=2.95, +0.0016 val_CE на 10 сидах). На длинном NM-1 (6200 шагов с warmdown) гипотеза синергии может ожить, но не гарантия.

Главная ожидаемая выгода NM-1 от `_interval`/`_both` — **экономия cholesky compute** (4-7× меньше refresh без падения качества). Это особенно ценно для speedrun.

## Возможные проблемы

- **Скачивание данных**: первый запуск требует `python data/cached_fineweb10B.py 50`. Если HF-прокси блокирует — разрешить или скачать отдельно через зеркало.
- **torch.compile**: первый запуск компилируется ~5-10 мин. Это входит в общий wall time.
- **Память**: NM-1 требует ≥80 GB GPU. H200 96 GB — хватит.
