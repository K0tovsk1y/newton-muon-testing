# Наши модификации NM-1 (для H200)

## Файлы

- `train_gpt_newton_muon_1.py` — оригинал авторов (NM-base).
- `train_gpt_newton_muon_1_gamma.py` — наш вариант **NM-gamma**: damping по Frobenius норме K вместо trace.
- `train_gpt_newton_muon_1_interval.py` — наш вариант **NM-interval**: refresh преconditioner'а по накопленному пути lr вместо фикс. 32 шагов.
- `train_gpt_newton_muon_1_both.py` — наш вариант **NM-both**: gamma+interval вместе.

Все 3 наших файла = оригинал + ~20 строк в классе `Muon` (+ строка инстанциации). Остальной код их не трогался.

## Параметры (где лежат и как менять)

В каждом нашем файле гиперпараметры — в строке инстанциации `optimizer2 = Muon(...)` около 765-й строки. Можно править саму строку, ИЛИ передать через env-переменную при запуске (рекомендую — не нужно дёргать код):

| Переменная | По умолчанию | Используется в файлах | Что регулирует |
|---|---|---|---|
| `NM_PATH_THRESHOLD` | `0.013` | interval, both | Порог накопленного пути для refresh. Меньше → refresh чаще. |
| `NM_RIDGE_MULT` | `0.2` | gamma, both | Множитель `ridge_mult` в формуле damping. Меньше → preconditioner влиятельнее. |

### Дефолт `path_threshold = 0.013`

Подобран под их трапециевидный scheduler: на плато (4400 итераций) `lr_muon = 0.0004`, что даёт refresh каждые `0.013/0.0004 = 32.5` шага — почти то же что фикс=32 в оригинале. После warmdown (с итерации 4400), lr падает → refresh становится реже (это и есть смысл оптимизации). К концу обучения интервал растёт до 100+ шагов.

### Дефолт `precond_ridge_mult = 0.2`

Тот же что у авторов в оригинале. ВНИМАНИЕ для NM-gamma: Frob норма даёт RMS собственных значений, что ≥ среднему (trace/d). Поэтому с тем же `ridge_mult=0.2` Frob-damping в среднем сильнее trace. Возможно, для Frob нужен меньший ridge_mult для эквивалентного эффекта.

## Что прогонять

Полный сравнительный набор (5 прогонов):

1. `train_gpt_adam_1.py` — AdamW baseline (есть результат в README: loss 3.380, time 7228s)
2. `train_gpt_muon_1.py` — Muon baseline (loss 3.279, 7314s)
3. `train_gpt_newton_muon_1.py` — NM-base (loss 3.261, 7443s) — наши прогоны должны примерно повторить это число
4. `train_gpt_newton_muon_1_gamma.py` — NM-gamma (наш)
5. `train_gpt_newton_muon_1_interval.py` — NM-interval (наш)
6. `train_gpt_newton_muon_1_both.py` — NM-both (наш)

Если время ограничено, минимум: `newton_muon_1` (повторить базовый результат на нашем железе) + наши 3 варианта = 4 прогона.

## Запуск на удалённом сервере (выживание detach'а)

### Через tmux (рекомендуется)

```bash
# первая сессия
tmux new -s nm-base
python train_gpt_newton_muon_1.py 2>&1 | tee logs/nm-base.txt
# Ctrl-b затем d — detach
# logout/disconnect — процесс продолжает работу

# подключиться обратно
tmux attach -t nm-base

# вторая сессия параллельно (если несколько GPU)
tmux new -s nm-both
python train_gpt_newton_muon_1_both.py 2>&1 | tee logs/nm-both.txt
```

### Через nohup (если нет tmux)

```bash
mkdir -p logs
nohup python train_gpt_newton_muon_1_gamma.py > logs/nm-gamma.txt 2>&1 &
disown
```

### Последовательно (один GPU, гонит все варианты подряд)

```bash
mkdir -p logs
tmux new -s nm-sequential
for variant in '' '_gamma' '_interval' '_both'; do
  python train_gpt_newton_muon_1${variant}.py 2>&1 | tee logs/nm${variant:-_base}.txt
done
```

### Локальный sweep по гиперпараметрам

Раз мы не знаем оптимальных значений на нашем железе, разумно пройтись узким диапазоном вокруг дефолтов. Делать только для одиночных версий (gamma и interval), потом в `_both` подставить лучшее.

**Sweep по `NM_PATH_THRESHOLD` (только для interval-варианта):**
```bash
mkdir -p logs
tmux new -s nm-interval-sweep
for thr in 0.0065 0.013 0.026 0.052; do
  NM_PATH_THRESHOLD=$thr python train_gpt_newton_muon_1_interval.py 2>&1 \
    | tee logs/interval_thr${thr}.txt
done
```
Базовое значение 0.013 даёт refresh~32 шага на плато. Меньшие — чаще, бóльшие — реже. Если оптимум окажется в крайних точках диапазона — расширить.

**Sweep по `NM_RIDGE_MULT` (только для gamma-варианта):**
```bash
mkdir -p logs
tmux new -s nm-gamma-sweep
for r in 0.05 0.1 0.2 0.4; do
  NM_RIDGE_MULT=$r python train_gpt_newton_muon_1_gamma.py 2>&1 \
    | tee logs/gamma_ridge${r}.txt
done
```
Frob норма ≥ trace/d, поэтому пробуем включая значения МЕНЬШЕ авторского 0.2. Если оптимум на 0.05 — расширить вниз.

**После того как найдены лучшие thr* и r*:**
```bash
tmux new -s nm-both-best
NM_PATH_THRESHOLD=<thr*> NM_RIDGE_MULT=<r*> \
  python train_gpt_newton_muon_1_both.py 2>&1 | tee logs/both_best.txt
```

## Что искать в логах

Скрипт пишет `step:<i>/6200 val_loss:<v> train_time:<t>ms step_avg:<a>ms` каждые 100 шагов. Финальная метрика — `val_loss` ближе к концу обучения.

Из README авторов:
| Метод | Final val_loss | Wall time (s, H100) |
|---|---|---|
| AdamW | 3.3801 | 7228.4 |
| Muon | 3.2793 | 7314.1 |
| Newton-Muon | 3.2611 | 7443.3 |

На H200 ожидать пропорционально быстрее (≈ 1.4-1.6x). Время на 1 прогон: ~4500-5000s ≈ 75-85 минут.

## Возможные проблемы

- **Скачивание данных**: первый запуск требует `python data/cached_fineweb10B.py 50` (50 шардов FineWeb). Если прокси блокирует HF — нужно прокинуть `HF_HUB_OFFLINE=0 https_proxy=... ...` или скачать отдельно.
- **torch.compile**: первый запуск компилируется ~5-10 мин. Это входит в общее время.
- **Память**: NM-1 требует ≥80 GB GPU (по их README). H200 96 GB — норм.
