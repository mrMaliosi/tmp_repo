# EBLO + `openJiuwen` Tournament Harness (Budget-Aware Token Estimation for Agentic AI)

Двухконтурная система, объединяющая оффлайн-эволюционный оркестратор **EBLO (Evolutionary Blazing-fast LLM Orchestrator)** и боевой турнирный харнесс на базе **`openJiuwen` (`openjiuwen==0.1.19`)** для соревнования **Budget-Aware Token Estimation for Agentic AI**.

## 0. Результаты реального прогона (проверено на живом LLM-прокси)

Полный турнир из 100 задач (80 выполнимых + 20 ловушек), 5 параллельных воркеров:

| Метрика | Значение |
|---|---|
| Решено выполнимых задач | **80/80 (100%)**, все ответы проверены верны |
| Собрано баллов | **14280 / 14280 (максимум)** |
| Ловушки (20 шт) | Все отсеяны, ни одна не получила `answer.txt` |
| Израсходовано C_eff | **5592.5** (~70 эффективных токенов на задачу) |
| Время | ~4 мин (лимит 1 ч) |
| 429 rate-limit | 33 события, все выжиты retry-механикой |

## 1. Сборка и запуск Docker-контейнера

Образ на базе `python:3.11.9-slim-bookworm`. В рантайме харнесс импортирует только `httpx`; `openjiuwen==0.1.19` ставится пинованным как базовый фреймворк харнесса (без его тяжёлых опциональных зависимостей — `pip --no-deps`, что удерживает образ существенно ниже лимита 4 ГБ и сборку быстрее 1000 с).

```bash
# Сборка турнирного образа
docker build -t eblo-openjiuwen-harness:latest .

# Запуск с флагами Регламента (Этапы 1–4)
docker run --rm \
  --read-only \
  --tmpfs /tmp:rw,size=2g \
  -e HOME=/tmp \
  -e LLM_BASE_URL="http://127.0.0.1:8080/v1" \
  -e LLM_API_KEY="competition-key" \
  -e TASKS="/data/tasks" \
  -e OUT="/data/tasks/summary_out.json" \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --cpus 8 \
  --memory 16g \
  --pids-limit 512 \
  -v /path/to/tasks:/data/tasks:rw \
  --workdir /app \
  --entrypoint python \
  eblo-openjiuwen-harness:latest /app/run_agent.py --tasks-dir /data/tasks
```

Локальный запуск без Docker:

```bash
PYTHONPATH=. python run_agent.py --tasks-dir <путь к задачам> --out summary_out.json \
    --workers 5 [--budget-limit 20000]
```

## 2. Алгоритм оценки расхода токенов (`harness/estimator.py` & `harness/budget_ledger.py`)

1. **Точная формула эффективного бюджета ($C_{\text{eff}}$):**
   Все вызовы к прокси (`$LLM_BASE_URL`) перехватываются в `BudgetLedger` и учитываются по формуле соревнования:
   $$C_{\text{eff}} = T_{\text{completion}} + T_{\text{reasoning}} + 0.1 \cdot T_{\text{prompt}}$$
2. **Zero-Token сканер 100 задач (`harness/task_scanner.py`):**
   За $\approx 0.02$ с (лимит $< 1.5$ с) обходит все папки в `--tasks-dir`, игнорируя служебный мусор macOS (`.DS_Store`, `__MACOSX`, `._*`), и извлекает 21 статический признак $\mathbf{x}_i$: домен (`math`, `coding`, `data_analysis`, `search`), `reward`, длину условия, число формул/вопросов, цикломатическую сложность AST `mylib.py`, число ассертов в `test_mylib.py`, схему и число битых строк в `sales.csv`, а также граф сущностей `person_XX` / `company_XX` в `workspace/kb/*.md`.
3. **Предсказание $\hat{C}_i(m)$ и квантиля $\hat{C}_{i, p90}(m)$:**
   Классическая регрессионная модель (коэффициенты эволюционно оптимизированы через EBLO и сохранены в `harness/evolved_artifacts/best_policy.json`) предсказывает ожидаемый расход и верхний квантиль $p_{90}$, который устанавливается как индивидуальный предохранитель (`EarlyStopException`) в `BudgetLedger`.
4. **Портфельное планирование (Stochastic Knapsack ROI, `harness/scheduler.py`):**
   Очередь задач ранжируется по ожидаемой отдаче баллов на единицу эффективного бюджета:
   $$\text{Score}_i = \frac{\text{reward}_i \cdot \hat{P}_i(\text{feasible})^{\gamma} \cdot \hat{P}_i(\text{success} \mid m^*)}{\hat{C}_{i, p90}(m^*) + C_{\text{overhead}}}$$
   Если в `--tasks-dir` обнаружено $\le 5$ задач (Этап 3 — отборочный тур из 2 задач), автоматически включается `qualifying_mode=True`, отключающий отсев и выделяющий повышенный резерв бюджета для 100% решения обеих задач.

## 3. Трехуровневая детекция невыполнимых задач (`harness/unfeasible_filter.py`)

Для защиты бюджета от 15–25% невыполнимых задач-ловушек реализован трехуровневый фильтр:
- **Рубеж 0 (Статический Zero-Token фильтр, 0 токенов):**
  - В `search`: проверяет присутствие запрашиваемых сущностей (`person_XXXX`, `company_XXXX`) в локальном индексе `workspace/kb/*.md` и выявляет числовые противоречия для одного и того же атрибута сущности в разных файлах.
  - В `data_analysis`: проверяет наличие запрашиваемых в `task.json` столбцов в заголовках `sales.csv` и непустоту таблицы.
  - В `coding`: выполняет AST-разбор `test_mylib.py` и выявляет синтаксически разрушенные тесты или взаимно противоречащие ассерты вида `f(x) == A` и `f(x) == B`.
  - В `math`: детектирует несовместные линейные инварианты и парадоксы четности.
- **Рубеж 1 (Дешевый Probe-запрос, `qwen3.6-35b-a3b-noreason`, `max_tokens=120`):**
  Запускается для пограничных задач с высоким `reward`, классифицируя сводку задачи как `SOLVABLE` или `UNSOLVABLE_TRAP`.
- **Рубеж 2 (In-Flight контроль деградации):**
  Если при решении задачи в `/tmp` два последовательных запуска тестов не уменьшают число ошибок или обращаются к отсутствующим ресурсам `workspace/`, сессия немедленно прерывается (`UNFEASIBLE`).

## 4. Маршрутизация моделей, доменные солверы и отказоустойчивость

- **Маршрутизация 4 моделей (`harness/model_router.py`):**
  По умолчанию для первой попытки используются модели без рассуждений (`qwen3.6-35b-a3b-noreason` и `qwen3.8-27b-noreason`), что экономит в 4–8 раз больше эффективных токенов за счет $T_{\text{reasoning}} = 0$. Переход на рассуждающие модели (`qwen3.6-35b-a3b`, `qwen3.8-27b`) выполняется только при провале локальной проверки в `/tmp` на задачах с высоким `reward`.
- **Перенос вычислений в `/tmp`:**
  - `search_solver.py`: строит локальный индекс BM25 + 2-шаговый граф связей по `workspace/kb/*.md` и отправляет в LLM только топ-5 релевантных сниппетов ($\sim 500$ токенов вместо всех 30 файлов).
  - `data_solver.py`: считывает локально только заголовок и 3 строки `sales.csv`, генерирует короткий Python-скрипт агрегации, исполняет его в `/tmp` на полном CSV и записывает проверенный результат.
  - `coding_solver.py`: копирует `workspace/` в `/tmp/work_<task_id>`, запускает тесты, подает в модель функцию и стектрейс, применяет патч и записывает `answer.txt` **только после успешного прохождения `test_mylib.py`**.
  - `math_solver.py`: генерирует и исполняет проверочный Python-скрипт в `/tmp`.
- **Отказоустойчивость (`harness/jiuwen_agent.py`):**
  - *Фолбэк моделей:* ответ 403/404 вида «модель недоступна на тарифе» блокирует только эту модель и прозрачно подменяет её ближайшей доступной из четырёх разрешённых (порядок: от дешёвых к дорогим), не прерывая прогон.
  - *429 rate-limit:* до 3 повторов с экспоненциальным backoff (1.5→3→6 с, потолок 8 с) с уважением заголовка `Retry-After`; в реальном прогоне это спасло 33 вызова.
  - *Сетевые сбои:* 3 подряд connection-error/timeout → честная остановка `proxy_exhausted` вместо часа таймаутов.
  - *JSONL-лог прогона* (`<out>.log`): каждый LLM-вызов (токены по типам, модель), каждый ретрай, исход каждой задачи и финальная причина остановки (`stop_reason`: `queue_completed` / `global_budget_exhausted` / `proxy_exhausted`) — прогон никогда не умирает молча.
- **Атомарная запись (`Atomic Persistence`):**
  Каждое проверенное решение мгновенно записывается через временный файл + `os.fsync()` + `os.replace()` в `/data/tasks/<task_id>/answer.txt` и синхронно обновляет `$OUT` под `asyncio.Lock`.

## 5. Оффлайн-эволюция EBLO и калибровка симулятора

Симулятор турнира (`eblo/competition_case/tournament_sim.py`) воспроизводит поведение прокси: классы сложности задач, коррелированные ошибки модели (ретрай той же моделью не даёт нового шанса — только эскалация), логнормальный шум токенов, зашумлённый Tier-1 probe. Таблицы симулятора калибруются по реальному прогону:

```bash
# 1. Сгенерировать обучающий турнир (100 задач, seed 2026)
PYTHONPATH=. python tools/gen_training_set.py

# 2. Реальный прогон против живого прокси (пишет real_run_out.json + .log)
PYTHONPATH=. python run_agent.py --tasks-dir train_tasks_100 --out real_run_out.json \
    --workers 5 --budget-limit 200000

# 3. Калибровка симулятора по факту прогона -> eblo/competition_case/sim_calibration.json
PYTHONPATH=. python tools/calibrate_sim.py

# 4. Эволюция политики на откалиброванном симуляторе -> harness/evolved_artifacts/best_policy.json
PYTHONPATH=. python -m eblo.competition_case.export_to_harness
```

Текущая эволюционированная политика (`gen2_cand_3`, средний фитнес по бюджетам 6K/15K/40K): решает **71/80** выполнимых задач и отсеивает **20/20** ловушек против 68/80 у базовой политики; бюджетные испытания в зоне дефицита подтверждают корректность приоритизации.

## 6. Тесты

```bash
# 1. Отборочный этап (Этап 3: 2 задачи, 100% решение)
PYTHONPATH=. python3 tests/test_qualifying_two.py

# 2. Полный турнир (Этап 4: 100 задач, 4 домена, 20% ловушек, мусор macOS)
PYTHONPATH=. python3 tests/test_full_100_sim.py

# 3. Фолбэк моделей (403 free-tier) и ретраи 429
PYTHONPATH=. python3 tests/test_model_fallback.py

# 4. Базовый пайплайн EBLO (TSPTW + каскадный валидатор + AST-кэш)
PYTHONPATH=. python3 tests/test_pipeline.py
```

Все тесты детерминированы (`accuracy_scale=1e9, noise_sigma=0.0`) и проходят без сети.
