# EBLO + `openJiuwen` Tournament Harness (Budget-Aware Token Estimation for Agentic AI)

Двухконтурная система, объединяющая оффлайн-эволюционный оркестратор **EBLO (Evolutionary Blazing-fast LLM Orchestrator)** и боевой турнирный харнесс на базе **`openJiuwen` (`openjiuwen==0.1.19`)** для соревнования **Budget-Aware Token Estimation for Agentic AI**.

## 1. Сборка и запуск Docker-контейнера

Контейнер спроектирован на базе `python:3.11.9-slim-bookworm` с жестко зафиксированными версиями пакетов в `requirements.txt` (`openjiuwen==0.1.19`, `httpx==0.23.3`, `pydantic==1.10.4`, `numpy==2.1.3`, `scikit-learn==1.5.2`) и полностью совместим с ограничениями жюри (`--read-only`, `--tmpfs /tmp:rw,size=2g`, `-e HOME=/tmp`, `--cap-drop ALL`, `--security-opt no-new-privileges`, `--cpus 8`, `--memory 16g`, `--pids-limit 512`):

```bash
# Сборка турнирного образа (< 4 ГБ)
docker build -t eblo-openjiuwen-harness:latest .

# Запуск в полном соответствии с флагами Регламента (Этапы 1–4)
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

## 2. Алгоритм оценки расхода токенов (`harness/estimator.py` & `harness/budget_ledger.py`)

1. **Точная формула эффективного бюджета ($C_{\text{eff}}$):**
   Все вызовы к прокси (`$LLM_BASE_URL`) перехватываются в `BudgetLedger` и учитываются по формуле соревнования:
   $$C_{\text{eff}} = T_{\text{completion}} + T_{\text{reasoning}} + 0.1 \cdot T_{\text{prompt}}$$
2. **Zero-Token сканер 100 задач (`harness/task_scanner.py`):**
   За $\approx 0.02\text{ с}$ (лимит $< 1.5\text{ с}$) обходит все папки в `--tasks-dir`, игнорируя служебный мусор macOS (`.DS_Store`, `__MACOSX`, `._*`), и извлекает 21 статический признак $\mathbf{x}_i$: домен (`math`, `coding`, `data_analysis`, `search`), `reward`, длину условия, число формул/вопросов, цикломатическую сложность AST `mylib.py`, число ассертов в `test_mylib.py`, схему и число битых строк в `sales.csv`, а также граф сущностей `person_XX` / `company_XX` в `workspace/kb/*.md`.
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

## 4. Анализ траекторий, выбор моделей и доменные солверы (`harness/domain_solvers/`)

- **Маршрутизация 4 моделей (`harness/model_router.py`):**
  По умолчанию для первой попытки используются модели без рассуждений (`qwen3.6-35b-a3b-noreason` и `qwen3.8-27b-noreason`), что экономит в 4–8 раз больше эффективных токенов за счет $T_{\text{reasoning}} = 0$. Переход на рассуждающие модели (`qwen3.6-35b-a3b`, `qwen3.8-27b`) выполняется только при провале локальной проверки в `/tmp` на задачах с высоким `reward`.
- **Перенос вычислений в `/tmp`:**
  - `search_solver.py`: строит локальный индекс BM25 + 2-шаговый граф связей по `workspace/kb/*.md` и отправляет в LLM только топ-5 релевантных сниппетов ($\sim 500$ токенов вместо всех 30 файлов).
  - `data_solver.py`: считывает локально только заголовок и 3 строки `sales.csv`, генерирует короткий Python-скрипт агрегации, исполняет его в `/tmp` на полном CSV и записывает проверенный результат.
  - `coding_solver.py`: копирует `workspace/` в `/tmp/work_<task_id>`, запускает тесты, подает в модель функцию и стектрейс, применяет патч и записывает `answer.txt` **только после успешного прохождения `test_mylib.py`**.
  - `math_solver.py`: генерирует и исполняет проверочный Python-скрипт в `/tmp`.
- **Атомарная запись (`Atomic Persistence`):**
  Каждое проверенное решение мгновенно записывается через временный файл + `os.fsync()` + `os.replace()` в `/data/tasks/<task_id>/answer.txt` и синхронно обновляет `$OUT` под `asyncio.Lock`.

## 5. Запуск тестов и оффлайн-эволюции EBLO

```bash
# 1. Отборочный этап (Этап 3: 2 задачи, 100% решение)
PYTHONPATH=. python3 tests/test_qualifying_two.py

# 2. Полный турнир (Этап 4: 100 задач, 4 домена, 20% ловушек, мусор macOS)
PYTHONPATH=. python3 tests/test_full_100_sim.py

# 3. Базовый пайплайн EBLO (TSPTW + каскадный валидатор + AST-кэш)
PYTHONPATH=. python3 tests/test_pipeline.py

# 4. Оффлайн-эволюция политики харнесса и экспорт в harness/evolved_artifacts/best_policy.json
PYTHONPATH=. python3 -m eblo.competition_case.export_to_harness
```
