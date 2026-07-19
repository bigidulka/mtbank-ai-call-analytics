# Конкурентный benchmark: безопасный статический контур

## Назначение и границы

Этот контур сравнения не исполняет чужой код. Он не делает `clone`, не импортирует пакеты конкурентов, не запускает их тесты, Dockerfile, бинарные файлы или сетевые сервисы. По умолчанию `scripts/analyze_competitors.py` читает только frozen manifest и выдаёт `unknown`; статический разбор возможен лишь для явно переданных, заранее полученных исходников в отдельном credentials-free окружении.

Когорта в `evals/competitors/manifest.yaml` — **историческая реконструкция**, а не заявка на точный снимок GitHub на момент исходного поиска. Исходный запрос был `q=mtbank&sort=updated&order=desc`; пользовательский transcript от `2026-07-15T11:37:13.074Z` сообщил 44 результата. Позднее публичный поиск вернул 45: `vbuyel/mtbank-ai-hiring` создан `2026-07-15T12:14:03Z`, после исходного запроса, поэтому исключён.

В manifest намеренно записано `checked_at_utc: null`: транспорт исследования не передал надёжное время HTTP/локальных часов. SHA — наблюдения текущего GitHub REST, не доказательство SHA в исторический момент. `AlexeyShakal/MTBank` имеет подтверждённый статус пустого репозитория и не имеет immutable commit. Архивные `MarkCesium/mtbank_hackathon` и `skazmasters/mtbank` остаются в когорте. Валидатор закрепляет порядок и полное содержание 44 записей через canonical list и SHA-256 digest; исключение `vbuyel/mtbank-ai-hiring` также имеет точный frozen record.

Текущий проект не включён в эти 44 записи. `--candidate-root` делает Git-inspection только для фиксированного локального корня этого проекта с отключёнными hooks/fsmonitor и sanitized Git environment. Произвольный root получает `untrusted_candidate_requires_explicit_identity`; его исходники можно только статически прочитать, но нельзя получать Git identity. Dirty trusted working tree получает `commit_status: uncommitted` и не получает immutable SHA.

## Воспроизводимый статический запуск

```bash
python scripts/analyze_competitors.py
python scripts/analyze_competitors.py --sources-dir /safe/prefetched --candidate-root . --output artifacts/competitive-analysis.json
```

Каталог `/safe/prefetched` должен содержать только явно подготовленные директории `owner__repository`, извлечённые по SHA из manifest. Сканер пропускает symlink, submodule, `.git`, LFS pointer, binary/non-UTF-8 и файл более 1 MiB; также ограничивает дерево 5 000 подходящими файлами. Для каждого совпадения он записывает `{repo, sha, path, line, rule_id, excerpt_hash, status}`, а не содержание строки.

Любой prose-файл (`.md`, `.rst`, `.txt`) и любой путь в `docs/` имеют `claim_only` для всех правил, включая implementation, tests, security и bonuses; они не являются проверенным баллом. Репозиторий игры или нерелевантный репозиторий не получает ноль: его применимость `unknown`/`out_of_scope`, пока не появится доказательство предметной релевантности.

## Рубрика и состояние отчёта

`evals/competitors/rubric.yaml` определяет единый максимум 100 + 15 бонусных баллов для конкурентов и кандидата. Проверяемые сигналы охватывают Pipeline, вложения, ASR, diarization/roles, независимые LLM/tool trajectories, API/Compose, security/privacy, persistence, observability, тесты, документацию и бонусы. Частичные баллы возможны только для пар независимых сигналов, по явно заданному правилу rubric. Документация не получает static verified points: prose остаётся claim-only, а независимая аттестация нужна для её учёта.

`unknown`, `claim_only` и `out_of_scope` не равны нулю и не добавляют verified points. Даже накопленные `verified_points_observed` не являются рейтингом. `comparative_score` остаётся `null`, пока все применимые критерии не подтверждены статически/аттестацией, а кандидат не имеет одновременно immutable release SHA и доказательства образа. Поэтому даже при безопасно доступных исходниках все записи сохраняют `score_status: unknown`; недоступный или отклонённый архив не доказывает отсутствие реализации.

## Обновление метаданных и ограничения

`scripts/fetch_competitor_metadata.py` не делает сетевых запросов без явного `--refresh`. Он использует только публичный GitHub REST, не читает и не печатает токены. По умолчанию он только показывает diff; изменение frozen manifest требует одновременно `--refresh --write-reconstruction` и ручного review diff.

Workflow `.github/workflows/competitive-benchmark.yml` выполняет только metadata-only запрос и manifest-only static report по расписанию/вручную. Он не может менять когорту, не получает write permissions и не исполняет код репозиториев. Ограничения этого benchmark: API metadata могут измениться, поиск не доказывает предметную релевантность, статический анализ не доказывает runtime-поведение, а отсутствие безопасно доступного архива остаётся unknown.

## Финальный запуск frozen cohort

19 июля 2026 года выполнен `static-only` запуск rubric для 44 frozen записей и локального кандидата. Из 43 записей с immutable SHA безопасно извлечены bounded UTF-8 text files для 38; пять codeload-архивов отклонены после превышения лимита 25 MiB (`fedosikser/mtbank_coop`, `weblov33/mtbank_coop`, `Fuz483/MTBankGame`, `marinchi03/mtbank`, `skazmasters/mtbank`). `AlexeyShakal/MTBank` не имеет immutable commit. Из записей cohort анализатор завершил scan с verified source signals для 21, без verified signals — для 17, `source_not_provided` — для 6 (пять отклонённых архивов и empty repository).

При извлечении принимались только regular files допустимых text-типов: archive ≤25 MiB, file ≤1 MiB, text tree ≤16 MiB, ≤10 000 archive members и ≤5 000 text files. Symlink, hardlink, device, traversal, binary/non-UTF-8 и LFS pointer не извлекались. Код конкурентов не клонировался, не импортировался, не собирался и не исполнялся.

Candidate использует immutable SHA `18c81880b0b2ef318eb0c00ec3e9020381678e63` и дал 94 `verified_points_observed` и 10 bonus points при 280 просмотренных подходящих файлах (`commit_status: clean_git_sha`). Его `comparative_score` остаётся `null`, а `score_status` — `unknown` исключительно потому, что отсутствуют release image evidence и release attestation, а не из-за uncommitted tree. У всех 44 competitors `comparative_score` также `null`: observed points — не ранжирование, а отсутствие безопасного архива остаётся `unknown`.

Machine-readable final report: `artifacts/competitive-analysis-final.json`.

## Актуальный targeted comparison AI-решений

17 июля 2026 года отдельно проверены восемь актуальных AI-assignment repositories из верхней части GitHub search и текущий локальный working tree. Источники были получены по immutable commit SHA как bounded UTF-8 text-only archives: symlink, binary, LFS, `.git`, файлы более 1 MiB и лишние metadata не извлекались. Код конкурентов не устанавливался и не исполнялся.

Это не итоговые баллы задания. `verified_points_observed` означает только найденные независимые static signals; документация не получает баллы без attestation, runtime claims не доказываются regex-сканером, а `comparative_score` остаётся `null`.

| Репозиторий | SHA | Static verified | Static bonus | Основные не найденные signals |
|---|---|---:|---:|---|
| **candidate** | `18c81880b0b2ef318eb0c00ec3e9020381678e63` (`clean_git_sha`) | **94** | **10** | release image evidence и release attestation отсутствуют; score остаётся `null` |
| `devAsmodeus/mtbank-ai-hiring` | `e170ccd` | 83 | 10 | полная tool trajectory, вторая половина security/privacy |
| `JustiZzZz/mtbank-ai-transcription` | `31b6565` | 79 | 10 | role-resolution половина, tool trajectory, Compose половина |
| `ib0gdan/speech-analytics` | `3e8a742` | 75 | 10 | tool trajectory, privacy половина, persistence |
| `vbuyel/mtbank-ai-hiring` | `24bc3b9` | 67 | 10 | полная tool trajectory, privacy половина, persistence, observability |
| `antonsokol1542-beep/MTBank-AI-Engineer-` | `01862b1` | 67 | 10 | полная tool trajectory, privacy половина, persistence, observability |
| `Carcajo/mtbank-ai-call-analytics` | `101518c` | 62 | 10 | role-resolution половина, tool trajectory, privacy половина, persistence, observability |
| `PchelentsovRoman/mtbank-test-assignment` | `6933cac` | 0 | 0 | source implementation signals не найдены |
| `kutsydanil/AI-Contact-Analytics` | `34c410b` | 0 | 0 | source implementation signals не найдены |

Machine-readable result: `artifacts/competitive-analysis-current.json`.

Удалённый `vbuyel/mtbank-ai-hiring` отражает commit `24bc3b9`, а не candidate `18c81880b0b2ef318eb0c00ec3e9020381678e63`. Candidate уже идентифицирован этим immutable SHA (`commit_status: clean_git_sha`) и прошёл ту же rubric в static-only контуре. Итоговый рейтинг по-прежнему нельзя объявлять исключительно из-за отсутствующих release image evidence и release attestation, а не из-за uncommitted tree: `comparative_score` остаётся `null`, `score_status` — `unknown`.
