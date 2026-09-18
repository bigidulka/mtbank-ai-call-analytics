# Architecture

Исходное задание сохранено без изменений в [assignment.md](assignment.md). Принятые
границы описаны в ADR `0001`–`0004` в [adr/](adr/).

```mermaid
flowchart LR
    Browser[Браузер] --> Gateway[nginx gateway]
    Gateway --> OpenWebUI[OpenWebUI]
    Gateway --> Pipelines[OpenWebUI Pipelines]
    Pipelines --> API
    subgraph application["application-internal"]
        API[FastAPI API] --> Speech[speech service]
        API --> Postgres[(PostgreSQL)]
        Speech --> Postgres
        API -->|HTTPS + bearer| ModelGateway[OpenAI-compatible gateway]
    end
    subgraph monitoring["monitoring-internal"]
        OTel[OpenTelemetry Collector]
        Prometheus[Prometheus]
        Tempo[Tempo]
        Grafana[Grafana]
    end
    API -.-> OTel
    OTel -.-> Prometheus
    OTel -.-> Tempo
    Grafana -.-> Prometheus
    Grafana -.-> Tempo
```

Последовательность одного анализа:

```mermaid
sequenceDiagram
    participant P as OpenWebUI Pipeline
    participant A as FastAPI API
    participant W as AnalysisWorkflow
    participant S as Speech service
    participant R as Agent runtime
    participant D as PostgreSQL
    P->>A: upload WAV/MP3/OGG (bearer + attachment envelope)
    A->>W: analyze(FileAnalyzeInput)
    W->>S: normalize → transcribe → diarize → resolve roles
    S-->>W: TranscriptSnapshot (сегменты, роли, revisions)
    par четыре агента параллельно
        W->>R: classifier
        W->>R: quality
        W->>R: compliance
        W->>R: summarizer
    end
    R-->>W: typed terminal outputs (evidence = segment IDs)
    W->>W: deterministic aggregation + grounding checks
    W->>D: sanitized analysis record
    W-->>A: AnalyzeResponse
    A-->>P: JSON / markdown для чата
```

`gateway` — единственный host binding; API, speech, PostgreSQL и monitoring не
публикуют порт хоста. Text chat проходит `OpenWebUI Pipeline generator → authenticated
/assistant/stream SSE → bounded model/tool loop → provider stream`; наружу выходят
только safe progress и final-answer deltas. Buffered `/assistant` собирает тот же stream.
Один shared speech workflow используется REST и OpenWebUI Pipeline:
безопасная загрузка/normalization → canonical speech → четыре bounded agents
(`classifier`, `quality`, `compliance`, `summarizer`) → deterministic aggregation →
sanitary persistence.

Evidence хранит версии кода, policies/prompts, dataset, speech-компонентов и моделей,
а также hashes. Он не предназначен для хранения аудио, transcript, prompt, provider
response или ключей. Текущая архитектура не доказывает release readiness: реальный
cloud E2E, local model artifacts, GPU и Grafana evidence остаются gates из
[release-checklist.md](release-checklist.md).
