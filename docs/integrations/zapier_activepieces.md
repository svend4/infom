# Zapier и Activepieces → InfoM

## Zapier

### Настройка Webhooks by Zapier

1. **Trigger:** любой (Gmail, Google Sheets, Notion, RSS, Slack...)
2. **Action:** Webhooks by Zapier → POST
3. **URL:** `https://infom-api.railway.app/webhook`
4. **Payload type:** json
5. **Data:**
```
action    | index_and_query
text      | {{trigger_field_with_text}}
question  | Кратко резюмируй и выдели главное
mode      | hybrid
```

### Пример: Gmail → граф → Google Docs

```
Trigger: Gmail - New Email
    ↓
Action 1: Webhooks - POST /webhook
    action=index, text={{Body Plain}}
    ↓
Action 2: Webhooks - POST /webhook
    action=query, question="О чём это письмо? Что важно?"
    ↓
Action 3: Google Docs - Create Document
    title=Анализ: {{Subject}}
    content={{2.result}}
```

---

## Activepieces (open-source альтернатива Make.com)

### Установка (self-hosted, бесплатно)

```bash
# Docker Compose
git clone https://github.com/activepieces/activepieces
cd activepieces
docker compose up -d
# Открыть: http://localhost:8080
```

### Flow: Любой триггер → InfoM → результат

```
[Trigger: Schedule / Webhook / App]
    ↓
[HTTP Request]
    method: POST
    url: https://infom-api.railway.app/webhook
    headers: {"Content-Type": "application/json"}
    body: {
      "action": "index_and_query",
      "text": "{{trigger.data.content}}",
      "question": "{{trigger.data.question || 'Что важно в этом тексте?'}}",
      "mode": "hybrid"
    }
    ↓
[Branch: ok == true?]
    ├── YES → [Send to Telegram/Email/Notion]
    └── NO  → [Send error notification]
```

---

## Сравнение платформ

| Платформа | Бесплатно | Self-host | Сложность | Лучше для |
|-----------|-----------|-----------|-----------|-----------|
| Make.com | 1000 оп/мес | ✗ | Низкая | Быстрый старт |
| n8n | ∞ self-host | ✓ | Средняя | Разработчики |
| Zapier | 100 задач/мес | ✗ | Низкая | Простые цепочки |
| Activepieces | ∞ self-host | ✓ | Низкая | Альтернатива Make |
| Pipedream | 300 events/день | ✗ | Средняя | JS/Python код |

---

## Пример с Pipedream (код на Python)

```python
# Pipedream Step: Python
import requests

def handler(pd: "pipedream"):
    # Индексируем
    r1 = requests.post("https://infom-api.railway.app/index", json={
        "text": pd.steps["trigger"]["event"]["body"],
        "reset": False,
    })

    # Спрашиваем
    r2 = requests.post("https://infom-api.railway.app/query", json={
        "question": "Что самое важное в этом документе?",
        "mode": "hybrid",
    })

    return {
        "indexed": r1.json(),
        "answer":  r2.json()["result"],
    }
```
