# n8n → InfoM GraphRAG

## Установка n8n (self-hosted, бесплатно)

```bash
# Docker (рекомендуется)
docker run -p 5678:5678 n8nio/n8n

# или через npx
npx n8n
```

## Workflow 1: Telegram бот → граф → ответ

```
[Telegram Trigger]
    message.text → /index <текст>  или  /ask <вопрос>
    ↓
[IF: команда index?]
    ├── YES →
    │   [HTTP Request]
    │   POST https://infom-api.railway.app/index
    │   body: {"text": "{{ $json.message.text.replace('/index ', '') }}"}
    │   ↓
    │   [Telegram: Send Message]
    │   text: "Проиндексировано ✓ — {{ $json.result }}"
    │
    └── NO →
        [HTTP Request]
        POST https://infom-api.railway.app/query
        body: {
          "question": "{{ $json.message.text.replace('/ask ', '') }}",
          "mode": "hybrid"
        }
        ↓
        [Telegram: Send Message]
        text: "{{ $json.result }}"
```

## Workflow 2: GitHub webhook → индексация README

```
[Webhook Trigger] ← GitHub push event
    ↓
[HTTP Request: получить README]
    GET https://raw.githubusercontent.com/{{owner}}/{{repo}}/main/README.md
    ↓
[HTTP Request: индексировать]
    POST /index
    body: {"text": "{{ $json.body }}", "reset": false}
    ↓
[HTTP Request: анализ]
    POST /query
    body: {"question": "Что изменилось в проекте?", "mode": "global"}
    ↓
[Slack: уведомление]
    message: "Обновление {{ $json.repo }}: {{ $json.result }}"
```

## Workflow 3: Расписание — ночной дайджест

```
[Cron: каждую ночь в 23:00]
    ↓
[HTTP Request: статистика]
    GET /stats
    ↓
[HTTP Request: глобальный запрос]
    POST /query
    body: {"question": "Что самое важное в графе знаний сегодня?", "mode": "global"}
    ↓
[Gmail / Outlook: отправить письмо]
    subject: "InfoM дайджест {{ $now }}"
    body: "Граф: {{ $('stats').item.json.result }}\n\nВыводы: {{ $json.result }}"
```

## Нода HTTP Request — конфигурация

```
Method:          POST
URL:             https://infom-api.railway.app/webhook
Authentication:  None (или Header Auth если добавишь API ключ)
Content Type:    JSON
Body:
  {
    "action": "{{ $json.action }}",
    "text":   "{{ $json.text }}",
    "question": "{{ $json.question }}",
    "mode":   "hybrid"
  }
```

## Code нода (JavaScript) — обработка ответа

```javascript
// Извлечь ключевые строки из result
const result = $input.item.json.result;
const lines = result.split('\n').filter(l => l.trim());
const answer = lines[2] || result.substring(0, 200);

return [{
  json: {
    answer,
    graph_nodes:       $input.item.json.graph.nodes,
    graph_modularity:  $input.item.json.graph.modularity,
    timestamp:         new Date().toISOString(),
  }
}];
```
