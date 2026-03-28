# Make.com → InfoM GraphRAG

## Сценарий 1: Notion → граф → Slack

```
[Notion: новая страница]
    ↓
[HTTP: POST /webhook]
    body: {
      "action": "index",
      "text": "{{Notion.content}}",
      "reset": false
    }
    ↓
[HTTP: POST /webhook]
    body: {
      "action": "query",
      "question": "Какие ключевые идеи в этом документе?",
      "mode": "hybrid"
    }
    ↓
[Slack: отправить сообщение]
    text: "{{2.result}}"
```

## Сценарий 2: RSS лента → индексация → Email дайджест

```
[RSS: новые статьи (каждые 6 часов)]
    ↓
[Iterator: для каждой статьи]
    ↓
[HTTP: POST /index]
    body: {"text": "{{title}} {{description}}", "reset": false}
    ↓
[Aggregator: собрать все статьи]
    ↓
[HTTP: POST /query]
    body: {"question": "Что нового в технологиях сегодня?", "mode": "global"}
    ↓
[Email: отправить дайджест]
    subject: "Дайджест {{formatDate(now; 'DD.MM.YYYY')}}"
    body: "{{result}}"
```

## Сценарий 3: Google Sheets → анализ → запись обратно

```
[Google Sheets: Watch rows (новые строки)]
    ↓
[HTTP: POST /index]
    body: {"text": "{{A}} {{B}} {{C}}", "reset": false}
    ↓
[HTTP: POST /query]
    body: {
      "question": "Классифицируй этот элемент",
      "mode": "local"
    }
    ↓
[Google Sheets: Update row]
    column D: "{{result}}"
```

## Настройка HTTP модуля в Make.com

| Поле | Значение |
|------|---------|
| URL | `https://infom-api.railway.app/webhook` |
| Method | POST |
| Headers | `Content-Type: application/json` |
| Body type | Raw |
| Content type | JSON (application/json) |

## Универсальный action=index_and_query (один запрос)

```json
{
  "action": "index_and_query",
  "text": "{{входящий_текст}}",
  "question": "Кратко резюмируй ключевые идеи",
  "mode": "hybrid",
  "reset": false,
  "meta": {
    "source": "notion",
    "page_id": "{{Notion.id}}"
  }
}
```

Ответ:
```json
{
  "ok": true,
  "action": "index_and_query",
  "result": "...",
  "elapsed": 1.23,
  "graph": {"nodes": 47, "communities": 6, "modularity": 0.42}
}
```
