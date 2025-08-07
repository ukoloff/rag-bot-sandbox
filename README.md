# Песочница GigaChat

## Образец .env
```
GIGACHAT_CREDENTIALS=XXXXXXXX
BOT_TOKEN=322233322:XXXXXXXX

```

## Установка сертификатов МинЦифры
Описана [здесь](sandbox/ruca/)

## Копирование рабочих папок с локальной машины в Docker
Контейнер `tg` уже должен существовать,
но может быть не запущен
```sh
# База данных Chroma DB
docker compose cp chroma.kb/. tg:/repo/chroma.kb/.

# Журналы работы
docker compose cp  tg:/repo/sandbox/output/. sandbox/output/.
```
