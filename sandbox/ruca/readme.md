# Установка сертификатов МинЦифры

https://github.com/WISEPLAT/gigachain

```
uv pip install gigachain-cli
uv run --no-sync gigachain install-rus-certs
uv sync
```
Но при этом требуется ещё `git`.
Если его нет и не нужно,
ничего не доустанавливая,
запускайте [утилитку](__main__.py)
```
uv run sandbox/ruca
```
