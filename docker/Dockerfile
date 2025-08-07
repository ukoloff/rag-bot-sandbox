FROM alpine

LABEL maintainer="Stanislav.Ukolov@omzglobal.com"

RUN --mount=type=cache,target=/var/cache/apk apk add git
COPY bin/* /bin/
