FROM python:3.7.9-slim-buster

LABEL maintainer="capstone-reproduction"

# NOTE vs official Dockerfile: the apt-get update/upgrade step is dropped because
# Debian buster is EOL and its apt repos are archived (build fails on exit 100).
# The pinned scientific wheels install from PyPI without needing apt/build tools.

VOLUME ["/data", "/results"]
WORKDIR /app

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt; \
    rm -rf /tmp/* /var/tmp/* ~/.cache/pip

ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "execute-algorithm" ]
