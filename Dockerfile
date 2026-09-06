FROM python:3.12-slim-bookworm

WORKDIR /app

# Define so the script knows not to download a new driver version, as
# this Docker image already downloads a compatible chromedriver
ENV AUTO_SOUTHWEST_CHECK_IN_DOCKER=1
ENV AUTO_SOUTHWEST_CHECK_IN_BROWSER_PATH=/usr/bin/chromium

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        chromium \
        chromium-driver \
        fonts-liberation \
        tzdata \
        xauth \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --home-dir /app auto-southwest-check-in
RUN chown -R auto-southwest-check-in:auto-southwest-check-in /app

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
RUN chown -R auto-southwest-check-in:auto-southwest-check-in \
    /usr/local/lib/python3.12/site-packages/seleniumbase/drivers

COPY . .
RUN chown -R auto-southwest-check-in:auto-southwest-check-in /app
USER auto-southwest-check-in

ENTRYPOINT ["python3", "-u", "southwest.py"]
