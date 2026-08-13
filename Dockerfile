# Step 1: Use the official Swift Linux environment to build your package
FROM swift:latest AS builder
WORKDIR /app
COPY . .
RUN swift build -c release

# Step 2: Set up the live running environment
FROM ubuntu:22.04
WORKDIR /app

# Install system dependencies and ttyd
RUN apt-get update && apt-get install -y \
    curl \
    libsqlite3-0 \
    && curl -LO https://github.com \
    && chmod +x ttyd.x86_64 \
    && mv ttyd.x86_64 /usr/local/bin/ttyd

# Copy your freshly compiled binary from Step 1
COPY --from=builder /app/.build/release/BeeNeuralNetSim .

# Expose port 8080 for Render web traffic
EXPOSE 8080

# Tell the server to launch ttyd and your real ML app on start
CMD ["ttyd", "-p", "8080", "./BeeNeuralNetSim"]
