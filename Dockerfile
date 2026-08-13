# Step 1: Build the Swift App natively in the cloud
FROM swift:latest AS builder
WORKDIR /app
COPY . .
RUN swift build -c release

# Step 2: Use an official image that already has ttyd built-in
FROM tsl0922/ttyd:latest
WORKDIR /app

# Install Swift system runtime dependencies
RUN apt-get update && apt-get install -y \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy your compiled binary from Step 1
COPY --from=builder /app/.build/release/BeeNeuralNetSim .

# Expose port 8080 for web traffic
EXPOSE 8080

# Run the app 24/7 inside the web terminal
CMD ["ttyd", "-p", "8080", "./BeeNeuralNetSim"]
