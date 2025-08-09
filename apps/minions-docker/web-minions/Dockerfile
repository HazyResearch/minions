# Multi-stage build for Advanced Minions Web Interface
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files from the correct path
COPY apps/minions-docker/web-minions/package*.json ./

# Install dependencies
RUN npm install

# Copy source files from the correct path
COPY apps/minions-docker/web-minions/ .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx configuration from the correct path
COPY apps/minions-docker/web-minions/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 8080
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8080/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
