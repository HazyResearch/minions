# Frontend Dockerfile for Minions Docker Web Interface
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY apps/minions-docker/web/package*.json ./

# Install dependencies
RUN npm install

# Copy frontend source files
COPY apps/minions-docker/web/ ./

# Expose port 8080
EXPOSE 8080

# Set environment variables
ENV NODE_ENV=development
ENV HOST=0.0.0.0
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/ || exit 1

# Start the development server
CMD ["npm", "run", "dev"]
