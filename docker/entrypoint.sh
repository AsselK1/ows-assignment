#!/bin/bash
# Entrypoint script for Kazakhstan Procurement API container
# Validates environment, performs optional migrations, then starts uvicorn

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Kazakhstan Procurement API...${NC}"

# =============================================================================
# 1. VALIDATE REQUIRED ENVIRONMENT VARIABLES
# =============================================================================

REQUIRED_VARS=(
    "CLICKHOUSE_HOST"
    "CLICKHOUSE_DB"
    "QDRANT_HOST"
)

echo -e "${YELLOW}Validating required environment variables...${NC}"
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "${RED}ERROR: Required environment variable '$var' is not set${NC}"
        exit 1
    fi
    echo "  ✓ $var=${!var}"
done

# Optional variables with defaults
CLICKHOUSE_PORT="${CLICKHOUSE_PORT:-9000}"
CLICKHOUSE_USER="${CLICKHOUSE_USER:-default}"
CLICKHOUSE_PASSWORD="${CLICKHOUSE_PASSWORD:-}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
APP_CONFIG_PATH="${APP_CONFIG_PATH:-config.yaml}"

echo "  ✓ CLICKHOUSE_PORT=${CLICKHOUSE_PORT} (default)"
echo "  ✓ CLICKHOUSE_USER=${CLICKHOUSE_USER} (default)"
echo "  ✓ QDRANT_PORT=${QDRANT_PORT} (default)"
echo "  ✓ APP_CONFIG_PATH=${APP_CONFIG_PATH} (default)"

# =============================================================================
# 2. VERIFY CONFIG FILE EXISTS
# =============================================================================

echo -e "${YELLOW}Verifying configuration file...${NC}"
if [ ! -f "$APP_CONFIG_PATH" ]; then
    echo -e "${RED}ERROR: Config file not found at $APP_CONFIG_PATH${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ Config file found: $APP_CONFIG_PATH${NC}"

# =============================================================================
# 3. OPTIONAL: TEST CLICKHOUSE CONNECTIVITY (non-blocking)
# =============================================================================

echo -e "${YELLOW}Testing ClickHouse connectivity (informational)...${NC}"
if command -v clickhouse-client &> /dev/null; then
    if clickhouse-client --host "$CLICKHOUSE_HOST" \
        --port "$CLICKHOUSE_PORT" \
        --user "$CLICKHOUSE_USER" \
        --password "$CLICKHOUSE_PASSWORD" \
        --query "SELECT 1" &> /dev/null; then
        echo -e "  ${GREEN}✓ ClickHouse is reachable${NC}"
    else
        echo -e "  ${YELLOW}⚠ ClickHouse connection test failed (will retry at startup)${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠ clickhouse-client not available (skipping connectivity test)${NC}"
fi

# =============================================================================
# 4. OPTIONAL: TEST QDRANT CONNECTIVITY (non-blocking)
# =============================================================================

echo -e "${YELLOW}Testing Qdrant connectivity (informational)...${NC}"
if command -v curl &> /dev/null; then
    if curl -sf "http://$QDRANT_HOST:$QDRANT_PORT/health" &> /dev/null; then
        echo -e "  ${GREEN}✓ Qdrant is reachable${NC}"
    else
        echo -e "  ${YELLOW}⚠ Qdrant connection test failed (will retry at startup)${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠ curl not available (skipping Qdrant connectivity test)${NC}"
fi

# =============================================================================
# 5. EXPORT ENVIRONMENT FOR UVICORN
# =============================================================================

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# =============================================================================
# 6. START UVICORN SERVER
# =============================================================================

echo -e "${GREEN}Starting uvicorn server on 0.0.0.0:8000...${NC}"
echo ""

# Run uvicorn with production settings:
# - host: 0.0.0.0 (listen on all interfaces inside container)
# - port: 8000 (standard API port)
# - workers: 1 (single worker in container; use replicas for scaling)
# - log-level: info (balanced logging)
# - loop: uvloop (if available; falls back to asyncio)
# - http: auto (uvicorn chooses best HTTP implementation)
# - access-log: enabled by default
# - use-colors: auto (respects TTY)

exec uvicorn src.api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --access-log
