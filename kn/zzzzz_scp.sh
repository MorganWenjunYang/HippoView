#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env file not found"
    exit 1
fi

# Check if REMOTE_HOST is set
if [ -z "$REMOTE_HOST" ]; then
    echo "Error: REMOTE_HOST not set in .env file"
    exit 1
fi

# Check if REMOTE_USER is set
if [ -z "$REMOTE_USER" ]; then
    echo "Error: REMOTE_USER not set in .env file"
    exit 1
fi

# Set SSH options
SSH_OPTS=""
if [ ! -z "$SSH_PORT" ]; then
    SSH_OPTS="$SSH_OPTS -p $SSH_PORT"
fi
if [ ! -z "$SSH_KEY" ]; then
    SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
fi

# Create remote directory
ssh $SSH_OPTS ${REMOTE_HOST} "mkdir -p $REMOTE_PATH"

# Copy files to remote host
scp $SSH_OPTS kn/Dockerfile.kg ${REMOTE_HOST}:$REMOTE_PATH/Dockerfile.kg
scp $SSH_OPTS ./docker-compose.yml ${REMOTE_HOST}:$REMOTE_PATH/docker-compose.yml
scp $SSH_OPTS kn/docker-variables.env ${REMOTE_HOST}:$REMOTE_PATH/docker-variables.env
# scp $SSH_OPTS kn/kn_schema.yaml ${REMOTE_HOST}:$REMOTE_PATH/kn_schema.yaml
# scp $SSH_OPTS kn/bc_config.yaml ${REMOTE_HOST}:$REMOTE_PATH/bc_config.yaml
# scp $SSH_OPTS kn/create_kn.py ${REMOTE_HOST}:$REMOTE_PATH/create_kn.py
# scp $SSH_OPTS kn/bc_adaptor_mongodb.py ${REMOTE_HOST}:$REMOTE_PATH/bc_adaptor_mongodb.py

