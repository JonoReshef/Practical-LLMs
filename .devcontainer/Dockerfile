# Dev Container Dockerfile for LLM-Explainer
# Ubuntu-based container for a fully interactive development environment
# Provides isolation from host filesystem with full CLI capabilities
# Note: MLX acceleration is NOT available in containers (requires Apple Silicon hardware)

FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# uv configuration for container environment
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

# Install system dependencies and development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core utilities
    git \
    curl \
    wget \
    ca-certificates \
    gnupg \
    # Build essentials for Python packages
    build-essential \
    # Useful CLI tools
    vim \
    nano \
    htop \
    tree \
    jq \
    ripgrep \
    fd-find \
    bat \
    fzf \
    tmux \
    zsh \
    # Networking tools
    net-tools \
    iputils-ping \
    dnsutils \
    # Python build dependencies
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (LTS) for Claude Code CLI
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Python 3.14 via uv
RUN uv python install 3.14

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Create non-root user with sudo access
# Ubuntu 24.04 has an existing 'ubuntu' user with UID/GID 1000, so we modify it
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update && apt-get install -y sudo \
    && (userdel -r ubuntu 2>/dev/null || true) \
    && (groupdel ubuntu 2>/dev/null || true) \
    && (getent group $USER_GID || groupadd --gid $USER_GID $USERNAME) \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/zsh $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Set up user directories
RUN mkdir -p /home/$USERNAME/.local/bin \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME

# Set up workspace directory
WORKDIR /workspaces/LLM-explainer

# Switch to non-root user
USER $USERNAME

# Add local bin and uv-managed Python to PATH
ENV PATH="/home/$USERNAME/.local/bin:$PATH"

# Set default shell to zsh
ENV SHELL=/bin/zsh

# Create symlinks for common tools with different names
RUN sudo ln -sf /usr/bin/batcat /usr/local/bin/bat \
    && sudo ln -sf /usr/bin/fdfind /usr/local/bin/fd
