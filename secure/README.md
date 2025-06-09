# Secure Minions Documentation

This guide covers two secure communication protocols in the Minions ecosystem:

1. **Secure Minions Chat** — An end-to-end encrypted chat system using confidential VMs
2. **Secure Minions Local-Remote Protocol** — A secure implementation of the Minions protocol with local-remote model collaboration

Both protocols provide end-to-end encryption, attestation verification, and secure communication channels.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Secure Minions Chat](#secure-minions-chat)
  - [Overview](#overview)
  - [Method 1: Connect to Hosted Server](#method-1-connect-to-hosted-secure-minions-chat-server)
  - [Method 2: Setup Your Own Server](#method-2-setup-your-own-secure-minions-chat-server)
- [Secure Minions Local-Remote Protocol](#secure-minions-local-remote-protocol)
  - [Overview](#overview-1)
  - [Features](#features)
  - [Basic Usage](#basic-usage)
  - [Advanced Usage](#advanced-usage)
  - [Security Features](#security-features)
  - [Configuration Options](#configuration-options)

## Prerequisites

Install the secure dependencies for both protocols:

```bash
pip install -e ".[secure]"
```

---

# Secure Minions Chat

## Overview

Secure Minions Chat provides an end-to-end encrypted chat system using confidential VMs on Azure and the Minions project. The system combines:

1. **Remote Inference Server** — securely running on a confidential NVIDIA H100 GPU VM
2. **Local Chat Client** — running on your local machine, connected to the remote secure server

There are two ways to use the secure chat:

- **Method 1**: Connect to an existing secure Minions Chat remote server
- **Method 2**: Setup your own secure Minions Chat remote server

## Method 1: Connect to Hosted Secure Minions Chat Server

This section walks you through installing and running the Minions chat client locally to connect with an existing secure remote inference server.

### 1. Clone the Minions Repository

```bash
git clone https://github.com/HazyResearch/minions.git
cd minions
```

### 2. Install Minions Locally

Install in editable mode from the top level directory:

```bash
pip install -e ".[secure]"
```

### 3. Request Access to the Attestation Public Key

Please fill out this form and we will email you the key: https://forms.gle/21ZAH9NqkehUwbiQ7.

### 4. Launch Secure Chat (Command Line)

Connect to the hosted server:

```bash
python secure/minions_chat.py --supervisor_url "https://minions-tee.eastus2.cloudapp.azure.com:443" --trusted_attesation_pem <path to pem file>
```

### 4. Launch the Streamlit Chat App (Web UI)

Run the visual chat interface via Streamlit:

```bash
streamlit run minions_secure_chat.py
```

> **Note**: In the app, configure the **Supervisor URL** as `https://minions-tee.eastus2.cloudapp.azure.com:443` in the sidebar before submitting messages.

## Method 2: Setup Your Own Secure Minions Chat Server

### Part 1: Remote Inference Server Setup

#### 1. Provision a Confidential VM + Secure GPU on Azure

Follow [Azure CGPU onboarding documentation](https://github.com/Azure/az-cgpu-onboarding/blob/main/docs/Confidential-GPU-H100-Manual-Installation-%28PMK-with-Powershell%29.md) to launch a confidential VM with H100 support.

#### 2. Install System Dependencies

SSH into the VM and run:

```bash
sudo apt update
sudo apt install -y build-essential git cmake ninja-build
sudo apt install -y nvidia-cuda-toolkit
```

#### 3. Clone and Set Up Minions

```bash
git clone https://github.com/HazyResearch/minions.git
cd minions
```

#### 4. Create a Virtual Environment

```bash
python3 -m venv .venv-msecure
source .venv-msecure/bin/activate
```

Verify the Python path:

```bash
which python
# Expected output: /home/<vm-username>/minions/.venv-msecure/bin/python
```

#### 5. Install Python Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install uv
uv pip install -e .
uv pip install "sglang[all]>=0.4.6.post2"
```

#### 6. Install NVIDIA GPU Attestation Tool

```bash
git clone https://github.com/NVIDIA/nvtrust.git
cd nvtrust/guest_tools/gpu_verifiers/local_gpu_verifier
pip3 install .
python3 -m verifier.cc_admin
```

See [NVIDIA GPU Attestation Tool](https://github.com/NVIDIA/nvtrust/tree/main/guest_tools/gpu_verifiers/local_gpu_verifier) for details.

#### 7. Open Firewall Port for Server Access

In the Azure portal, go to **Networking** for your VM and add two inbound rules to allow: (1) **TCP port 443** and (2) **TCP port 80**.

#### 8. Install azure utils for SNP attestation reports
Following the instructions [here](https://github.com/Azure/confidential-computing-cvm-guest-attestation/tree/main/cvm-attestation-sample-app#build-instructions-for-linux-using-self-contained-attestation-lib) we do:
```bash
git clone https://github.com/Azure/confidential-computing-cvm-guest-attestation.git
cd confidential-computing-cvm-guest-attestation/cvm-attestation-sample-app
sudo ./ClientLibBuildAndInstall.sh
cmake .
make
```

When the local client asks for the attestation report, the following command will be run from a Python process to generate a fresh token:
```bash
sudo ./AttestationClient -o token
```

#### 9. Configure a DNS name for your server

In the Azure portal, for your VM, under **Overview** -> **Essentials**, select DNS Name. A new page will open which will enable you to assign a DNS name label (see field "DNS name label (optional)"). This will give you a DNS address of the following form: `<dns_name>.<region>.cloudapp.azure.com`

#### 10. Generate PEM Keys for secure communication

On the VM, complete the following steps:

1. Download Certbot

```
sudo apt update
sudo apt install -y nginx snapd
sudo snap install core; sudo snap refresh core
sudo snap install --classic certbot
sudo ln -s /snap/bin/certbot /usr/bin/certbot
```

2. Generate PEM keys
   `sudo certbot --nginx -d <DNS ADDRESS>`

This will generate a certificate an a key at the following locations

```
Certificate: /etc/letsencrypt/live/<DNS ADDRESS>/fullchain.pem
Key: /etc/letsencrypt/live/<DNS ADDRESS>/privkey.pem
```

#### 11. Add proper endpoint routing in the NGINX server

1. Open `/etc/nginx/sites-available/default`
2. Then add the following snippet (make the necessary modifications to the DNS Address). This will make sure that nginx forwards traffic to our flask server using https.

```
server {


    index index.html index.htm index.nginx-debian.html;
    server_name <DNS ADDRESS>; # managed by Certbot

    listen 443 ssl http2; # managed by Certbot
    ssl_certificate /etc/letsencrypt/live/<DNS ADDRESS>/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/<DNS ADDRESS>/privkey.pem; # managed by Certbot
    include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains" always;


    location / {
        proxy_pass         https://127.0.0.1:5056;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Additionally, in the Azure portal, go to **Networking** for your VM and add remove the inbound role for port 80. We want to make sure to enforce TLS‐only traffic.

#### 12. Set HuggingFace Token

```bash
export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
```

#### 13. Launch Secure Inference Server

Note: for the command below, your paths the the `ssl-cert` and `ssl-key` come from Step #10 above.

```bash
python secure/remote/worker_server.py --sglang-model "google/gemma-3-4b-it" --ssl-cert <path to certificate> --ssl-key <path to key> --attestation-key-path <location to save attesation private/public keys>
```

### 14. Share the public PEM attestation file with all local clients connecting

Navigate to the path provided in `attestation-key-path` when setting up server and share the `attestation_public.pem` file.

### Part 2: Set up the Local Chat Client

#### 1. Clone the Minions Repository

```bash
git clone https://github.com/HazyResearch/minions.git
cd minions
```

#### 2. Install Minions Locally

```bash
pip install -e ".[secure]"
```

#### 3. Launch Secure Chat (Command Line)

Replace `<AZURE_IP_ADDRESS>` and `<PORT>` with your server details:

```bash
python secure/minions_chat.py --supervisor_url "http://<AZURE_IP_ADDRESS>:<PORT>"
```

#### 4. Launch the Streamlit Chat App (Web UI)

```bash
streamlit run minions_secure_chat.py
```

> **Note**: Configure the **Supervisor URL** as `http://<AZURE_IP_ADDRESS>:<PORT>` in the sidebar.

---

# Secure Minions Local-Remote Protocol

## Overview

The Secure Minions Local-Remote Protocol (`secure/minions_secure.py`) provides an end-to-end encrypted implementation of the Minions protocol that enables secure communication between a local worker model and a remote supervisor server. This protocol includes attestation verification, perfect forward secrecy, and replay protection.

## Features

- **End-to-End Encryption**: All communication encrypted using shared keys from Diffie-Hellman key exchange
- **Attestation Verification**: Verifies integrity and authenticity of the remote supervisor server
- **Perfect Forward Secrecy**: Uses ephemeral key pairs for each session
- **Replay Protection**: Implements nonce-based protection against replay attacks
- **Multi-Modal Support**: Supports text, images, PDFs, and folder processing
- **Comprehensive Logging**: Detailed conversation logs with timing and security metrics

## Basic Usage

### Python API

```python
from minions.clients import OllamaClient
from secure.minions_secure import SecureMinionProtocol

# Initialize local client
local_client = OllamaClient(model_name="llama3.2")

# Create secure protocol instance
protocol = SecureMinionProtocol(
    supervisor_url="https://your-supervisor-server.com",
    local_client=local_client,
    max_rounds=3,
    system_prompt="You are a helpful AI assistant."
)

# Run a secure task
result = protocol(
    task="Analyze this document for key insights",
    context=["Your document content here"],
    max_rounds=2
)

print(f"Final Answer: {result['final_answer']}")
print(f"Session ID: {result['session_id']}")
print(f"Log saved to: {result['log_file']}")

# Clean up the session
protocol.end_session()
```

### Command Line Interface

```bash
python secure/minions_secure.py \
    --supervisor_url https://your-supervisor-server.com \
    --local_client_type ollama \
    --local_model llama3.2 \
    --max_rounds 3
```

## Advanced Usage

### With Image Processing

```python
result = protocol(
    task="Describe what you see in this image",
    context=["Additional context if needed"],
    image_path="/path/to/image.jpg"
)
```

### With PDF Processing

```python
result = protocol(
    task="Summarize the key points from this document",
    context=[],
    pdf_path="/path/to/document.pdf"
)
```

### With Folder Processing

```python
result = protocol(
    task="Analyze all documents in this folder",
    context=[],
    folder_path="/path/to/documents/"
)
```

### With Custom Callback

```python
def message_callback(sender, message, is_final=False):
    print(f"[{sender}]: {message}")

protocol = SecureMinionProtocol(
    supervisor_url="https://your-supervisor-server.com",
    local_client=local_client,
    callback=message_callback
)
```

## Security Features

The protocol implements several security measures:

1. **Attestation Verification**: Verifies that the remote supervisor is running in a trusted environment
2. **Key Exchange**: Uses Diffie-Hellman key exchange for establishing shared secrets
3. **Message Encryption**: All messages are encrypted using AES-GCM with the shared key
4. **Message Authentication**: Messages are signed to prevent tampering
5. **Nonce Protection**: Sequential nonces prevent replay attacks
6. **Session Management**: Ephemeral keys are destroyed after each session

## Configuration Options

- `supervisor_url`: URL of the remote supervisor server
- `local_client`: Local model client (e.g., OllamaClient, MLXLMClient)
- `max_rounds`: Maximum number of conversation rounds (default: 3)
- `callback`: Optional callback function for real-time message updates
- `log_dir`: Directory for saving conversation logs (default: "secure_minion_logs")
- `system_prompt`: Custom system prompt for the local worker

### Output Format

The protocol returns a comprehensive result dictionary:

```python
{
    "final_answer": "The generated answer",
    "session_id": "unique-session-identifier",
    "supervisor_messages": [...],  # Full supervisor conversation
    "worker_messages": [...],      # Full worker conversation
    "remote_usage": Usage(),       # Remote model usage stats
    "local_usage": Usage(),        # Local model usage stats
    "log_file": "/path/to/log.json",
    "timing": {                    # Detailed timing information
        "setup": {...},
        "rounds": [...],
        "total_time": 45.2
    }
}
```

### Logging

All conversations are automatically logged with detailed metadata including:

- Complete message history
- Security session information
- Timing metrics for each operation
- Usage statistics for both local and remote models
- Encryption and verification status

Logs are saved as JSON files in the specified log directory with timestamps and task identifiers.
