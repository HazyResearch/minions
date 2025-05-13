# Secure Minions Chat Manual Setup!

This guide walks you through the manual installation and configuration of a secure chat system using a confidential VM on Azure and the [Minions](https://github.com/HazyResearch/minions) project.

The goal is to enable private, encrypted chat by combining:

1. A Remote Inference Server — securely running on a confidential NVIDIA H100 GPU VM
2. A Local Chat Client — running on your local machine, connected to the remote secure server

There are two ways to use the secure chat:

- [Method 1](#method-1-connect-to-a-hosted-secure-minions-chat-remote-server): Setup a local client and connect to an existing **secure** Minions Chat remote server
- [Method 2](#method-2-setup-your-own-secure-minions-chat-remote-server): Setup a local client and setup up your own **secure** Minions Chat remote server

In this guide, we will walk you through both ways.

## Table of Contents

### Method 1: Connect to a hosted Secure Minions Chat remote server

- [Clone the Minions Repository](#1-clone-the-minions-repository)
- [Install Minions Locally](#2-install-minions-locally)
- [Launch Secure Chat (Command Line)](#3-launch-secure-chat-command-line)
- [Launch the Streamlit Chat App (Web UI)](#4-launch-the-streamlit-chat-app-web-ui----if-you-want-to-use-the-web-ui)

### Method 2: Setup your own Secure Minions Chat remote server

- **Part 1: Remote Inference Server Setup**
  - [Provision a Confidential VM + secure GPU on Azure](#1-provision-a-confidential-vm--and-secure-gpu-on-azure)
  - [Install System Dependencies](#2-install-system-dependencies)
  - [Clone and Set Up Minions](#3-clone-and-set-up-minions)
  - [Create a Virtual Environment](#4-create-a-virtual-environment)
  - [Install Python Dependencies](#5-install-python-dependencies)
  - [Install NVIDIA GPU Attestation Tool](#6-install-nvidia-gpu-attestation-tool)
  - [Open Firewall Port for Server Access](#7-open-firewall-port-for-server-access)
  - [Set HuggingFace Token](#8-set-huggingface-token)
  - [Launch Secure Inference Server](#9-launch-secure-inference-server)
- **Part 2: Set up the Local Chat Client**
  - [Clone the Minions Repository](#1-clone-the-minions-repository-1)
  - [Install Minions Locally](#2-install-minions-locally-1)
  - [Launch Secure Chat (Command Line)](#3-launch-secure-chat-command-line-1)
  - [Launch the Streamlit Chat App (Web UI)](#4-launch-the-streamlit-chat-app-web-ui----if-you-want-to-use-the-web-ui-1)

## Method 1: Connect to an existing secure Minions Chat remote server

This section walks you through installing and running the Minions chat client locally to connect with a secure remote inference server (e.g., on an Azure confidential GPU VM).

---

### 1. Clone the Minions Repository

```bash
git clone https://github.com/HazyResearch/minions.git
cd minions
```

### 2. Install Minions Locally

Install in editable mode:

```bash
pip install -e .
```

### 3. Launch Secure Chat (Command Line)

Replace `<AZURE_IP_ADDRESS>` and `<PORT>` with the following IP and port: `http://20.57.33.122:5056`

```bash
python secure/minions_chat.py --supervisor_url "http://20.57.33.122:5056"
```

### 4. Launch the Streamlit Chat App (Web UI) -- if you want to use the web UI

Run the visual chat interface via Streamlit:

```bash
streamlit run minions_secure_chat.py
```

> **Note**: In the app, make sure you configure the **Supervisor URL** as `http://<AZURE_IP_ADDRESS>:<PORT>` in the sidebar before submitting messages.

## Method 2: Setup your own secure Minions Chat remote server

### Part 1: Remote Inference Server Setup

This section walks you through installing and running the Minions chat client locally to connect with a secure remote inference server (e.g., on an Azure confidential GPU VM).

---

#### 1. Provision a Confidential VM + and secure GPU on Azure

Follow [Azure CGPU onboarding documentation](https://github.com/Azure/az-cgpu-onboarding/blob/main/docs/Confidential-GPU-H100-Manual-Installation-%28PMK-with-Powershell%29.md) to launch a confidential VM with H100 support.

---

#### 2. Install System Dependencies

SSH into the VM and run the following:

```bash
sudo apt update
sudo apt install -y build-essential git cmake ninja-build
sudo apt install -y nvidia-cuda-toolkit
```

---

#### 3. Clone and Set Up Minions

```bash
git clone https://github.com/HazyResearch/minions.git
cd minions
```

---

#### 4. Create a Virtual Environment

```bash
python3 -m venv .venv-msecure
source .venv-msecure/bin/activate
```

Verify the Python path:

```bash
which python
# Expected output:
# /home/<vm-username>/minions/.venv-msecure/bin/python
```

---

#### 5. Install Python Dependencies

Upgrade pip and core packaging tools:

```bash
pip install --upgrade pip setuptools wheel
```

Install [uv](https://github.com/astral-sh/uv), a fast Python package manager:

```bash
pip install uv
```

Install Minions in editable mode:

```bash
uv pip install -e .
```

> ✅ **Note**: You may want to move this into your setup script under `security[remote]` extras depending on your environment needs.

Install SGLang with full extras:

```bash
uv pip install "sglang[all]>=0.4.6.post2"
```

---

#### 6. Install NVIDIA GPU Attestation Tool

```
git clone https://github.com/NVIDIA/nvtrust.git
cd nvtrust/guest_tools/gpu_verifiers/local_gpu_verifier
pip3 install .
python3 -m verifier.cc_admin
```

Please see [NVIDIA GPU Attestation Tool](https://github.com/NVIDIA/nvtrust/tree/main/guest_tools/gpu_verifiers/local_gpu_verifier) for more details.

---

#### 7. Open Firewall Port for Server Access

In the Azure portal, go to **Networking** for your VM and add an inbound rule to allow **TCP port 5056**.

---

#### 8. Set HuggingFace Token

Export your Hugging Face token to enable model loading:

```bash
export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
```

---

#### 9. Launch Secure Inference Server

Start the SGLang server with your preferred model (e.g., Gemma-4B):

```bash
python secure/remote/worker_server.py --sglang-model "google/gemma-3-4b-it"
```

### Part 2: Set up the Local Chat Client

This section walks you through installing and running the Minions chat client locally to connect with a secure remote inference server (e.g., on an Azure confidential GPU VM).

---

#### 1. Clone the Minions Repository

```bash
git clone https://github.com/HazyResearch/minions.git
cd minions
```

#### 2. Install Minions Locally

Install in editable mode:

```bash
pip install -e .
```

#### 3. Launch Secure Chat (Command Line)

Replace `<AZURE_IP_ADDRESS>` and `<PORT>` with your remote server's IP and port (e.g., `http://20.57.33.122:5056`):

```bash
python secure/minions_chat.py --supervisor_url "http://<AZURE_IP_ADDRESS>:<PORT>"
```

#### 4. Launch the Streamlit Chat App (Web UI) -- if you want to use the web UI

Run the visual chat interface via Streamlit:

```bash
streamlit run minions_secure_chat.py
```

> **Note**: In the app, make sure you configure the **Supervisor URL** as `http://<AZURE_IP_ADDRESS>:<PORT>` in the sidebar before submitting messages.
