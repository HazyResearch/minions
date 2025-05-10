# Secure Minions Chat Manual Setup!

This guide walks you through the manual installation and configuration of a secure chat system using a confidential VM on Azure and the [Minions](https://github.com/HazyResearch/minions) project.

The goal is to enable private, encrypted chat by combining:

1. A Remote Inference Server — securely running on a confidential NVIDIA H100 GPU VM
2. A Local Chat Client — running on your local machine, connected to the remote secure server

You’ll set up both sides of the system so that all messages are encrypted, authenticated, and processed inside a trusted execution environment (TEE).

## Part 1:Remote Inference Server

### 1. Provision a Confidential VM + and secure GPU on Azure

Follow [Azure CGPU onboarding documentation](https://github.com/Azure/az-cgpu-onboarding/blob/main/docs/Confidential-GPU-H100-Manual-Installation-%28PMK-with-Powershell%29.md) to launch a confidential VM with H100 support.

---

### 2. Install System Dependencies

SSH into the VM and run the following:

```bash
sudo apt update
sudo apt install -y build-essential git cmake ninja-build
sudo apt install -y nvidia-cuda-toolkit
```

---

### 3. Clone and Set Up Minions

```bash
git clone https://github.com/HazyResearch/minions.git
cd minions
```

---

### 4. Create a Virtual Environment

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

### 5. Install Python Dependencies

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

### 6. Open Firewall Port for Server Access

In the Azure portal, go to **Networking** for your VM and add an inbound rule to allow **TCP port 5056**.

---

### 7. Set HuggingFace Token

Export your Hugging Face token to enable model loading:

```bash
export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
```

---

### 8. Launch Secure Inference Server

Start the SGLang server with your preferred model (e.g., Gemma-4B):

```bash
python secure/remote/worker_server.py --sglang-model "google/gemma-3-4b-it"
```

## Part 2: Set up the Local Chat Client

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

### 3. Launch Secure Chat

Replace `<AZURE_IP_ADDRESS>` and `<PORT>` with your remote server’s IP and port (e.g., `http://20.57.33.122:5056`):

```bash
python secure/minions_chat.py --supervisor_url "http://<AZURE_IP_ADDRESS>:<PORT>"
```

### 4. Launch the Streamlit Chat App (Optional)

Run the visual chat interface via Streamlit:

```bash
streamlit run minions_secure_chat.py
```

> **Note**: In the app, make sure you configure the **Supervisor URL** as `http://<AZURE_IP_ADDRESS>:<PORT>` in the sidebar before submitting messages.
