FROM python:3.11-slim

# install dependencies
RUN apt-get update && apt-get install -y wget curl git 



# set up working directory
WORKDIR /app
COPY . .

# install python packages
RUN pip install --upgrade pip \
    && pip install -e . \
    && pip install -e .[mlx] \
    && pip install streamlit

# install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# install tokasaurus
RUN wget -qO- https://astral.sh/uv/install.sh | sh && export PATH="/root/.local/bin:$PATH" && uv pip install --system --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tokasaurus==0.0.1.post1

# set up environment variables
# Do not set these variables in Dockerfile
ENV OLLAMA_FLASH_ATTENTION=1
ENV OPENAI_API_KEY=
ENV AZURE_OPENAI_API_KEY=
ENV OPENAI_BASE_URL=
ENV OPENROUTER_API_KEY=
ENV OPENROUTER_BASE_URL=
ENV TOGETHER_API_KEY=
ENV PERPLEXITY_API_KEY=
ENV PERPLEXITY_BASE_URL=
ENV TOKASAURUS_BASE_URL=
ENV ANTHROPIC_API_KEY=
ENV GROQ_API_KEY=
ENV DEEPSEEK_API_KEY=

# expose port
EXPOSE 8501

# start service
CMD sh -c "ollama serve & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"