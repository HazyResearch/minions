FROM python:3.11-slim

# install dependencies
RUN apt-get update && apt-get install -y wget curl git 

# install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# set up working directory
WORKDIR /app
COPY . .

# install python packages
RUN pip install --upgrade pip \
    && pip install -e . \
    && pip install -e .[mlx] \
    && pip install streamlit

# set up environment variables
ENV OLLAMA_FLASH_ATTENTION=1

# expose port
EXPOSE 8501

# start service
CMD sh -c "ollama serve & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"