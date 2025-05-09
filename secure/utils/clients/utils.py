from typing import Optional
import subprocess
import psutil


class ServerMixin:

    def launch_server(self, launch_command: str, port: int, capture_output: bool = False):
        print(f"Starting server with command: {launch_command}")
        if capture_output:
            kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
        else:
            kwargs = {}
        self.server_process = subprocess.Popen(launch_command, shell=True, **kwargs)
        self.wait_for_ping(port, self.server_process, max_retries=500, ping_endpoint="health")
        print(f"Started server with pid {self.server_process.pid}")

    def __del__(self):
        if hasattr(self, "server_process"):
            print(f"Killing server (pid {self.server_process.pid})...")
            self._terminate_process_tree(self.server_process.pid)
            print("Done killing server.")

    @staticmethod
    def _terminate_process_tree(pid):
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)

            for child in children:
                child.terminate()

            gone, alive = psutil.wait_procs(children, timeout=5)

            for p in alive:
                p.kill()

            parent.terminate()
            parent.wait(5)
        except psutil.NoSuchProcess:
            pass
    
    @staticmethod
    def wait_for_ping(
        port,
        popen: subprocess.Popen,
        retry_seconds=2,
        max_retries=500,
        ping_endpoint: str = "ping",
    ):
        import time
        import requests
        # wait for the server to start, by /ping-ing it
        print(f"Waiting for server to start on port {port}...")
        for i in range(max_retries):
            try:
                requests.get(f"http://localhost:{port}/{ping_endpoint}")
                return
            except requests.exceptions.ConnectionError as e:
                print(f"ConnectionError: {e}")
                if popen.poll() is not None:
                    raise RuntimeError(
                        f"Server died with code {popen.returncode} before starting."
                    )

                print(f"Server not yet started (attempt {i}) retrying... (this is normal)")
                time.sleep(retry_seconds)

        raise RuntimeError(f"Server not started after {max_retries} attempts.")
    

    @staticmethod
    def find_free_port():
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))  # Bind to a free port provided by the host.
            return s.getsockname()[1]  # Return the port number assigned.


