import subprocess
import threading
import time
import re
import os


class PowerMonitor:
    def __init__(self, mode="auto", interval=1.0):
        """
        Initialize the power monitor.

        :param mode: "mac" to use powermetrics, "nvidia" to use nvidia-smi, or "auto" to detect.
        :param interval: Sampling interval in seconds.
        """
        if mode == "auto":
            # Auto-detect the appropriate mode
            if self._is_nvidia_available():
                mode = "nvidia"
            elif self._is_mac():
                mode = "mac"
            else:
                raise ValueError(
                    "Could not auto-detect monitoring mode. Please specify 'mac' or 'nvidia'."
                )

        if mode not in ["mac", "nvidia"]:
            raise ValueError("Mode must be either 'mac', 'nvidia', or 'auto'.")

        self.mode = mode
        self.interval = interval
        self.running = False
        self.data = []  # List to store tuples of (timestamp, measurement dict)
        self.thread = None
        self.start_time = None
        self.end_time = None

    def _is_nvidia_available(self):
        """Check if nvidia-smi is available on the system."""
        try:
            subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _is_mac(self):
        """Check if the system is a Mac."""
        return os.uname().sysname == "Darwin"

    def parse_powermetrics(self, output: str) -> dict:
        """
        Parse the powermetrics output to extract power information.
        Expected lines in the output:
          CPU Power: 4382 mW
          GPU Power: 0 mW
          ANE Power: 0 mW
          Combined Power (CPU + GPU + ANE): 4382 mW
        """
        data = {}
        cpu_match = re.search(r"CPU Power:\s*([0-9]+)\s*mW", output)
        gpu_match = re.search(r"GPU Power:\s*([0-9]+)\s*mW", output)
        ane_match = re.search(r"ANE Power:\s*([0-9]+)\s*mW", output)
        combined_match = re.search(
            r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([0-9]+)\s*mW", output
        )

        if cpu_match:
            data["CPU Power"] = int(cpu_match.group(1))
        if gpu_match:
            data["GPU Power"] = int(gpu_match.group(1))
        if ane_match:
            data["ANE Power"] = int(ane_match.group(1))
        if combined_match:
            data["Combined Power"] = int(combined_match.group(1))
        return data

    def start(self):
        """Start the background monitoring thread."""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        return self  # Return self for method chaining

    def _monitor(self):
        """Internal method that polls the appropriate power tool until stopped."""
        while self.running:
            timestamp = time.time()
            measurement = None

            if self.mode == "mac":
                try:
                    result = subprocess.run(
                        [
                            "sudo",
                            "powermetrics",
                            "--samplers",
                            "cpu_power,gpu_power",
                            "-n",
                            "1",
                            "-i",
                            "100",
                        ],
                        stdin=open("/dev/null", "r"),
                        capture_output=True,
                        text=True,
                    )
                    measurement = self.parse_powermetrics(result.stdout)
                except Exception as e:
                    measurement = {"error": str(e)}

            elif self.mode == "nvidia":
                try:
                    result = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=power.draw",
                            "--format=csv,noheader,nounits",
                        ],
                        universal_newlines=True,
                    )
                    # Split the output into lines. Each line is expected to be a number.
                    lines = result.strip().splitlines()
                    gpu_values = []
                    for line in lines:
                        try:
                            gpu_values.append(float(line.strip()))
                        except ValueError:
                            pass
                    if gpu_values:
                        avg_gpu_power = sum(gpu_values) / len(gpu_values)
                        measurement = {
                            "GPU Power (avg)": avg_gpu_power,
                            "Individual GPU Power": gpu_values,
                        }
                    else:
                        measurement = {"error": "No valid GPU power values parsed."}
                except Exception as e:
                    measurement = {"error": str(e)}

            self.data.append((timestamp, measurement))
            time.sleep(self.interval)
        self.end_time = time.time()

    def get_final_estimates(self):
        """
        Compute final estimates for energy and average power based solely on the measured data.

        This estimation is based on:
          - The total runtime (seconds) of the job.
          - The average measured power on the system, converted to Watts.
            (Uses "Combined Power" for Mac or "GPU Power (avg)" for NVIDIA.)
          - The measured energy consumption on the system (energy = average_power * runtime).

        :return: A dict with final estimates, with values formatted to include units.
        """
        if not self.start_time or not self.end_time:
            return {"error": "Monitoring has not been properly started and stopped."}

        # Total runtime in seconds.
        runtime = self.end_time - self.start_time

        # Select measurement key and conversion based on the mode.
        if self.mode == "mac":
            measurement_key = "Combined Power"  # in mW from powermetrics
            conversion = 1 / 1000.0  # Convert mW to W.
        elif self.mode == "nvidia":
            measurement_key = "GPU Power (avg)"  # in W from nvidia-smi
            conversion = 1.0
        else:
            return {"error": "Unknown monitoring mode."}

        valid_measurements = [
            m[measurement_key]
            for _, m in self.data
            if isinstance(m, dict) and measurement_key in m
        ]
        if not valid_measurements:
            return {"error": "No valid power measurements available."}

        # Calculate average power in the correct units.
        avg_power_value = sum(valid_measurements) / len(valid_measurements)
        avg_power_W = avg_power_value * conversion

        # Measured energy consumption (in Joules).
        energy_measured = avg_power_W * runtime

        return {
            "Runtime": f"{runtime:.2f} s",
            "Average Measured Power": f"{avg_power_W:.2f} W",
            "Measured Energy": f"{energy_measured:.2f} J",
        }

    def stop(self):
        """Stop the background monitoring thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join()
        return self  # Return self for method chaining

    def get_stats(self):
        """
        Retrieve the collected measurements.
        :return: List of tuples (timestamp, measurement dict)
        """
        return self.data
