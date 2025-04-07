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

    def get_total_time(self):
        """Get the total time the monitor has been running."""
        return self.end_time - self.start_time

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
                    print(lines)
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
          - For Mac, also reports individual component power (CPU, GPU, ANE).

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

            # Also track individual components for Mac
            component_keys = ["CPU Power", "GPU Power", "ANE Power"]
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

        result = {
            "Runtime": f"{runtime:.2f} s",
            "Average Measured Power": f"{avg_power_W:.2f} W",
            "Measured Energy": f"{energy_measured:.2f} J",
        }

        # Add individual component power for Mac
        if self.mode == "mac":
            for key in component_keys:
                valid_component_measurements = [
                    m[key] for _, m in self.data if isinstance(m, dict) and key in m
                ]
                if valid_component_measurements:
                    avg_component_power = sum(valid_component_measurements) / len(
                        valid_component_measurements
                    )
                    avg_component_power_W = avg_component_power * conversion
                    component_energy = avg_component_power_W * runtime
                    result[f"Average {key}"] = f"{avg_component_power_W:.2f} W"
                    result[f"{key} Energy"] = f"{component_energy:.2f} J"

        return result

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


def cloud_inference_energy_estimate(
    tokens=500,  # number of output tokens per query
    active_parameters=100e9,  # active parameters (e.g., 100 billion)
    flops_per_token_factor=2,  # FLOPs required per parameter per token
    gpu_peak_flops=9.89e14,  # GPU's theoretical max FLOPs per second (NVIDIA H100)
    utilization=0.10,  # effective utilization (10% of peak performance)
    gpu_power_rating=1500,  # peak power per GPU in watts (including overhead)
    power_utilization_factor=0.70,  # effective power usage (70% of rated power)
):
    """
    Estimate the energy consumption of a GPU inference task based on approximations from epoch ai
    https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use#appendix
    """
    # Calculate total FLOPs required for the query
    total_flops = tokens * flops_per_token_factor * active_parameters

    # Calculate the effective FLOPs per second based on utilization
    effective_flops_per_sec = gpu_peak_flops * utilization

    # Compute inference time in seconds
    inference_time_seconds = total_flops / effective_flops_per_sec

    # Calculate effective power consumption in watts
    effective_power = gpu_power_rating * power_utilization_factor

    # Energy consumed in watt-seconds
    energy_watt_seconds = inference_time_seconds * effective_power

    # Convert energy to watt-hours (1 Wh = 3600 watt-seconds)
    energy_watt_hours = energy_watt_seconds / 3600

    return (effective_power, energy_watt_seconds, energy_watt_hours)


def cloud_inference_energy_estimate_w_model_attributes(
    input_tokens=0,
    output_tokens=500,
    model_attr=None,
    gpu_attr=None,
    inference_wall_time_sec=None,
):
    """
    Estimate energy consumption of a GPU inference task based on approximations from Epoch AI
    (Taking into account difference in impact of input and output tokens)
    https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use#appendix
    """
    if model_attr is None:
        # approximation for GPT-4o from scaling up Mixtral 8x22b model (open-source model with known architecture)
        # should change if better approximations exist
        model_attr = {
            "num_active_params": 100e9,  # number of active parameters in the model (used during inference)
            "hidden_dim": 8448.42,  # model dimension (size of hidden state = embedding size)
            "attn_head_dim": 150.1,  # attention heads dimension
            "num_attn_heads": 57.0,  # number of attention heads
            "num_layers": 77.0,  # number of transformer blocks in model
            "flops_per_tkn_factor": 2,
            "flops_per_tkn_factor_attn": 4,
        }

    if gpu_attr is None:
        """
        GPU utilization higher during prefill stage because of parallel processing of inputs
        during decoding stage, new output tokens are generated sequentially with the auto-regressive function
        (https://arxiv.org/pdf/2410.18038v1)
        power utilization is very high during prefill stage (compute-heavy)
        differs from less compute-intense decoding stage
        (https://www.microsoft.com/en-us/research/uploads/prod/2024/03/GPU_Power_ASPLOS_24.pdf)
        """

        # estimates for H100 GPU
        gpu_attr = {
            "peak_flops": 9.89e14,
            "gpu_prefill_util": 0.5,
            "gpu_decoding_util": 0.1,
            "power_rating": 1500,
            "power_prefill_util": 1.0,
            "power_decoding_util": 0.75,
        }

    # peak GPU Joules per FLOP
    peak_gpu_joules_per_flop = gpu_attr["power_rating"] / gpu_attr["peak_flops"]

    # prefill stage calculations
    prefill_flops = (
        input_tokens
        * model_attr["flops_per_tkn_factor"]
        * model_attr["num_active_params"]
    )
    prefill_flops += (
        (input_tokens**2)
        * model_attr["flops_per_tkn_factor_attn"]
        * model_attr["num_attn_heads"]
        * model_attr["attn_head_dim"]
        * model_attr["num_layers"]
    )
    prefill_energy_joules = (
        prefill_flops
        * (gpu_attr["power_prefill_util"] * peak_gpu_joules_per_flop)
        / gpu_attr["gpu_prefill_util"]
    )

    # decoding stage calculations
    decoding_mean_tokens = (input_tokens + (input_tokens + output_tokens - 1)) / 2
    decoding_flops = (
        output_tokens
        * model_attr["flops_per_tkn_factor"]
        * model_attr["num_active_params"]
    )
    decoding_flops += (
        (decoding_mean_tokens * output_tokens)
        * model_attr["flops_per_tkn_factor_attn"]
        * model_attr["num_attn_heads"]
        * model_attr["attn_head_dim"]
        * model_attr["num_layers"]
    )

    decoding_energy_joules = (
        decoding_flops
        * (gpu_attr["power_decoding_util"] * peak_gpu_joules_per_flop)
        / gpu_attr["gpu_decoding_util"]
    )

    if inference_wall_time_sec is not None:
        total_flops = prefill_flops + decoding_flops
        empirical_util = total_flops / (
            inference_wall_time_sec * gpu_attr["peak_flops"]
        )
        empirical_energy_joules = inference_wall_time_sec * gpu_attr["power_rating"]

        return {
            "inference_wall_time_sec": inference_wall_time_sec,
            "empirical_utilization": empirical_util,
            "total_energy_joules": empirical_energy_joules,
            "total_energy_wh": empirical_energy_joules / 3600,
            "prefill_energy_joules": prefill_energy_joules,
            "decoding_energy_joules": decoding_energy_joules,
        }

    else:

        return {
            "prefill_energy_joules": prefill_energy_joules,
            "decoding_energy_joules": decoding_energy_joules,
            "total_energy_joules": prefill_energy_joules + decoding_energy_joules,
        }


class PowerMonitorContext:
    def __init__(self, mode="auto", interval=1.0):
        self.monitor = PowerMonitor(mode=mode, interval=interval)

    def __enter__(self):
        print("Starting power monitoring...")
        self.monitor.start()
        return self.monitor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop()
        print("\nPower monitoring stopped.")

        # Print the final energy estimates
        final_estimates = self.monitor.get_final_estimates()
        print("\nEnergy Consumption Metrics:")
        for key, value in final_estimates.items():
            print(f"{key}: {value}")
