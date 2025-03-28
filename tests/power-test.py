import time
from datetime import datetime

def read_energy():
    # Read energy consumption from RAPL in microjoules (µJ).
    with open("/sys/class/powercap/intel-rapl:0/energy_uj", "r") as f:
        return int(f.read().strip())

def get_power_usage(interval=0.5):
    # Calculate power consumption in watts over a given time interval.
    energy_start = read_energy()
    time.sleep(interval)
    energy_end = read_energy()
    
    power_watts = (energy_end - energy_start) / (interval * 1_000_000)  # Convert µJ to W
    return power_watts

if __name__ == "__main__":
    for i in range(100):
        power = get_power_usage()
        # Get the current time
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"{current_time}, power: {power:.6f} W")
        time.sleep(0.5)
