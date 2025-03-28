from minions.utils.energy_tracking import cloud_inference_energy_estimate, better_cloud_inference_energy_estimate

"""
test to compare different cloud inference energy estimates as 
number of input and output tokens are scaled
"""

# scaling number of input tokens
input_tokens = 100
output_tokens = 500

print(f"Fixed output tokens: {output_tokens}")
for i in range(5):
    _, orig_est, _ = cloud_inference_energy_estimate(tokens=input_tokens+output_tokens)
    new_est = better_cloud_inference_energy_estimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )
    new_est = new_est["total_energy_joules"]

    print(f"Input tokens: {input_tokens}, Original estimate: {orig_est:.2f}, New estimate: {new_est:.2f}")

    input_tokens *= 10