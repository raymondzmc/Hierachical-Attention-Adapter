import os
import matplotlib.pyplot as plt
import safetensors.torch

# Given data
output_path = 'output/mistral-samsum-structured'
for checkpoint_dir in os.listdir(output_path):
    checkpoint_path = os.path.join(output_path, checkpoint_dir)
    if not os.path.isdir(checkpoint_path):
        continue

    state_dict = {}
    for file in os.listdir(checkpoint_path):
        if file.endswith('.safetensors'):
            file_path = os.path.join(checkpoint_path, file)
            state_dict.update(safetensors.torch.load_file(file_path, device="cpu"))

    gate_norm = [(k, v.norm().item()) for k, v in state_dict.items() if 'adapter_gate' in k]
    # Extract layer numbers and corresponding values
    layers = [int(x[0].split('.')[2]) for x in gate_norm]
    values = [x[1] for x in gate_norm]

    # Sort data by layer number to ensure correct plotting
    sorted_data = sorted(zip(layers, values))
    sorted_layers, sorted_values = zip(*sorted_data)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_layers, sorted_values, marker='o', linestyle='-', color='b')
    plt.title('Adapter Gate Values Across Model Layers')
    plt.xlabel('Layer')
    plt.ylabel('Value')
    plt.xticks(sorted_layers)
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(checkpoint_path, 'gate_norm.png'))