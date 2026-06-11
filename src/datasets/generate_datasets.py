import sys
import os

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.generator import NetworkGenerator, save_network_to_json

def generate_all():
    print("Initializing Network Generator...")
    gen = NetworkGenerator(seed=42)
    scales = ["small", "medium", "large", "cloud-hybrid"]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for scale in scales:
        print(f"Generating {scale} network...")
        G = gen.generate(scale)
        filepath = os.path.join(current_dir, f"network_{scale}.json")
        save_network_to_json(G, filepath)
        print(f"Saved {scale} network to {filepath}")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")

if __name__ == "__main__":
    generate_all()
