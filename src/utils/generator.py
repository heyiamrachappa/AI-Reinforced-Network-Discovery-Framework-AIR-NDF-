import networkx as nx
import numpy as np
import random
from typing import Dict, List, Tuple, Any
import json
import os

class NetworkGenerator:
    """
    Generates synthetic enterprise network topologies for reinforcement learning simulations.
    Supports Small, Medium, Large, and Cloud-hybrid architectures with subnet partitioning,
    hierarchical routing, and detailed security profiles for each node.
    """
    
    HOST_TYPES = {
        "workstation": {"vuln_range": (0.1, 0.5), "det_range": (0.05, 0.2), "reward": 10},
        "web_server": {"vuln_range": (0.3, 0.8), "det_range": (0.1, 0.4), "reward": 50},
        "database_server": {"vuln_range": (0.1, 0.4), "det_range": (0.2, 0.6), "reward": 100},
        "active_directory": {"vuln_range": (0.05, 0.3), "det_range": (0.3, 0.8), "reward": 150},
        "iot_device": {"vuln_range": (0.6, 0.95), "det_range": (0.01, 0.1), "reward": 5},
        "cloud_storage": {"vuln_range": (0.1, 0.3), "det_range": (0.15, 0.5), "reward": 80},
        "cloud_vm": {"vuln_range": (0.2, 0.7), "det_range": (0.05, 0.3), "reward": 40},
        "router": {"vuln_range": (0.05, 0.2), "det_range": (0.1, 0.3), "reward": 20},
        "firewall": {"vuln_range": (0.01, 0.1), "det_range": (0.4, 0.9), "reward": 30}
    }
    
    SUBNETS = {
        "DMZ": {"allowed_hosts": ["web_server", "router", "firewall"], "base_ip": "192.168.10."},
        "CORPORATE": {"allowed_hosts": ["workstation", "router", "firewall", "iot_device"], "base_ip": "10.100.1."},
        "SECURE_DB": {"allowed_hosts": ["database_server", "router", "firewall"], "base_ip": "10.200.5."},
        "IT_ADMIN": {"allowed_hosts": ["workstation", "active_directory", "router"], "base_ip": "10.50.2."},
        "CLOUD_ZONE": {"allowed_hosts": ["cloud_vm", "cloud_storage", "router"], "base_ip": "172.16.20."}
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate(self, scale: str = "medium") -> nx.Graph:
        """
        Generates a network based on scale: 'small', 'medium', 'large', or 'cloud-hybrid'.
        """
        if scale == "small":
            return self._generate_enterprise(num_subnets=3, nodes_per_subnet=15)
        elif scale == "medium":
            return self._generate_enterprise(num_subnets=5, nodes_per_subnet=45)
        elif scale == "large":
            return self._generate_enterprise(num_subnets=8, nodes_per_subnet=120)
        elif scale == "cloud-hybrid":
            return self._generate_cloud_hybrid()
        else:
            raise ValueError(f"Unknown scale: {scale}. Choose from 'small', 'medium', 'large', 'cloud-hybrid'.")

    def _create_node_profile(self, node_id: int, host_type: str, subnet_name: str, ip: str) -> Dict[str, Any]:
        """Creates the detailed security and service profile of a simulated node."""
        profile = self.HOST_TYPES[host_type]
        vuln_score = round(random.uniform(*profile["vuln_range"]), 3)
        det_prob = round(random.uniform(*profile["det_range"]), 3)
        
        # Open services/ports
        all_ports = [22, 80, 443, 445, 1433, 3306, 3389, 8080]
        num_ports = random.randint(1, 4)
        open_ports = sorted(random.sample(all_ports, num_ports))
        
        # Host importance multiplier
        val_multiplier = 1.0
        if host_type == "active_directory":
            val_multiplier = 2.0
        elif host_type == "database_server":
            val_multiplier = 1.5
            
        reward = int(profile["reward"] * val_multiplier)

        return {
            "id": node_id,
            "host_type": host_type,
            "subnet": subnet_name,
            "ip": ip,
            "vulnerability_score": vuln_score,
            "detection_probability": det_prob,
            "discovery_reward": reward,
            "open_ports": open_ports,
            "os": random.choice(["Linux Kernel 5.15", "Windows Server 2022", "Windows 11", "Ubuntu 22.04 LTS", "Cisco IOS", "Embedded Firmware"]),
            "status": "undiscovered"  # Sim state tracking
        }

    def _generate_enterprise(self, num_subnets: int, nodes_per_subnet: int) -> nx.Graph:
        """Generates a structured enterprise network with subnets and routers."""
        G = nx.Graph()
        
        # Create core routers/firewalls connecting the subnets
        core_nodes = []
        for i in range(num_subnets):
            r_id = len(G)
            ip = f"10.0.0.{i+1}"
            G.add_node(r_id, **self._create_node_profile(r_id, "router", "CORE", ip))
            core_nodes.append(r_id)
            
        # Connect core routers in a ring or fully connected mesh
        for i in range(num_subnets):
            G.add_edge(core_nodes[i], core_nodes[(i + 1) % num_subnets])

        subnets_keys = list(self.SUBNETS.keys())
        
        # Populate each subnet
        for sub_idx, core_router in enumerate(core_nodes):
            subnet_name = subnets_keys[sub_idx % len(subnets_keys)]
            subnet_info = self.SUBNETS[subnet_name]
            
            # Subnet gateway firewall
            fw_id = len(G)
            fw_ip = subnet_info["base_ip"] + "1"
            G.add_node(fw_id, **self._create_node_profile(fw_id, "firewall", subnet_name, fw_ip))
            G.add_edge(core_router, fw_id)
            
            # Generate end hosts in subnet
            subnet_nodes = []
            for h in range(nodes_per_subnet):
                host_type = random.choice(subnet_info["allowed_hosts"])
                node_id = len(G)
                ip = subnet_info["base_ip"] + str(h + 2)
                G.add_node(node_id, **self._create_node_profile(node_id, host_type, subnet_name, ip))
                subnet_nodes.append(node_id)
                
            # Interconnect subnodes using scale-free (preferential attachment) within the subnet
            # And connect them to the firewall
            for node in subnet_nodes:
                G.add_edge(node, fw_id)
                
            # Add internal linkages to simulate local communication (local switches/connections)
            # Add about 1.5x edges as nodes to form a small scale-free mesh within the subnet
            for _ in range(int(nodes_per_subnet * 1.2)):
                n1 = random.choice(subnet_nodes)
                n2 = random.choice(subnet_nodes)
                if n1 != n2:
                    G.add_edge(n1, n2)

        return G

    def _generate_cloud_hybrid(self) -> nx.Graph:
        """Generates a complex cloud-hybrid network structure (on-prem + cloud VPCs)."""
        # A cloud-hybrid consists of an on-prem enterprise network (e.g. 100 nodes)
        # and a cloud environment (e.g. 100 nodes) connected via a VPN gateway.
        G = self._generate_enterprise(num_subnets=3, nodes_per_subnet=30)
        on_prem_size = len(G)
        
        # On-prem VPN gateway (firewall/router)
        vpn_gateway_on_prem = on_prem_size - 1  # Get one of the firewalls/routers
        
        # Cloud Subnets
        cloud_nodes = []
        cloud_core_id = len(G)
        G.add_node(cloud_core_id, **self._create_node_profile(cloud_core_id, "router", "CLOUD_CORE", "172.16.0.1"))
        
        # Bridge link representing VPN tunnel
        G.add_edge(vpn_gateway_on_prem, cloud_core_id)
        
        # Cloud Private VPC and Cloud Public VPC
        for zone in ["CLOUD_PUBLIC", "CLOUD_PRIVATE"]:
            # Firewall for the cloud zone
            fw_id = len(G)
            G.add_node(fw_id, **self._create_node_profile(fw_id, "firewall", zone, "172.16.10.1" if zone == "CLOUD_PUBLIC" else "172.16.20.1"))
            G.add_edge(cloud_core_id, fw_id)
            
            # Add instances
            allowed_hosts = ["cloud_vm", "cloud_storage"]
            if zone == "CLOUD_PUBLIC":
                allowed_hosts.append("web_server")
            else:
                allowed_hosts.append("database_server")
                
            for h in range(40):
                host_type = random.choice(allowed_hosts)
                node_id = len(G)
                ip = f"172.16.{'10' if zone == 'CLOUD_PUBLIC' else '20'}.{h+2}"
                G.add_node(node_id, **self._create_node_profile(node_id, host_type, zone, ip))
                
                # Cloud architecture often has security group styling (nodes connect to the gateway firewall)
                G.add_edge(node_id, fw_id)
                
                # Public cloud VMs might connect to database servers (cross zone communication via security rules)
                # We add some edges between CLOUD_PUBLIC and CLOUD_PRIVATE to represent backend APIs.
                if zone == "CLOUD_PRIVATE" and random.random() < 0.2:
                    # Connect to a random cloud public node
                    public_nodes = [n for n, attr in G.nodes(data=True) if attr.get("subnet") == "CLOUD_PUBLIC"]
                    if public_nodes:
                        G.add_edge(node_id, random.choice(public_nodes))
                        
        return G

def save_network_to_json(G: nx.Graph, filepath: str):
    """Serializes a networkx graph to a custom JSON format with node attributes and adjacency list."""
    data = {
        "nodes": [attr for n, attr in G.nodes(data=True)],
        "edges": list(G.edges())
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_network_from_json(filepath: str) -> nx.Graph:
    """Loads a networkx graph from a saved JSON representation."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    for n in data["nodes"]:
        G.add_node(n["id"], **n)
    for u, v in data["edges"]:
        G.add_edge(u, v)
    return G
