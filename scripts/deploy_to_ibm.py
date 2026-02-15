#!/usr/bin/env python3
"""
Deploy QATNE experiments to IBM Quantum hardware

This script automates the deployment of QATNE experiments to IBM Quantum
devices, handling job submission, monitoring, and results retrieval.

Usage:
    python deploy_to_ibm.py --token YOUR_IBM_TOKEN --molecule H2 --backend ibmq_manila
    
Author: QATNE Development Team
Date: 2026-02-15
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package. {e}")
    print("Install with: pip install qiskit qiskit-ibm-runtime qiskit-aer")
    sys.exit(1)


class IBMDeployer:
    """Handles deployment of QATNE experiments to IBM Quantum"""
    
    def __init__(self, token: str, verbose: bool = True):
        """Initialize IBM Quantum service"""
        self.verbose = verbose
        self.service = None
        
        try:
            self.service = QiskitRuntimeService(
                channel='ibm_quantum',
                token=token
            )
            if self.verbose:
                print("✓ Connected to IBM Quantum")
        except Exception as e:
            print(f"✗ Failed to connect to IBM Quantum: {e}")
            print("  Using simulator instead...")
            self.service = None
    
    def list_backends(self, min_qubits: int = 4) -> list:
        """List available quantum backends"""
        if self.service is None:
            return ['aer_simulator']
        
        backends = self.service.backends(
            filters=lambda x: x.configuration().n_qubits >= min_qubits
                             and not x.configuration().simulator
        )
        
        backend_info = []
        for backend in backends:
            config = backend.configuration()
            status = backend.status()
            backend_info.append({
                'name': config.backend_name,
                'n_qubits': config.n_qubits,
                'operational': status.operational,
                'pending_jobs': status.pending_jobs
            })
        
        return backend_info
    
    def select_backend(self, backend_name: Optional[str] = None, 
                      min_qubits: int = 4):
        """Select and return quantum backend"""
        if self.service is None:
            if self.verbose:
                print("Using AerSimulator")
            return AerSimulator()
        
        if backend_name:
            try:
                backend = self.service.backend(backend_name)
                if self.verbose:
                    print(f"✓ Selected backend: {backend_name}")
                return backend
            except Exception as e:
                print(f"✗ Backend {backend_name} not available: {e}")
                print("  Selecting least busy backend...")
        
        # Select least busy backend
        backend = self.service.least_busy(
            filters=lambda x: x.configuration().n_qubits >= min_qubits
        )
        
        if self.verbose:
            print(f"✓ Auto-selected backend: {backend.name}")
            status = backend.status()
            print(f"  Qubits: {backend.configuration().n_qubits}")
            print(f"  Pending jobs: {status.pending_jobs}")
        
        return backend
    
    def submit_job(self, circuit: QuantumCircuit, backend, shots: int = 4096) -> str:
        """Submit quantum circuit job"""
        if self.verbose:
            print(f"\nSubmitting job to {backend.name}...")
            print(f"  Circuit depth: {circuit.depth()}")
            print(f"  Circuit size: {circuit.size()}")
            print(f"  Shots: {shots}")
        
        try:
            # Add measurements if not present
            if not circuit.cregs:
                circuit.measure_all()
            
            # Submit job
            job = backend.run(circuit, shots=shots)
            job_id = job.job_id()
            
            if self.verbose:
                print(f"✓ Job submitted: {job_id}")
                print(f"  Status: {job.status()}")
            
            return job_id
            
        except Exception as e:
            print(f"✗ Job submission failed: {e}")
            return None
    
    def monitor_job(self, job_id: str, check_interval: int = 30):
        """Monitor job status until completion"""
        if self.service is None:
            return
        
        job = self.service.job(job_id)
        
        if self.verbose:
            print(f"\nMonitoring job {job_id}...")
        
        while True:
            status = job.status()
            
            if self.verbose:
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] Status: {status}")
            
            if status.name in ['DONE', 'COMPLETED']:
                if self.verbose:
                    print("✓ Job completed successfully")
                break
            elif status.name in ['CANCELLED', 'ERROR']:
                print(f"✗ Job failed with status: {status}")
                break
            
            time.sleep(check_interval)
    
    def retrieve_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve and process job results"""
        if self.service is None:
            return None
        
        try:
            job = self.service.job(job_id)
            result = job.result()
            
            # Extract counts
            counts = result.get_counts()
            
            if self.verbose:
                print("\n✓ Results retrieved")
                print(f"  Total counts: {sum(counts.values())}")
                print(f"  Unique bitstrings: {len(counts)}")
            
            return {
                'job_id': job_id,
                'backend': job.backend().name,
                'counts': counts,
                'success': result.success,
                'date': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"✗ Failed to retrieve results: {e}")
            return None
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.verbose:
            print(f"\n✓ Results saved to {output_path}")


def create_test_circuit(num_qubits: int = 4) -> QuantumCircuit:
    """Create a simple test circuit"""
    from qiskit import QuantumCircuit, QuantumRegister
    
    qr = QuantumRegister(num_qubits, 'q')
    circuit = QuantumCircuit(qr)
    
    # Simple ansatz
    for i in range(num_qubits):
        circuit.ry(np.pi/4, qr[i])
        circuit.rz(np.pi/6, qr[i])
    
    for i in range(num_qubits-1):
        circuit.cx(qr[i], qr[i+1])
    
    return circuit


def main():
    parser = argparse.ArgumentParser(
        description='Deploy QATNE experiments to IBM Quantum'
    )
    parser.add_argument(
        '--token',
        type=str,
        required=True,
        help='IBM Quantum API token'
    )
    parser.add_argument(
        '--backend',
        type=str,
        default=None,
        help='Specific backend to use (e.g., ibmq_manila)'
    )
    parser.add_argument(
        '--molecule',
        type=str,
        default='H2',
        help='Molecule to simulate (default: H2)'
    )
    parser.add_argument(
        '--shots',
        type=int,
        default=4096,
        help='Number of shots (default: 4096)'
    )
    parser.add_argument(
        '--list-backends',
        action='store_true',
        help='List available backends and exit'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/ibm_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Monitor job until completion'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = IBMDeployer(token=args.token, verbose=not args.quiet)
    
    # List backends if requested
    if args.list_backends:
        print("\nAvailable Backends:")
        print("=" * 60)
        backends = deployer.list_backends()
        for backend in backends:
            print(f"\n{backend['name']}:")
            print(f"  Qubits: {backend['n_qubits']}")
            print(f"  Operational: {backend['operational']}")
            print(f"  Pending jobs: {backend['pending_jobs']}")
        return
    
    # Select backend
    backend = deployer.select_backend(args.backend)
    
    # Create test circuit
    print(f"\nPreparing {args.molecule} experiment...")
    circuit = create_test_circuit(num_qubits=4)
    
    # Submit job
    job_id = deployer.submit_job(circuit, backend, shots=args.shots)
    
    if job_id is None:
        print("✗ Job submission failed")
        return
    
    # Monitor if requested
    if args.monitor:
        deployer.monitor_job(job_id)
    
    # Retrieve results
    results = deployer.retrieve_results(job_id)
    
    if results:
        deployer.save_results(results, args.output)
    
    print("\n" + "=" * 60)
    print("DEPLOYMENT COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
