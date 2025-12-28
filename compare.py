import asyncio
import aiohttp
import time
import statistics
import random
import subprocess
import sys
import os
import signal
import json
from typing import Dict, List, Tuple

# configuration
SERVER_HOST = "http://localhost:8000"
ENDPOINT = "/predict"
TEST_TEXTS = [
    "this movie is great!",
    "i didn't like this product.",
    "the weather is amazing today.",
    "customer service was terrible.",
    "i am neutral about this."
]

def kill_process_on_port(port: int):
    """find and kill any process listening on the specified port."""
    try:
        # lsof -i :<port> -t returns the pid
        cmd = f"lsof -i :{port} -t"
        pid = subprocess.check_output(cmd, shell=True).decode().strip()
        if pid:
            print(f"killing process {pid} on port {port}...")
            os.kill(int(pid), signal.SIGKILL)
            time.sleep(1) # wait for cleanup
    except subprocess.CalledProcessError:
        # no process found
        pass

async def send_request(session: aiohttp.ClientSession, url: str) -> float:
    """sends a single request and returns the latency in seconds."""
    text = random.choice(TEST_TEXTS)
    start = time.time()
    async with session.post(url, json={"text": text}) as response:
        await response.json()
        return time.time() - start

async def run_load_test(name: str, num_requests: int, concurrency: int) -> Dict:
    """runs the load test against the currently running server."""
    print(f"\n[{name}] starting load test: {num_requests} requests, {concurrency} concurrency")
    
    url = f"{SERVER_HOST}{ENDPOINT}"
    tasks = []
    
    # wait a bit for server to be fully ready
    await asyncio.sleep(2)

    conn = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=conn) as session:
        # warm up
        print(f"[{name}] warming up...")
        for _ in range(10):
            try:
                await send_request(session, url)
            except:
                pass
        
        print(f"[{name}] running benchmark...")
        start_total = time.time()
        
        for _ in range(num_requests):
            tasks.append(send_request(session, url))
            
        latencies = await asyncio.gather(*tasks)
        total_time = time.time() - start_total
        
    avg_latency_ms = statistics.mean(latencies) * 1000
    p50_ms = statistics.median(latencies) * 1000
    p95_ms = statistics.quantiles(latencies, n=20)[18] * 1000
    throughput = num_requests / total_time
    
    results = {
        "name": name,
        "throughput_req_per_sec": round(throughput, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "total_time_sec": round(total_time, 2)
    }
    
    print(f"[{name}] results: {json.dumps(results, indent=2)}")
    return results

def run_server(enable_batching: bool, env_vars: Dict[str, str] = None) -> subprocess.Popen:
    """starts the server subprocess with the given batching configuration."""
    env = os.environ.copy()
    env["ENABLE_BATCHING"] = str(enable_batching)
    if env_vars:
        env.update(env_vars)
    
    # ensure port is free
    kill_process_on_port(8000)
    
    print(f"\nstarting server with ENABLE_BATCHING={enable_batching} (env={env_vars})...")
    process = subprocess.Popen(
        [sys.executable, "-m", "src.api.server"],
        env=env,
        stdout=subprocess.DEVNULL, # silence server output for cleaner report
        stderr=subprocess.DEVNULL
    )
    return process

async def main():
    # settings for the comparison - SCALED UP
    NUM_REQUESTS = 1000
    CONCURRENCY = 100
    
    # 1. run without batching (baseline)
    server_process = run_server(enable_batching=False)
    try:
        # wait for startup
        await asyncio.sleep(5) 
        results_no_batch = await run_load_test("no_batching", NUM_REQUESTS, CONCURRENCY)
    finally:
        server_process.terminate()
        server_process.wait()
    
    # 2. run with batching (optimized)
    # Using larger batch size and lower latency to handle high throughput
    optimized_env = {
        "MAX_BATCH_SIZE": "64",
        "MAX_LATENCY_MS": "5.0"
    }
    server_process = run_server(enable_batching=True, env_vars=optimized_env)
    try:
        # wait for startup
        await asyncio.sleep(5)
        results_batch = await run_load_test("with_batching", NUM_REQUESTS, CONCURRENCY)
    finally:
        server_process.terminate()
        server_process.wait()
        
    # 3. print comparison
    print("\n" + "="*50)
    print("FINAL COMPARISON REPORT")
    print("="*50)
    print(f"{'Metric':<20} | {'No Batching':<15} | {'With Batching':<15} | {'Improvement':<15}")
    print("-" * 70)
    
    throughput_diff = ((results_batch['throughput_req_per_sec'] - results_no_batch['throughput_req_per_sec']) / results_no_batch['throughput_req_per_sec']) * 100
    latency_diff = ((results_no_batch['avg_latency_ms'] - results_batch['avg_latency_ms']) / results_no_batch['avg_latency_ms']) * 100
    
    print(f"{'Throughput (req/s)':<20} | {results_no_batch['throughput_req_per_sec']:<15} | {results_batch['throughput_req_per_sec']:<15} | {throughput_diff:+.1f}%")
    print(f"{'Avg Latency (ms)':<20} | {results_no_batch['avg_latency_ms']:<15} | {results_batch['avg_latency_ms']:<15} | {latency_diff:+.1f}% (faster)")
    print(f"{'P95 Latency (ms)':<20} | {results_no_batch['p95_latency_ms']:<15} | {results_batch['p95_latency_ms']:<15} | -")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
