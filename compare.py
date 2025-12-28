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
import csv
from typing import Dict, List, Tuple

# configuration
SERVER_HOST = "http://localhost:8000"
ENDPOINT = "/predict"
TEST_TEXTS = [
    "The future of AI is",
    "Once upon a time",
    "Python is a great language because",
    "The weather today is",
    "In a galaxy far far away"
]

def kill_process_on_port(port: int):
    """find and kill any process listening on the specified port."""
    try:
        cmd = f"lsof -i :{port} -t"
        pid = subprocess.check_output(cmd, shell=True).decode().strip()
        if pid:
            os.kill(int(pid), signal.SIGKILL)
            time.sleep(1)
    except subprocess.CalledProcessError:
        pass

async def send_request(session: aiohttp.ClientSession, url: str) -> float:
    text = random.choice(TEST_TEXTS)
    start = time.time()
    async with session.post(url, json={"text": text}) as response:
        await response.json()
        return time.time() - start

async def run_load_test(name: str, num_requests: int, concurrency: int) -> Dict:
    print(f"\n[{name}] starting load test: {num_requests} requests, {concurrency} concurrency")
    url = f"{SERVER_HOST}{ENDPOINT}"
    tasks = []
    
    await asyncio.sleep(40) # generous startup wait for gpt2 model loading

    conn = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=conn) as session:
        # warm up
        print(f"[{name}] warming up...")
        for _ in range(5):
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

def run_server(batching_type: str, env_vars: Dict[str, str] = None) -> subprocess.Popen:
    env = os.environ.copy()
    env["BATCHING_TYPE"] = batching_type
    if env_vars:
        env.update(env_vars)
    
    kill_process_on_port(8000)
    print(f"\nstarting server with BATCHING_TYPE={batching_type}...")
    
    process = subprocess.Popen(
        [sys.executable, "-m", "src.api.server"],
        env=env,
        # stdout=subprocess.DEVNULL, # enable logs for debugging
        # stderr=subprocess.DEVNULL
    )
    return process

async def run_experiment(num_requests: int, concurrency: int):
    results = {}
    
    # 1. NONE
    p = run_server("NONE")
    try:
        results["none"] = await run_load_test("none", num_requests, concurrency)
    finally:
        p.terminate()
        p.wait()

    # 2. DYNAMIC
    # For fair comparison with Continuous (which is naturally 'aggressive'), 
    # we tune Dynamic to be reasonably aggressive too.
    env = {"MAX_BATCH_SIZE": "32", "MAX_LATENCY_MS": "10.0"}
    p = run_server("DYNAMIC", env)
    try:
        results["dynamic"] = await run_load_test("dynamic", num_requests, concurrency)
    finally:
        p.terminate()
        p.wait()

    # 3. CONTINUOUS
    env = {"MAX_BATCH_SIZE": "32"}
    p = run_server("CONTINUOUS", env)
    try:
        results["continuous"] = await run_load_test("continuous", num_requests, concurrency)
    finally:
        p.terminate()
        p.wait()
        
    return results

if __name__ == "__main__":
    # We run a smaller test first because Generation is much slower than Classification
    experiments = [[100, 10], [200, 20]] 
    
    all_data = []

    for reqs, conn in experiments:
        print(f"\n>>> EXPERIMENT: {reqs} reqs, {conn} conn <<<")
        res = asyncio.run(run_experiment(reqs, conn))
        
        row = {
            "requests": reqs,
            "concurrency": conn,
            "throughput_none": res["none"]["throughput_req_per_sec"],
            "throughput_dynamic": res["dynamic"]["throughput_req_per_sec"],
            "throughput_continuous": res["continuous"]["throughput_req_per_sec"],
            "latency_none": res["none"]["avg_latency_ms"],
            "latency_dynamic": res["dynamic"]["avg_latency_ms"],
            "latency_continuous": res["continuous"]["avg_latency_ms"],
        }
        all_data.append(row)

    print("\nFINAL REPORT")
    print(json.dumps(all_data, indent=2))
