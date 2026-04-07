"""
Regression test: a peer removed from consensus during scale-down must be able
to rejoin when it restarts with the same URI.

Requires a running cluster-manager instance (default: http://localhost:7333).
Start with: CLUSTER_MANAGER_PORT=7333 cargo run
"""

import pathlib

import requests

from consensus_tests.fixtures import create_collection, upsert_random_points
from .utils import (
    assert_project_root,
    get_cluster_info,
    get_uri,
    processes,
    start_cluster,
    start_peer,
    wait_collection_exists_and_active_on_all_peers,
    wait_for,
    wait_for_peer_online,
    wait_for_some_replicas_not_active,
)

N_PEERS = 4
COLLECTION = "test_collection"
CM_URI = "http://localhost:7333"


def call_cm(peer_uris, urls_to_delete, internal_urls_map):
    resp = requests.post(
        f"{CM_URI}/manage",
        json={
            "urls": peer_uris,
            "urls_to_delete": urls_to_delete,
            "internal_urls_map": internal_urls_map,
            "rules": {"dry_run": False, "replicate": "by_replication_factor", "rebalance": "by_count"},
        },
        timeout=30,
    )
    return resp.json()


def test_scaledown_peer_rejoins(tmp_path: pathlib.Path):
    assert_project_root()
    requests.get(CM_URI, timeout=3)  # fails fast if CM is not running

    peer_api_uris, peer_dirs, bootstrap_uri = start_cluster(tmp_path, N_PEERS)

    create_collection(peer_api_uris[0], shard_number=3, replication_factor=2)
    wait_collection_exists_and_active_on_all_peers(
        collection_name=COLLECTION, peer_api_uris=peer_api_uris
    )
    upsert_random_points(peer_api_uris[0], 100)

    # Snapshot the mapping before killing anything
    internal_urls_map = {uri: get_uri(p.p2p_port) for uri, p in zip(peer_api_uris, list(processes))}

    result = call_cm(peer_api_uris, [], internal_urls_map)
    assert result["status"] == "Ok", f"Cluster not healthy: {result}"

    victim_peer_id = get_cluster_info(peer_api_uris[3])["peer_id"]

    # Kill node-3, preserve port for same-URI restart
    victim_process = processes.pop()
    victim_port = victim_process.p2p_port
    victim_process.kill()

    upsert_random_points(peer_api_uris[0], 100, fail_on_error=False)
    wait_for_some_replicas_not_active(peer_api_uris[0], COLLECTION)

    # Scale down: CM drains and removes node-3
    urls_to_delete = [peer_api_uris[3]]

    def victim_removed():
        call_cm(peer_api_uris, urls_to_delete, internal_urls_map)
        return str(victim_peer_id) not in get_cluster_info(peer_api_uris[0])["peers"]

    wait_for(victim_removed, wait_for_timeout=60, wait_for_interval=3)

    # Restart node-3 with same storage and same port (same URI)
    peer_api_uris[3] = start_peer(peer_dirs[3], "peer_3_restarted.log", bootstrap_uri, port=victim_port)

    # Node-3 should be able to rejoin consensus
    wait_for_peer_online(peer_api_uris[3])

    cluster_info = get_cluster_info(peer_api_uris[0])
    assert str(victim_peer_id) in cluster_info["peers"]
