import math

import pytest
import requests

from .helpers.collection_setup import drop_collection
from .helpers.helpers import qdrant_host_headers, request_with_validation
from .helpers.settings import QDRANT_HOST


FIELD_NAME = "text"
POINTS = [
    (0, "alpha alpha"),
    (1, "alpha beta"),
    (2, "beta"),
    (3, "gamma"),
]


def expected_idf(total, frequency):
    return math.log((total - frequency + 0.5) / (frequency + 0.5) + 1.0)


def expected_tf(count):
    # The test index uses k1=1 and b=0.
    return count / (count + 1.0)


@pytest.fixture(autouse=True)
def setup(collection_name):
    drop_collection(collection_name=collection_name)

    response = request_with_validation(
        api="/collections/{collection_name}",
        method="PUT",
        path_params={"collection_name": collection_name},
        body={
            "shard_number": 2,
            "vectors": {
                "size": 1,
                "distance": "Dot",
            },
        },
    )
    assert response.ok, response.text

    response = request_with_validation(
        api="/collections/{collection_name}/index",
        method="PUT",
        path_params={"collection_name": collection_name},
        query_params={"wait": "true"},
        body={
            "field_name": FIELD_NAME,
            "field_schema": {
                "type": "text",
                "tokenizer": "word",
                "lowercase": True,
                "bm25_config": {
                    "enable": True,
                    "k1": 1.0,
                    "b": 0.0,
                },
            },
        },
    )
    assert response.ok, response.text

    response = request_with_validation(
        api="/collections/{collection_name}/points",
        method="PUT",
        path_params={"collection_name": collection_name},
        query_params={"wait": "true"},
        body={
            "points": [
                {
                    "id": point_id,
                    "vector": [1.0],
                    "payload": {FIELD_NAME: text},
                }
                for point_id, text in POINTS
            ],
        },
    )
    assert response.ok, response.text

    yield
    drop_collection(collection_name=collection_name)


def query_text(collection_name, query_str, params=None):
    body = {
        "query": {
            "payload": {
                "text": {
                    "key": FIELD_NAME,
                    "query_str": query_str,
                },
            },
        },
        "limit": 10,
    }
    if params is not None:
        body["params"] = params

    response = request_with_validation(
        api="/collections/{collection_name}/points/query",
        method="POST",
        path_params={"collection_name": collection_name},
        body=body,
    )
    assert response.ok, response.text
    return response.json()["result"]["points"]


def test_payload_text_query_scores_and_updates_live_idf(collection_name):
    results = query_text(collection_name, "alpha")
    idf = expected_idf(total=4, frequency=2)

    assert [point["id"] for point in results] == [0, 1]
    assert results[0]["score"] == pytest.approx(expected_tf(2) * idf)
    assert results[1]["score"] == pytest.approx(expected_tf(1) * idf)

    response = request_with_validation(
        api="/collections/{collection_name}/points/delete",
        method="POST",
        path_params={"collection_name": collection_name},
        query_params={"wait": "true"},
        body={"points": [0]},
    )
    assert response.ok, response.text

    results = query_text(collection_name, "alpha")
    live_idf = expected_idf(total=3, frequency=1)

    assert [point["id"] for point in results] == [1]
    assert results[0]["score"] == pytest.approx(expected_tf(1) * live_idf)


def test_payload_text_query_rejects_empty_query(collection_name):
    response = requests.post(
        f"{QDRANT_HOST}/collections/{collection_name}/points/query",
        headers=qdrant_host_headers(),
        json={
            "query": {
                "payload": {
                    "text": {
                        "key": FIELD_NAME,
                        "query_str": "",
                    },
                },
            },
            "limit": 10,
        },
    )

    assert response.status_code == 422, response.text
    assert "query_str" in response.json()["status"]["error"]


def test_payload_text_query_rejects_unindexed_field(collection_name):
    response = requests.post(
        f"{QDRANT_HOST}/collections/{collection_name}/points/query",
        headers=qdrant_host_headers(),
        json={
            "query": {
                "payload": {
                    "text": {
                        "key": "missing_text",
                        "query_str": "alpha",
                    },
                },
            },
            "limit": 10,
        },
    )

    assert response.status_code == 400, response.text
    assert "missing_text" in response.json()["status"]["error"]


def test_payload_text_query_uses_corpus_scoped_idf(collection_name):
    corpus_params = {
        "idf": {
            "corpus": {
                "must": [{"has_id": [0, 1]}],
            },
        },
    }
    results = query_text(collection_name, "alpha", corpus_params)
    corpus_idf = expected_idf(total=2, frequency=2)

    # The corpus only changes IDF statistics; retrieval still includes point 1.
    assert [point["id"] for point in results] == [0, 1]
    assert results[0]["score"] == pytest.approx(expected_tf(2) * corpus_idf)
    assert results[1]["score"] == pytest.approx(expected_tf(1) * corpus_idf)

    empty_corpus_params = {
        "idf": {
            "corpus": {
                "must": [{"has_id": [999]}],
            },
        },
    }
    results = query_text(collection_name, "alpha", empty_corpus_params)
    empty_corpus_idf = expected_idf(total=0, frequency=0)

    assert [point["id"] for point in results] == [0, 1]
    assert results[0]["score"] == pytest.approx(expected_tf(2) * empty_corpus_idf)
    assert results[1]["score"] == pytest.approx(expected_tf(1) * empty_corpus_idf)
