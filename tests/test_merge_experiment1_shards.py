import json

from scripts.merge_experiment1_shards import merge_records, summarize


def _write_jsonl(path, records):
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_merge_experiment1_shards_loads_all_shard_records(tmp_path):
    _write_jsonl(
        tmp_path / "records_shard-00000-of-00002.jsonl",
        [{"question_id": "q2", "status": "failed"}, {"question_id": "q1", "status": "complete", "correct": True}],
    )
    _write_jsonl(
        tmp_path / "records_shard-00001-of-00002.jsonl",
        [{"question_id": "q3", "status": "complete", "correct": False}],
    )
    records = merge_records(tmp_path, 2)
    assert [record["question_id"] for record in records] == ["q1", "q2", "q3"]
    summary = summarize(records, 2)
    assert summary["completed"] == 2
    assert summary["failed"] == 1
    assert summary["accuracy"] == 0.5
