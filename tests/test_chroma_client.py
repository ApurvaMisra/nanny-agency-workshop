from nanny_workshop.chroma_client import NannyChroma


def test_add_and_query_records(tmp_path):
    db = NannyChroma(persist_dir=tmp_path, collection_name="test")
    db.add(
        ids=["a", "b"],
        documents=["bilingual nanny with CPR", "experienced with toddlers"],
        metadatas=[{"role": "nanny"}, {"role": "nanny"}],
        embeddings=[[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]],
    )
    results = db.query(query_embedding=[0.1, 0.2, 0.3], n_results=2)
    assert results["ids"][0][0] == "a"  # closest match first


def test_collection_persists_across_clients(tmp_path):
    db1 = NannyChroma(persist_dir=tmp_path, collection_name="persist")
    db1.add(
        ids=["x"],
        documents=["doc"],
        metadatas=[{"k": "v"}],
        embeddings=[[1.0, 0.0]],
    )
    db2 = NannyChroma(persist_dir=tmp_path, collection_name="persist")
    results = db2.query(query_embedding=[1.0, 0.0], n_results=1)
    assert results["ids"][0][0] == "x"
