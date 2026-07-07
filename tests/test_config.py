from aido_bba.config import data_root, brca_output_root

def test_paths_are_pathlike():
    assert data_root().name
    assert brca_output_root().name
