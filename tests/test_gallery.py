import numpy as np
import pytest
from PIL import Image

from keds.gallery.repository import Gallery, commit_gallery, image_digest, remove_images, safe_id
from keds.rag.gallery_reranking import GalleryReranker


def create_numeric_gallery(tmp_path):
    path = tmp_path / "unit-gallery"
    (path / "images").mkdir(parents=True)
    records = []
    for i in range(3):
        image = Image.new("RGB", (3, 3), (i * 70, 0, 10))
        image.save(path / "images" / f"{i}.png")
        records.append({"image_id": str(i), "image_path": f"images/{i}.png",
                        "pixel_sha256": image_digest(image)})
    return commit_gallery(path, records, np.eye(3, dtype=np.float32), "unit-math", "unit-math")


def test_search_ids_self_exclusion_and_transactional_delete(tmp_path):
    gallery = create_numeric_gallery(tmp_path)
    assert gallery.validate()["valid"]
    results = gallery.search([[1, 0, 0]], top_k=3, exclude_ids=["0"])
    assert len(results) == 2 and all(r["image_id"] != "0" for r in results)
    old_version = gallery.version
    updated = remove_images(gallery.path, iter(["0"]))
    assert len(updated.records) == 2 and updated.version != old_version
    assert len(gallery.records) == 3  # Already loaded snapshot remains valid.
    assert Gallery(gallery.path, version=old_version).image_path("0") == gallery.image_path("0")
    assert updated.search([[1, 0, 0]], exclude_ids=["1", "2"]) == []
    assert (gallery.path / "generations" / old_version).is_dir()
    with pytest.raises(ValueError, match="at least one"):
        remove_images(gallery.path, ["1", "2"])
    assert Gallery(gallery.path).version == updated.version


def test_failed_update_does_not_publish_invalid_generation(tmp_path):
    gallery = create_numeric_gallery(tmp_path)
    invalid = [{**gallery.records[0], "image_path": "images/no-such-file.png"}]
    with pytest.raises(ValueError, match="image path missing"):
        commit_gallery(gallery.path, invalid, gallery.embeddings[:1], "unit-math", "unit-math")
    assert Gallery(gallery.path).version == gallery.version


def test_gallery_detects_corrupt_index_and_unsafe_ids(tmp_path):
    gallery = create_numeric_gallery(tmp_path)
    (gallery.generation / "gallery.faiss").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        Gallery(gallery.path)
    for value in ("../etc", "../../", "a/b", "", "a\\b"):
        with pytest.raises(ValueError):
            safe_id(value)


def test_gallery_validation_detects_replaced_pixels(tmp_path):
    gallery = create_numeric_gallery(tmp_path)
    Image.new("RGB", (3, 3), "yellow").save(gallery.image_path("0"))
    with pytest.raises(ValueError, match="pixels differ"):
        gallery.validate()


def test_gallery_reranking_keeps_fixed_candidate_membership_and_reference_policy(tmp_path):
    gallery = create_numeric_gallery(tmp_path)
    ranker = GalleryReranker({"candidate_pool": 3,
        "weights": {"composed": 0, "visual": 0, "text": 1}})
    rows = ranker.rank(gallery, np.array([[1., 0., 0.]], dtype=np.float32),
                       np.array([1., 0., 0.]), np.array([0., 1., 0.]), 3,
                       exclude_ids=["0"], candidate_ids=["0", "1", "2"])
    assert rows[0]["image_id"] == "1"
    assert {row["image_id"] for row in rows} == {"1", "2"}
    assert all("cosine_score" in row and "score_type" in row for row in rows)

