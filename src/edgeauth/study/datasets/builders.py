"""Builders that turn each source dataset into the common on-disk format.

| name    | source                         | tests                                  | status               |
|---------|--------------------------------|----------------------------------------|----------------------|
| lfw     | LFW View 2 + Kumar attributes  | everything                             | fetched automatically|
| cfp-ff  | CFP frontal-frontal pairs      | baseline for the pose comparison       | fetched automatically|
| cfp-fp  | CFP frontal-profile pairs      | extreme pose (90° yaw)                 | fetched automatically|
| celeba  | CelebA test identities         | sex, eyeglasses, blur, age replication | fetched automatically|
| rfw     | Racial Faces in-the-Wild       | ethnicity replication                  | needs access request |
| bfw     | Balanced Faces in the Wild     | ethnicity x sex replication            | needs access request |

The RFW and BFW parsers follow the published file layouts but have not been run against the
real files: they need the data from the authors (academic access requests).
"""

import csv
import hashlib
import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from .. import lfw
from . import balanced_pairs, fold_for, half_of, write

CFP_URL = "http://www.cfpw.io/cfp-dataset.zip"  # the host serves no HTTPS; pinned by SHA-256
CFP_SHA256 = "666b87635e6af028177ac72a85f03099fac263baf09c21f333fa445f930f65b1"
CELEBA_REVISION = "2d738f56e0e7f925ea36ae7c808ea925264aacec"
CELEBA_SHARDS = {
    "test-00000-of-00003.parquet": "c22a47cdec9e5b36b004c6af906ab6d49066e52a27e74440b55eae3dd9c8d44a",
    "test-00001-of-00003.parquet": "205df6b286ccbdf9b2e67205cc86380fcdbe7673bf87c796a629bd7fafeda1b3",
    "test-00002-of-00003.parquet": "db9c0833ab64e6a156530f87b54b1da931ce5d10731ecb9789238503b009ef97",
}
CELEBA_URL = (
    "https://huggingface.co/datasets/flwrlabs/celeba/resolve/{revision}/"
    "img_align%2Bidentity%2Battr/{shard}"
)
SEED = 2022


def _download(url: str, target: Path, sha256: str | None = None) -> Path:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        request = Request(url, headers={"User-Agent": "EdgeAuth-Study/0.3"})
        with urlopen(request, timeout=300) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)
        partial.replace(target)
    if sha256:
        digest = hashlib.sha256()
        with target.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError(f"checksum mismatch for {target.name}")
    return target


# ---------------------------------------------------------------------------------- LFW --


def build_lfw(directory: Path, source: Path | None = None) -> None:
    raw = source or directory / "raw"
    lfw.fetch(raw)
    rows = lfw.parse_pairs(raw / "pairs.txt")
    attributes = lfw.parse_attributes(raw / "lfw_attributes.txt")
    people = lfw.person_groups(attributes)
    images = {}
    for image, scores in attributes.items():
        labels = lfw.image_groups(scores)
        images[image] = {key: labels[key] for key in lfw.BINARY}
    write(
        directory,
        {
            "name": "lfw",
            "description": "LFW View 2: 6,000 pairs in 10 identity-disjoint folds",
            "image_root": str((raw / "lfw").resolve()),
            "labels": "sex/ethnicity and conditions estimated by Kumar et al. (2009) attribute "
            "classifiers; not self-identified",
        },
        rows,
        people,
        images,
    )


# ---------------------------------------------------------------------------------- CFP --


def _cfp(directory: Path, protocol: str, source: Path | None) -> None:
    raw = source or directory.parent / "cfp-raw"
    if not (raw / "cfp-dataset").is_dir():
        archive = _download(CFP_URL, raw / "cfp-dataset.zip", CFP_SHA256)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(raw)
    base = raw / "cfp-dataset"

    def listing(name: str) -> dict[int, str]:
        entries = {}
        for line in (base / "Protocol" / name).read_text().split("\n"):
            if line.strip():
                index, path = line.split()
                entries[int(index)] = path.replace("../Data/Images/", "")
        return entries

    frontal, profile = listing("Pair_list_F.txt"), listing("Pair_list_P.txt")
    second = profile if protocol == "FP" else frontal
    rows, images = [], {}
    for split in range(1, 11):
        folder = base / "Protocol" / "Split" / protocol / f"{split:02d}"
        for genuine, name in ((True, "same.txt"), (False, "diff.txt")):
            for line in (folder / name).read_text().split("\n"):
                if not line.strip():
                    continue
                i, j = (int(v) for v in line.split(","))
                a, b = frontal[i], second[j]
                pa, pb = a.split("/")[0], b.split("/")[0]
                # CFP splits are not documented as identity-disjoint, so halves come from a
                # per-person hash and impostor pairs that straddle the halves are dropped.
                fa, fb = fold_for(pa, "cfp"), fold_for(pb, "cfp")
                if half_of(fa) != half_of(fb):
                    continue
                rows.append(lfw.PairRow(len(rows), fa, genuine, pa, a, pb, b))
                for image in (a, b):
                    images[image] = {"profile": "yes" if "/profile/" in image else "no"}
    write(
        directory,
        {
            "name": f"cfp-{protocol.lower()}",
            "description": f"CFP {protocol} protocol (Sengupta et al., 2016), 500 celebrities; "
            "halves re-assigned by person hash, straddling impostors dropped",
            "image_root": str((base / "Data" / "Images").resolve()),
            "labels": "no demographic labels; profile flag from the file layout",
        },
        rows,
        {},
        images,
    )


def build_cfp_ff(directory: Path, source: Path | None = None) -> None:
    _cfp(directory, "FF", source)


def build_cfp_fp(directory: Path, source: Path | None = None) -> None:
    _cfp(directory, "FP", source)


# ------------------------------------------------------------------------------- CelebA --


def build_celeba(directory: Path, source: Path | None = None) -> None:
    """CelebA test split (identities unseen in its training split), from a parquet mirror."""
    import pyarrow.parquet as pq

    raw = source or directory / "raw"
    image_dir = directory / "images"
    images_by_person: dict[str, list[str]] = {}
    labels: dict[str, dict[str, bool]] = {}
    for shard, digest in CELEBA_SHARDS.items():
        path = _download(
            CELEBA_URL.format(revision=CELEBA_REVISION, shard=shard), raw / shard, digest
        )
        table = pq.read_table(
            path, columns=["image", "celeb_id", "Male", "Eyeglasses", "Blurry", "Young"]
        )
        for offset, record in enumerate(table.to_pylist()):
            person = f"c{record['celeb_id']:05d}"
            image = f"{person}/{shard[:10]}_{offset:05d}.jpg"
            target = image_dir / image
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(record["image"]["bytes"])
            images_by_person.setdefault(person, []).append(image)
            labels[image] = {k: record[k] for k in ("Male", "Eyeglasses", "Blurry", "Young")}
    people = {}
    for person, pics in images_by_person.items():
        male = sum(labels[i]["Male"] for i in pics) / len(pics)
        people[person] = {"sex": "male" if male >= 0.5 else "female"}
    images = {
        image: {
            "eyeglasses": "yes" if value["Eyeglasses"] else "no",
            "blurry": "yes" if value["Blurry"] else "no",
            "young": "yes" if value["Young"] else "no",
        }
        for image, value in labels.items()
    }
    rows = balanced_pairs(images_by_person, "celeba", per_person=3, seed=SEED)
    write(
        directory,
        {
            "name": "celeba",
            "description": "CelebA test split (Liu et al., 2015): 1,000 identities unseen in "
            "training; 3 genuine pairs per person and as many same-half impostors",
            "image_root": str(image_dir.resolve()),
            "labels": "CelebA's annotated binary attributes (Male, Eyeglasses, Blurry, Young); "
            "no ethnicity labels",
            "source_revision": CELEBA_REVISION,
        },
        rows,
        people,
        images,
    )


# ---------------------------------------------------------------------------------- RFW --

RFW_RACES = {"African": "black", "Asian": "asian", "Caucasian": "white", "Indian": "indian"}


def build_rfw(directory: Path, source: Path | None = None) -> None:
    """RFW test set (Wang et al., 2019). ``source`` is the extracted ``test/`` folder.

    Expected layout: ``data/<Race>/<id>/<id>_<n>.jpg`` and ``txts/<Race>/<Race>_pairs.txt`` in
    LFW pair format (3 fields genuine, 4 fields impostor). Every RFW pair is within one race,
    so all impostors are same-ethnicity strangers.
    """
    if source is None:
        raise ValueError("RFW needs --source pointing at the extracted RFW test/ folder")
    rows, people = [], {}
    for race, label in RFW_RACES.items():
        for line in (source / "txts" / race / f"{race}_pairs.txt").read_text().split("\n"):
            fields = line.split()
            if len(fields) == 3:
                a, b = (fields[0], int(fields[1])), (fields[0], int(fields[2]))
            elif len(fields) == 4:
                a, b = (fields[0], int(fields[1])), (fields[2], int(fields[3]))
            else:
                continue
            fa, fb = fold_for(a[0], "rfw"), fold_for(b[0], "rfw")
            if half_of(fa) != half_of(fb):
                continue
            rows.append(
                lfw.PairRow(
                    len(rows),
                    fa,
                    a[0] == b[0],
                    a[0],
                    f"{race}/{lfw.image_path(*a)}",
                    b[0],
                    f"{race}/{lfw.image_path(*b)}",
                )
            )
            people[a[0]] = people[b[0]] = {"ethnicity": label}
    write(
        directory,
        {
            "name": "rfw",
            "description": "RFW test set: ~3,000 genuine and ~3,000 same-race impostor pairs per "
            "race; halves by person hash",
            "image_root": str((source / "data").resolve()),
            "labels": "race as assigned by the RFW authors (African, Asian, Caucasian, Indian)",
        },
        rows,
        people,
        {},
    )


# ---------------------------------------------------------------------------------- BFW --


def build_bfw(directory: Path, source: Path | None = None) -> None:
    """BFW (Robinson et al., 2020). ``source`` holds the images and the datatable CSV.

    Expected: a ``bfw-*-datatable.csv`` with ``p1, p2, label`` and ``a1, a2`` attribute codes
    such as ``asian_females``, image paths relative to ``source``.
    """
    if source is None:
        raise ValueError("BFW needs --source pointing at the extracted BFW folder")
    table = next(source.glob("bfw*datatable*.csv"))
    rows, people = [], {}
    with table.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            a, b = record["p1"], record["p2"]
            pa, pb = a.split("/")[-2], b.split("/")[-2]
            fa, fb = fold_for(pa, "bfw"), fold_for(pb, "bfw")
            if half_of(fa) != half_of(fb):
                continue
            rows.append(lfw.PairRow(len(rows), fa, record["label"] == "1", pa, a, pb, b))
            for person, code in ((pa, record["a1"]), (pb, record["a2"])):
                ethnicity, sex = code.split("_")
                people[person] = {"ethnicity": ethnicity, "sex": sex.rstrip("s")}
    write(
        directory,
        {
            "name": "bfw",
            "description": "BFW: balanced 8 subgroups (4 ethnicities x 2 sexes); halves by "
            "person hash",
            "image_root": str(source.resolve()),
            "labels": "ethnicity and sex as assigned by the BFW authors",
        },
        rows,
        people,
        {},
    )


BUILDERS = {
    "lfw": build_lfw,
    "cfp-ff": build_cfp_ff,
    "cfp-fp": build_cfp_fp,
    "celeba": build_celeba,
    "rfw": build_rfw,
    "bfw": build_bfw,
}
