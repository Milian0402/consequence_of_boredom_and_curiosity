"""Count dense terms and observed co-rating contributions without materializing a Gram matrix."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def count(path):
    expected = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Unexpected MovieLens 100K archive SHA-256")
    items = Counter()
    user_items = set()
    with ZipFile(path) as archive, archive.open("ml-100k/u.data") as rows:
        for row in rows:
            user, item, rating, _ = map(int, row.split())
            if (user, item) in user_items or not (1 <= rating <= 5):
                raise ValueError("Duplicate or invalid rating")
            user_items.add((user, item))
            items[item] += 1
    users = len({user for user, _ in user_items})
    return dict(users=users, movies=len(items), ratings=len(user_items),
                dense_terms=users*users*len(items), sparse_ordered_terms=sum(c*c for c in items.values()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    print(json.dumps(count(p.parse_args().archive), indent=2))
