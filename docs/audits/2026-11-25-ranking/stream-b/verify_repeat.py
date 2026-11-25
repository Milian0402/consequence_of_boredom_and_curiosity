"""Recompute two independent stream-B attempts, keeping each attempt's six paired timings separate."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

from verify import CONFIG, PHASE, REVISION, audit, rows, require

BENCHMARK = "b2cdc6a2985726cb1658dc7cc730b5a1c4b99680"
GITHUB = "https://github.com/Milian0402/consequence_of_boredom_and_curiosity/actions/runs"
PINNED_RUN = 45691645697
RAW_FILES = ("accuracy.csv", "timings.csv", "first-calls.csv", "case-eligibility.csv", "routes.csv",
             "eligibility.csv", "builds.csv", "sources.json", "native-features.txt", "preregistration.json",
             "protocol.md", "panel.csv", "stream-score.json", "stream-summary.csv")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def provenance(directory, attempt):
    p = json.loads((directory / "provenance.json").read_bytes())
    require((p["run"], p["attempt"], p["source_revision"], p["production_revision"]) ==
            (PINNED_RUN, attempt, BENCHMARK, REVISION), f"Stale or unrelated attempt {attempt}")
    require(p["job_id"] > 0 and p["artifact_id"] > 0 and p["artifact_name"] == f"world-rank-x86_64-{PINNED_RUN}-{attempt}",
            f"Unbound job or artifact for attempt {attempt}")
    require(p["zip_crc_passed"] is True and re.fullmatch("[0-9a-f]{64}", p["download_sha256"]) and
            p["hosted_digest"] == "sha256:" + p["download_sha256"] and p["archive_bytes"] > 0,
            f"Original artifact digest or CRC for attempt {attempt}")
    url = f"{GITHUB}/{PINNED_RUN}"
    require(p["run_url"] == url and p["job_url"] == f"{url}/job/{p['job_id']}" and
            p["job_conclusion"] in ("success", "failure"), f"GitHub job identity for attempt {attempt}")
    return p


def bind_host(data, prov):
    lines = data.decode().splitlines()
    kernel = lines[0].split()
    require(len(kernel) > 2 and kernel[0] == "Linux" and len(kernel[1]) > 1, "Private uname record absent")
    require(BENCHMARK in lines and any(line.split(":", 1)[-1].strip() == prov["host"]["model"]
            for line in lines if line.startswith("Model name:")), "Host source or model differs")
    return kernel[1]


def archive_host(archive, directory, prov):
    require(archive.stat().st_size == prov["archive_bytes"] and sha(archive.read_bytes()) == prov["download_sha256"],
            "Archive bytes differ from hosted digest")
    raw = directory / "raw"
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, "Original archive CRC")
        for name in list(RAW_FILES) + [p.name for p in raw.glob("stream-*-fixtures.txt")]:
            require(z.read(f"stream-b/{name}") == (raw / name).read_bytes(), f"Curated file differs from original {name}")
        ranking = json.loads(z.read("stream-b/ranking.json"))
        reduced = json.loads((raw / "ranking.json").read_bytes())
        entries = {n: [role, Path(p).name] for n, (role, p) in ranking["implementations"].items()}
        require(list(entries) == list(reduced["implementations"]) and entries == reduced["implementations"], "Ranking path rewrite")
        require(all(reduced[k] == ranking[k] for k in reduced if k != "implementations"), "Ranking metadata rewrite")
        binaries = [(s.split()[0], Path(s.split(maxsplit=1)[1]).name) for s in z.read("stream-b/stream-binaries.sha256").decode().splitlines()]
        require(binaries == [tuple(s.split()) for s in (raw / "stream-binaries.sha256").read_text().splitlines()], "Binary receipt path rewrite")
        return z.read("stream-b/host.txt")


def backend_receipts(archive, audited_binaries):
    """Return public source/version identifiers from authenticated original upload members."""
    with zipfile.ZipFile(archive) as z:
        def read(name):
            return z.read(name).decode()
        def field(name, pattern):
            match = re.search(pattern, read(name), re.M)
            require(match, f"Missing vendor identity in {name}")
            return match[1]
        receipts = {}
        for n in ("mkl", "mkl-jit"):
            receipts[n] = {"release": field(f"stream-b/{n}-build.txt", r"^version=([0-9.]+)$")}
        receipts["blasfeo"] = {"source_commit": field("stream-b/blasfeo-build.txt", r"revision=([0-9a-f]{40})"),
                               "target": field("stream-b/blasfeo-build.txt", r"target=(\w+)")}
        if "aocl.so" in audited_binaries:
            receipts["aocl"] = {"source_commit": field("stream-b/aocl-configuration.txt", r"^source_sha=([0-9a-f]{40})$"),
                                "source_tag": field("stream-b/aocl-configuration.txt", r"^release_tag=(\S+)$")}
        receipts["openblas"] = {"source_commit": field("stream-b/openblas-build.txt", r"Upstream OpenBLAS \S+ ([0-9a-f]{40})"),
                                "core": field("stream-b/openblas-build.txt", r"openblas_core=(\w+)")}
        build = "onednn/build/build-fingerprint.txt"
        onednn = {"source_tag": field(build, r"^source_tag=(\S+)$"),
                  "source_commit": field(build, r"^source_commit=([0-9a-f]{40})$"),
                  "runtime_version": field("onednn/qualified-oneshot.log", r"^runtime=(\S+)")}
        for sha256, basename in re.findall(r"^([0-9a-f]{64})  .*/([^ /]+)$", read(build), re.M):
            if basename in audited_binaries:
                require(sha256 == audited_binaries[basename], f"oneDNN published vs measured binary {basename}")
            if basename == "libdnnl.so":
                onednn["engine_sha256"] = sha256
        require("engine_sha256" in onednn, "oneDNN engine receipt")
        receipts["onednn"] = onednn
        return receipts


def host_receipt(provs, data):
    private = [bind_host(d, p) for d, p in zip(data, provs)]
    return {"run": PINNED_RUN, "source_revision": BENCHMARK, "job_ids": [p["job_id"] for p in provs],
            "archive_sha256": [p["download_sha256"] for p in provs], "host_record_sha256": [sha(d) for d in data],
            "different_vm_hostnames": private[0] != private[1]}


def series(raw, filename, cols, value):
    records = {}
    for r in rows(raw, filename):
        key = (r["implementation"], r["family"])
        records.setdefault(key, []).append((tuple(r[c] for c in cols), float(r[value])))
    return {k: tuple(sorted(v)) for k, v in records.items()}


def compare(first, second, archives=None, host_files=None, receipt=None, save_receipt=None):
    dirs = (first, second)
    provs = [provenance(d, i) for i, d in enumerate(dirs, 1)]
    require(all(provs[0][k] != provs[1][k] for k in ("job_id", "artifact_id", "download_sha256", "file_manifest_sha256")),
            "Attempt two reuses the original job, artifact or ZIP contents")
    for filename, cols, val in [("timings.csv", ("block", "round", "seed"), "seconds"),
                                ("first-calls.csv", ("mode", "block", "seed"), "first_sgemm_seconds")]:
        a, b = [series(d / "raw", filename, cols, val) for d in dirs]
        common = a.keys() & b.keys()
        require(common and all(a[k] != b[k] for k in common), f"Reused complete timing series in {filename}")
    audits = [audit(d / "raw") for d in dirs]
    for a, p in zip(audits, provs):
        require(a["host"] == p["host"]["model"] and p["host"]["architecture"] == "x86_64", "Compact host identity")
    host_evidence = None
    if archives or host_files:
        data = ([archive_host(z, d, p) for z, d, p in zip(archives, dirs, provs)] if archives
                else [path.read_bytes() for path in host_files])
        host_evidence = host_receipt(provs, data)
        if archives:
            host_evidence["vendor_builds"] = [backend_receipts(z, a["binaries"]) for z, a in zip(archives, audits)]
        if save_receipt:
            save_receipt.write_text(json.dumps(host_evidence, indent=2) + "\n")
    if receipt:
        recorded = json.loads(receipt.read_bytes())
        expected = {"run": PINNED_RUN, "source_revision": BENCHMARK, "job_ids": [p["job_id"] for p in provs],
                    "archive_sha256": [p["download_sha256"] for p in provs]}
        require(all(recorded[k] == v for k, v in expected.items()) and type(recorded["different_vm_hostnames"]) is bool and
                len(recorded["host_record_sha256"]) == 2 and all(re.fullmatch("[0-9a-f]{64}", x) for x in recorded["host_record_sha256"]),
                "VM comparison receipt not bound to both attempts")
        if host_evidence:
            require(all(recorded.get(k) == v for k, v in host_evidence.items()), "VM comparison receipt differs from original records")
        host_evidence = recorded
    require(not save_receipt or host_evidence, "Original host files or archives required to save a VM receipt")
    common_sources = ("sources.json", "preregistration.json", "panel.csv", "protocol.md")
    require(all((first / "raw" / s).read_bytes() == (second / "raw" / s).read_bytes() for s in common_sources),
            "Frozen source, geometry or numerical protocol changed")
    builds = [{r["implementation"]: r for r in rows(d / "raw", "builds.csv") if r["status"] == "qualified"} for d in dirs]
    binsets = [a["binaries"] for a in audits]
    products = [a["coverage"]["methods"] for a in audits]
    inventories = {"same_admitted_methods": set(products[0]) == set(products[1]), "methods": products,
                   "same_binary_by_filename": {f: binsets[0][f] == binsets[1][f] for f in sorted(set(binsets[0]) & set(binsets[1]))},
                   "same_object_source_isa_flags_by_worker": {}}
    for name in sorted(builds[0].keys() & builds[1].keys()):
        a, b = [x[name] for x in builds]
        inventories["same_object_source_isa_flags_by_worker"][name] = {k: a[k] == b[k] for k in
            ("object_sha256", "source_sha256", "trace_source_sha256", "trace_worker_sha256", "flags", "isa")}
    results = {}
    for target in CONFIG:
        present = [target in a["cases"] for a in audits]
        mode = {"available": present, "gates": [a["gates"].get(target) for a in audits]}
        results[target] = mode
        if not all(present):
            continue
        families = list(audits[0]["cases"][target])
        require(families == list(audits[1]["cases"][target]), "Layout order differs")
        mode["gate_passed_both"] = all(g["passed"] for g in mode["gates"])
        mode["layouts"] = {}
        win_sets = {k: [set(), set()] for k in ("incumbent", "best_oneshot", "best_any", "onednn_cached")}
        for f in families:
            per_attempt = []
            for i, a in enumerate(audits):
                r = a["cases"][target][f]
                row = {"candidate_ms": r["candidate_ms"]}
                for k in win_sets:
                    v = r["external"].get("onednn-cached-plan") if k == "onednn_cached" else r[k]
                    row[k] = {stat: v[stat] for stat in ("method", "time_ms", "speedup", "wins")} if v else None
                    if f.startswith("fresh-") and v and v["speedup"] > 1:
                        win_sets[k][i].add(f)
                per_attempt.append(row)
            mode["layouts"][f] = per_attempt
        mode["fresh_layout_wins"] = {k: {"attempt_1": sorted(v[0]), "attempt_2": sorted(v[1]), "both": sorted(v[0] & v[1])}
                                     for k, v in win_sets.items()}
    attempts = [{k: p[k] for k in ("run", "attempt", "job_id", "artifact_id", "download_sha256", "job_url", "job_conclusion")}
                for p in provs]
    for record, audit_result in zip(attempts, audits):
        record.update(cpu_model=audit_result["host"], logical_cpu=audit_result["logical_cpu"],
                      numerical_failures=audit_result["numerical_failures"], max_errors=audit_result["max_errors"],
                      coverage=audit_result["coverage"])
    return {"phase": PHASE, "benchmark": BENCHMARK, "release": REVISION, "source_patch_sha256": audits[0]["source_patch_sha256"],
            "attempts": attempts, "same_cpu_model": audits[0]["host"] == audits[1]["host"], "vm_comparison": host_evidence,
            "programs": inventories, "modes": results}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("first", type=Path)
    cli.add_argument("second", type=Path)
    opts = cli.add_mutually_exclusive_group()
    opts.add_argument("--archives", nargs=2, type=Path, metavar=("ZIP1", "ZIP2"),
                      help="bind the compact evidence and private VM comparison to both original uploads")
    opts.add_argument("--host-files", nargs=2, type=Path, metavar=("HOST1", "HOST2"),
                      help="compare original host records from previously authenticated uploads")
    cli.add_argument("--host-receipt", type=Path, help="compare or replay the privacy-preserving VM receipt")
    cli.add_argument("--save-host-receipt", type=Path)
    args = cli.parse_args()
    print(json.dumps(compare(args.first, args.second, args.archives, args.host_files, args.host_receipt,
                             args.save_host_receipt), indent=2, allow_nan=False))
