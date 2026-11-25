"""Replay an extracted factorial evidence packet without a network connection."""
import csv,hashlib,json,re,tempfile
from pathlib import Path
from score import score,read,sha,need
from report import render
from compare_replay import file_pair

def replay(root):
    receipt=json.loads((root/"provenance.json").read_text())
    member=receipt["member_sha256"];raw=root/"raw"
    score_file=root/"score.py"
    needed={"score.json","effects.csv","interactions.csv","sources.json","panel.csv","historical-endpoints.csv","previous-panel.csv",
            "raw.csv","routes.csv","binaries.csv","protocol.md","estimands.md","factors.json","compile-flags.json","isa.txt","page-size.txt","host.txt"}
    entries={x.name for x in raw.iterdir() if x.is_file()}
    need(entries==needed,"Wrong retained official subset")
    for n in needed:need(sha((raw/n).read_bytes())==member[n],"Original member SHA differs: "+n)
    src=json.loads((raw/"sources.json").read_text())
    need(int(src["run"])==receipt["run"] and src["benchmark"]==receipt["revision"] and receipt["job"]=="layout-factors" and receipt["job_status"]=="success","Wrong measurement provenance")
    for n,h in src["source_sha256"].items():
        need(sha((root/"benchmark"/n).read_bytes())==h,"Benchmark file identity: "+n)
    need(sha(score_file.read_bytes())==src["source_sha256"]["score.py"]==member["score.py"],"Scoring program differs")
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp);d=score(raw,out)
        for n in ("score.json","effects.csv","interactions.csv"):
            file_pair(out/n,raw/n)
    for v in read(root/"logical-input-hashes.csv"):
        need(d["inputs"][v["family"]][v["mode"]+"/"+v["seed"]]==[v["a_hash"],v["b_hash"]],"Independent input fingerprints differ")
    need((root/"RESULTS.md").read_text()==render(d),"Report replay differs")
    orig=receipt["artifact_sha256"]
    need(re.fullmatch("[a-f0-9]{64}",orig) and receipt["artifact_id"]>0,"Malformed original artifact receipt")
    for l in (root/"SHA256SUMS").read_text().splitlines():
        h,k=l.split("  ",1);need(sha((root/k).read_bytes())==h,"Packet inventory differs "+k)
    print(f"Matched {len(needed)} hosted members and replayed {d['coverage']['warm_complete_calls']:,} measured calls.")
if __name__=="__main__":replay(Path(__file__).resolve().parent)
