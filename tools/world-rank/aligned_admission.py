"""Admit only complete, verified method/layout pairs to the aligned-core scores."""
import csv
import math

def qualified_layouts(args, cases, entries, results, seeds):
    with (args.out / "accuracy.csv").open() as f:
        checks = list(csv.DictReader(f))
    fields = ("m", "n", "k", "pad")
    with (args.out / "first-calls.csv").open("w",newline="") as f:
        columns=("family",*fields,"implementation","mode","block","seed","first_sgemm_seconds","onednn_impl","verified")
        writer=csv.DictWriter(f,columns)
        writer.writeheader()
        for row in checks:
            details=dict(p.split("=",1) for p in row["detail"].split(",") if "=" in p)
            writer.writerow({k:row.get(k,details.get(k,"")) for k in columns})
    def shape(row):
        return tuple(int(row[key]) for key in fields)
    routes = {}
    route_files = ["routes.csv", "fastmmw-routes.csv"]
    if args.phase == "aligned-core":
        route_files.append("aligned-routes.csv")
    for filename in route_files:
        with (args.out / filename).open() as stream:
            for row in csv.DictReader(stream):
                key = row["implementation"], shape(row)
                if key in routes:
                    raise ValueError(f"Duplicate actual route {key}")
                routes[key] = row["passed"] == "1"
    first, second = seeds
    required = {("random",1,first),("random",2,second),("cancel",0,first),("scaled",0,first)}
    result = set()
    with (args.out / "case-eligibility.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(("family",*fields,"implementation","verified_cases","warm_pairs","qualified","reason"))
        for index, case in enumerate(cases):
            geometry=shape(case)
            for name in entries:
                inspected = [r for r in checks if r["implementation"]==name and shape(r)==geometry]
                gate = set()
                for r in inspected:
                    detail = dict(field.split("=",1) for field in r["detail"].split(",") if "=" in field)
                    if (r["verified"]=="1" and r["threads"]=="1" and r["cpu"].isdigit() and
                        int(r["checked"])==case["m"]*case["n"] and int(r["nonfinite"])==0 and
                        int(r["guard_changes"])==0 and detail.get("input_changes")=="0" and
                        0<=float(r["max_abs_error"])<=0.002 and
                        math.isfinite(float(detail["first_sgemm_seconds"])) and float(detail["first_sgemm_seconds"])>0 and
                        (not name.startswith("onednn-") or detail.get("onednn_impl")) and
                        (not name.startswith("hybrid-") or int(detail["hybrid_vendor_calls"]) ==
                         49+int(case["m"]%4!=0)+int(case["n"]%4!=0))):
                        gate.add((r["mode"],int(r["block"]),int(r["seed"])))
                observed=results.get((index,name),{})
                reason = "all_checks_passed"
                if (name.startswith(("cob-","fastmmw-")) or name == "hk-packed-forms") and not routes.get((name,geometry),False):
                    reason="route_unqualified"
                elif gate!=required or len(inspected)!=4:
                    reason="incomplete_or_failed_full_fp64"
                elif set(observed)!={(b,r) for b in (1,2) for r in range(3)}:
                    reason="incomplete_paired_timings"
                accepted=reason=="all_checks_passed"
                if accepted: result.add((index,name))
                writer.writerow((case["family"],*geometry,name,len(gate),len(observed),int(accepted),reason))
    return result
