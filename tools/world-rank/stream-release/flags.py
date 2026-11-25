"""Extract and verify actual compiler target flags for the public archive."""
from pathlib import Path
import json
import shlex
import sys

commands, method, output = sys.argv[1:]
native=method.startswith("cmake-native-")
enabled=not method.endswith("default")
dialect="intel" if method.endswith("intel") else "att"
profiles={}
for record in json.loads(Path(commands).read_text()):
    command=record.get("arguments") or shlex.split(record["command"])
    obj=command[command.index("-o")+1]
    if obj.endswith("/cob_matmul.dir/src/gemm.c.o"):
        name="public"
    elif obj.endswith("/cob_stream_b_avx2.dir/src/cob_stream_b_kernel.c.o"):
        name="avx2"
    elif obj.endswith("/cob_stream_b_avx512.dir/src/cob_stream_b_kernel.c.o"):
        name="avx512"
    else:
        continue
    if name in profiles:
        raise ValueError("duplicate public target "+name)
    flags=[x for x in command if x.startswith(("-D","-m","-O"))]
    arches=[x for x in flags if x.startswith("-march=")]
    expected="-march=native" if native and name=="public" else "-march=x86-64"
    if not arches or arches[-1]!=expected or "-masm="+dialect not in flags:
        raise ValueError((method,name,flags))
    if name=="public":
        if ("-DCOB_ENABLE_X86_STREAM_B=1" in flags)!=enabled or any(x.startswith(("-mavx","-mfma")) for x in flags):
            raise ValueError((method,name,flags))
    else:
        needed="-mavx2" if name=="avx2" else "-mavx512f"
        if needed not in flags or "-mfma" not in flags:
            raise ValueError((method,name,flags))
    profiles[name]=flags
if set(profiles)!=({"public","avx2","avx512"} if enabled else {"public"}):
    raise ValueError(f"{method}: incorrect target set {set(profiles)}")
Path(output).write_text(json.dumps(profiles,indent=2)+"\n")
