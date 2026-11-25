"""Require positive calls into a real strict FP32 assembly implementation with the measured shape."""
import re

GEOMETRY=('m','n','k','lda','ldb','ldc')

def validate(layout, stdout, method):
    dims={key:int(layout[key]) for key in GEOMETRY}
    if method not in ('acl','blis'):
        raise ValueError('unexpected independent Arm peer')
    detected=[]
    if isinstance(stdout,bytes):stdout=stdout.decode()
    for line in stdout.splitlines():
        if not line.startswith('ARM_PEER_TRACE '):continue
        parts=dict(word.split('=',1) for word in line.split()[1:])
        if parts.get('method')!=method or any(int(parts[k])!=v for k,v in dims.items()):
            raise ValueError('native independent Arm peer or crop mismatch')
        name=parts['kernel']; calls=int(parts['calls']); bits=int(parts['sve_bits'])
        if calls<1 or bits<0 or bits%128: raise ValueError('invalid kernel entry counters/ISA')
        if method=='blis':
            if name!='bli_sgemm_armsve_asm_2vx10_unindexed' or bits <128 or int(parts['kernel_mnk'])<dims['m']*dims['n']*dims['k']:
                raise ValueError('BLIS selected incorrect kernel or missed rows')
        else:
            if not re.fullmatch(r'arm_gemm::(?:sve_(?:hybrid_fp32_mla|interleaved_fp32_mmla|interleaved_fp32_mla|sgemm)|a64_(?:sgemm_asimd|hybrid_fp32_mla|interleaved_fp32_mla|smallK_hybrid_fp32_mla)|sme2?_interleaved_nomerge_fp32_mopa)\w*',name):
                raise ValueError('unexpected Arm Compute Library kernel')
            if '::sve_' in name and bits==0:raise ValueError('SVE kernel recorded without SVE')
        detected.append(dict(kernel=name,calls=calls,sve_bits=bits,**({'kernel_mnk':int(parts['kernel_mnk'])} if method=='blis' else {})))
    if not detected:raise ValueError('native peer did not enter any qualified FP32 assembly kernel')
    return dict(route='+'.join(sorted({item['kernel'] for item in detected})),method=method,observations=detected,**dims)
