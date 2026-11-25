"""Classify actual released vendor-leaf calls without attributing forwarding to COB."""
import csv

FIELDS = ('core_m','core_n','core_k','calls','core_calls','core_overwrites','positive_updates',
          'negative_updates','depth_updates','right_strips','bottom_strips','quadrants_ok','edges_ok','provider_ok')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def validate_release(layout, stdout, provider=None, method=None):
    """Return scalar public receipt for one complete ordinary/trace parity worker pair.

    The caller independently validates the worker FP64 checks, CPU, caller
    strides, parent guards and byte equality before admitting this trace.
    """
    if method is None:
        method = 'release-'+(provider or 'openblas')+'-aligned-two'
    name=method.split('-')
    require(len(name)>=3 and name[0]=='release' and name[1] in ('openblas','mkl','armpl'), 'Unregistered release configuration')
    if provider is None: provider=name[1]
    require(name[1]==provider and name[2:] in (['two'],['aligned','two']), 'Method/provider mismatch')
    aligned = name[2:] == ['aligned','two']
    lines = stdout.splitlines()
    detailed = [s for s in lines if s.startswith('SIGNED_TRACE,')]
    route = [s for s in lines if s.startswith('RAGGED_VENDOR_ROUTE,')]
    leaves = [s for s in lines if s.startswith('RAGGED_VENDOR_LEAF,')]
    require(len(route)==len(detailed)==1, 'Expected one public product in provider trace')
    key_values = [s.split('=',1) for s in detailed[0].split(',')[1:]]
    require([pair[0] for pair in key_values] == list(FIELDS), 'Malformed Winograd receipt')
    witness = {q:int(v) for q,v in key_values}
    _,total,stored,full,identity = route[0].split(',',4)
    total,stored,full = map(int, (total,stored,full))
    require(witness['provider_ok']==1 and identity != 'unresolved', 'Unverified serial leaf bindings')
    if provider is not None:
        require((provider=='openblas' and identity.startswith('OpenBLAS 0.3.34 ')) or
                (provider=='mkl' and identity == 'libmkl_intel_lp64.so.3') or
                (provider=='armpl' and (identity=='libarmpl_lp64.so' or identity.startswith('libarmpl_lp64.so.'))),
                'Ordinary configuration and traced vendor identity disagree')
    require(total==witness['calls'] and stored==len(leaves)==(total if total<=64 else 0) and full in (0,1),
            'Incomplete vendor-boundary statement')
    dimensions = []
    operations = []
    for index,row in enumerate(csv.reader(leaves)):
        require(len(row)==10 and int(row[1])==index, 'Missing vendor-call sequence')
        shape = tuple(map(int,row[2:8]))
        m,n,k,lda,ldb,ldc = shape
        op = (float(row[8]),float(row[9]))
        require(min(m,n,k)>0 and lda>=k and ldb>=n and ldc>=n and op in ((1.,0.),(1.,1.)),
                'Unexpected unsigned row-major product')
        dimensions.append(shape)
        operations.append(op)
    m,n,k,lda,ldb,ldc = (int(layout[x]) for x in ('m','n','k','lda','ldb','ldc'))
    oa,ka=(64,32) if aligned else (4,4)
    cm,cn,ck=m//oa*oa,n//oa*oa,k//ka*ka
    require((cm,cn,ck)==tuple(witness[q] for q in FIELDS[:3]), 'Wrong projected Winograd extent')
    expected = 49 + (m!=cm) + (n!=cn) + (aligned and k!=ck)
    if total >= 49:
        require(total==expected and witness['core_calls']==witness['core_overwrites']==49 and
                not witness['positive_updates'] and not witness['negative_updates'] and
                witness['depth_updates']==int(aligned and k!=ck) and witness['right_strips']==int(n!=cn) and
                witness['bottom_strips']==int(m!=cm) and witness['quadrants_ok']==witness['edges_ok']==1,
                'Two-level leaf shape, destination quadrant, edge pointer, update or CPU disagrees')
        require(all(s[:3]==(cm//4,cn//4,ck//4) for s in dimensions[:49]) and
                all(op==(1.,0.) for op in operations[:49]), 'Wrong terminal leaf extent/update')
        tail = []
        if aligned and k!=ck: tail.append(((cm,cn,k-ck,lda,ldb,ldc),(1.,1.)))
        if n!=cn: tail.append(((cm,n-cn,k,lda,ldb,ldc),(1.,0.)))
        if m!=cm: tail.append(((m-cm,n,k,lda,ldb,ldc),(1.,0.)))
        require(list(zip(dimensions[49:],operations[49:]))==tail and full==0,'Changed fringe semantics')
        label='aligned-two-winograd' if aligned else 'two-level-winograd'
    elif not total:
        require(full==0,'Contradictory zero-call fallback')
        label='native-no-vendor-calls'
    elif full:
        require(total==1 and dimensions[0]==(m,n,k,lda,ldb,ldc) and operations[0]==(1.,0.), 'Full forwarding arguments')
        label='ordinary-provider-forward'
    else:
        # The paired public output and provider identity may remain sound if a
        # released allocation/dispatch fallback is taken. Do not label it as
        # the 49-product kernel; preserve the complete observed decomposition.
        label='other-released-provider-path'
    return dict(route=label, plain_depth_terms_expected=int(not aligned and k!=ck and total==expected), provider_identity=identity, full_provider_forward=full,
                leaf_observations='|'.join(s.removeprefix('RAGGED_VENDOR_LEAF,') for s in leaves),
                **witness)
