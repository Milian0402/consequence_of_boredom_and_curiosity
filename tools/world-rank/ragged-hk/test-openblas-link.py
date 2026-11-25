"""Check static-provider qualification without executing an SGEMM or requiring a BLAS installation."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import runpy
import unittest

verify = runpy.run_path(str(Path(__file__).with_name('check-openblas-link.py')))['verify']
COB, BLAS = Path('/install/libcob_matmul.a'), Path('/native/libopenblas.a')
SYMBOLS = '0000 T cblas_sgemm\n0008 T openblas_get_num_threads\n'
RECORDS = '\n'.join(f'ld: {archive}({member}): {relation} {api}'
                    for api, owner in [('cblas_sgemm','cblas_sgemm.o'), ('openblas_get_num_threads','memory.o')]
                    for archive, member, relation in ((COB,'gemm.c.o','reference to'), (BLAS,owner,'definition of')))


class StaticProviderTest(unittest.TestCase):
    def check(self, trace=RECORDS, nm=SYMBOLS):
        with TemporaryDirectory() as tmp:
            log, symbols = Path(tmp)/'link.log', Path(tmp)/'symbols'
            log.write_text(trace)
            symbols.write_text(nm)
            with redirect_stdout(StringIO()):
                verify(COB, BLAS, log, symbols)

    def test_only_the_called_interfaces_are_required(self):
        self.check()
        self.check(nm=SYMBOLS + '0010 T sgemm_\n')

    def test_reject_wrong_or_missing_proof(self):
        for trace, nm in [
            (RECORDS.replace(str(COB),'/foreign/libcob_matmul.a'), SYMBOLS),
            (RECORDS.replace(str(BLAS),'/foreign/libopenblas.a'), SYMBOLS),
            (RECORDS + '\nld: /foreign/lib.so: definition of cblas_sgemm', SYMBOLS),
            (RECORDS.replace('reference to cblas_sgemm','reference to sgemm_'), SYMBOLS),
            (RECORDS, SYMBOLS.replace('openblas_get_num_threads','sgemm_')),
            (RECORDS, SYMBOLS.replace('T cblas_sgemm','U cblas_sgemm')),
        ]:
            with self.subTest(trace=trace, nm=nm), self.assertRaisesRegex(ValueError,'cblas_sgemm|openblas_get_num_threads'):
                self.check(trace,nm)


if __name__ == '__main__':
    unittest.main()
