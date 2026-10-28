"""Use measured model setup, excluding only its diagnostic reference kernels."""
import ast
import textwrap
from native_sgemm_runtime_sources import runtime as direct_runtime
from native_sgemm_prepacked_profile_sources import profile


def runtime(configs):
    measured=profile()
    start=measured.index('                nodes={')
    end=measured.index('                def enabled(on):',start)
    setup=measured[start:end]
    a=setup.index('                        reference_kernel=')
    b=setup.index('                    kernel=templates[geometry][-1]',a)
    setup=setup[:a]+setup[b:]
    line='                    references.append(DirectSgemmSwap(native,consumer,reference_kernels[geometry],geometry,configs[geometry],extension,api))\n'
    assert setup.count(line)==1;setup=setup.replace(line,'')
    assert 'reference' not in setup
    base=direct_runtime(configs,direct=True)
    a=base.index('        nodes={');b=base.index('        self.graph,self.swaps,self.templates',a)
    source=base[:a]+textwrap.indent(textwrap.dedent(setup),'        ')+base[b:]
    old='from .native_sgemm_direct_params import DirectSgemmSwap'
    assert source.count(old)==1
    source=source.replace(old,
        'from .native_sgemm_prepacked_swap import PrepackedSgemmSwap,NativeOnlyFilterSwap\n'
        'from .native_winograd_gemm_fp16_packed_kernel import transformed_gemm_fp16_packed')
    old='direct_params=True,template_bytes=sum(row[1].numel()*4 for row in templates.values()),'
    assert source.count(old)==1
    source=source.replace(old,'''prepacked_weights=True,direct_params=True,
            numerical_selected=sum(isinstance(s,PrepackedSgemmSwap) for s in swaps),
            native_only_selected=sum(isinstance(s,NativeOnlyFilterSwap) for s in swaps),
            cached_native_filter_bytes=sum(s.cache.cache.numel()*4 for s in swaps),
            packed_filter_bytes=sum(s.wh.numel()*4 for s in swaps if isinstance(s,PrepackedSgemmSwap)),
            template_bytes=sum(row[1].numel()*2+row[2].numel()*4 for row in templates.values()),''')
    old='        assert not self.closed\n        self.graph.replay()'
    assert source.count(old)==1
    source=source.replace(old,'        assert not self.closed\n        for swap in self.swaps:swap.refresh_if_needed()\n        self.graph.replay()')
    ast.parse(source);return source
