"""Profile a warmed real denoiser graph replay; never produce a score."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import traceback

import torch
import torch.utils.checkpoint
import unifolm_wma.utils.inference_runtime as runtime


class ProfileComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--require-cudnn-tf32", choices=("any", "on", "off"),
                        default="any")
    parser.add_argument("--trace-fp32-attention", action="store_true")
    parser.add_argument("--allocation-audit", action="store_true")
    parser.add_argument("--filter-low-audit", action="store_true")
    parser.add_argument("--weight-cache-audit", action="store_true")
    parser.add_argument("--linear-low-audit", action="store_true")
    parser.add_argument("--inline-low-audit", action="store_true")
    parser.add_argument("--attention-parts-audit", action="store_true")
    args, inference_args = parser.parse_known_args()
    assert not (args.allocation_audit and args.filter_low_audit)
    assert sum([args.allocation_audit,args.filter_low_audit,args.weight_cache_audit,args.linear_low_audit,args.inline_low_audit,args.attention_parts_audit])<=1
    args.diagnostic_dir.mkdir(parents=True, exist_ok=False)
    original_configure = runtime.configure_denoiser
    original_attention = None
    audits = []
    linear_audit=None
    if args.attention_parts_audit:
        assert args.require_cudnn_tf32=='off'
        from audit_attention_hidden_parts import Audit
        linear_audit=Audit()
    if args.linear_low_audit:
        from audit_decision_prepacked_low import Audit
        linear_audit=Audit(stage='world' if args.require_cudnn_tf32=='off' else 'decision')
    if args.inline_low_audit:
        from audit_decision_inline_low import Audit
        linear_audit=Audit()
    if args.trace_fp32_attention:
        import unifolm_wma.utils.triton_temporal_attention_fp32 as attention_helper
        original_attention = attention_helper.fused_temporal_attention_fp32
        seen = set()

        def traced_attention(query, key, value, heads, scale, **options):
            signature = (tuple(query.shape), heads, torch.backends.cudnn.allow_tf32)
            if signature not in seen:
                seen.add(signature)
                print('REAL_FP32_ATTENTION_CALL ' + json.dumps(dict(
                    shape=list(query.shape), heads=heads,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32,
                    matmul_tf32=torch.backends.cuda.matmul.allow_tf32)), flush=True)
            return original_attention(query, key, value, heads, scale, **options)

        attention_helper.fused_temporal_attention_fp32 = traced_attention

    def configure(model, *positional, **keywords):
        original_configure(model, *positional, **keywords)
        original = model.apply_model
        from audit_fp16_scalar_residual209 import Audit
        layout_audit = Audit(model)
        audits.append(layout_audit)
        calls = 0
        eligible_calls = 0
        encoder_calls = {'action': 0, 'state': 0}
        encoder_records = []
        for label in encoder_calls:
            encoder = getattr(model.model.diffusion_model, label + '_unet').obs_encoder
            original_encoder = encoder.forward

            def observed(*values, label=label, original_encoder=original_encoder,
                         **options):
                encoder_calls[label] += 1
                return original_encoder(*values, **options)

            encoder.forward = observed

        def record_encoder_calls(before):
            delta = {k: encoder_calls[k] - before[k] for k in encoder_calls}
            encoder_records.append(dict(public_call=calls, encoder_python_calls=delta))
            (args.diagnostic_dir / 'condition-encoder-counts.json').write_text(
                json.dumps(encoder_records, indent=2) + '\n')
            print('CONDITION_ENCODER_CALLS ' + json.dumps(encoder_records[-1]), flush=True)

        def profiled(*values, **options):
            nonlocal calls, eligible_calls
            calls += 1
            eligible = (args.require_cudnn_tf32 == "any" or
                        torch.backends.cudnn.allow_tf32 ==
                        (args.require_cudnn_tf32 == "on"))
            if eligible:
                eligible_calls += 1
            before = encoder_calls.copy()
            if not eligible or eligible_calls < 3:
                result = original(*values, **options)
                record_encoder_calls(before)
                return result
            torch.cuda.synchronize()
            if linear_audit is not None:
                linear_audit.report(args.diagnostic_dir)
            if args.weight_cache_audit:
                from audit_native_weight_cache_coverage import inspect
                inspect(original, model, args.diagnostic_dir)
            if args.filter_low_audit:
                assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
                owners=[entry['numerical_sgemm'] for entry in original.entries.values()
                        if entry.get('numerical_sgemm') is not None]
                assert len(owners)==1 and len(owners[0].swaps)==23
                owner=owners[0];rng=torch.cuda.get_rng_state();audit_rows=[]
                with torch.no_grad():
                    for plan,swap in zip(owner.report['plans'],owner.swaps):
                        geometry=plan['geometry'];co,columns,ci=geometry
                        cache=swap.cache.cache;version=swap.cache.version
                        assert cache.numel()==36*co*ci and not swap.cache.closed
                        if hasattr(swap,'wl'):
                            low=swap.wl;kind='packed15'
                        else:
                            high=cache.half()
                            low=((cache-high.float())*2048.).half();kind='Kahan8'
                        counts=torch.count_nonzero(low.reshape(36,-1),dim=1).tolist()
                        assert swap.cache.version==version
                        audit_rows.append(dict(weight=plan['weight'],geometry=geometry,kind=kind,
                            counts=counts,zero_components=[i for i,n in enumerate(counts) if n==0],
                            master_version_unchanged=True,filter_bytes=cache.numel()*4))
                        del low
                assert torch.equal(rng,torch.cuda.get_rng_state())
                assert sum(r['kind']=='packed15' for r in audit_rows)==15
                assert sum(r['kind']=='Kahan8' for r in audit_rows)==8
                audit=dict(rows=audit_rows,RNG_unchanged=True,original23_cache_owners=True,
                    numerical_path_unchanged=True,instrumented=True,no_official_score=True)
                (args.diagnostic_dir/'filter-low-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
                print('FILTER_LOW_AUDIT '+json.dumps(dict(weights=23,
                    components=23*36,zero_components=sum(len(r['zero_components']) for r in audit_rows))),flush=True)
            from torch.utils._pytree import tree_flatten
            rng = torch.cuda.get_rng_state()
            versions = {n:p._version for n,p in model.named_parameters()}
            reference = original(*values, **options)
            leaves, spec = tree_flatten(reference)
            assert len(leaves) == 3
            copies = [v.detach().clone() for v in leaves]
            torch.cuda.synchronize()
            layout_audit.report(original, model, args.diagnostic_dir,values,options)
            actual, actual_spec = tree_flatten(original(*values, **options))
            torch.cuda.synchronize()
            assert actual_spec == spec and len(actual) == 3
            for a,b in zip(actual,copies):
                assert a.dtype == b.dtype == torch.float32 and a.shape == b.shape
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                assert torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))
            assert torch.equal(rng,torch.cuda.get_rng_state())
            assert versions == {n:p._version for n,p in model.named_parameters()}
            proof = dict(three_outputs_F32_bitwise_exact=True,RNG_unchanged=True,
                all_master_versions_unchanged=True,no_native_graph_parameter_writes=True,
                original_graph_owners_retained=True,no_official_score=True)
            (args.diagnostic_dir/'replay-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
            print('ACTIVATION_LAYOUT_REPLAY_CHECK passed=true',flush=True)
            record_encoder_calls(before)
            raise ProfileComplete

        model.apply_model = profiled

    runtime.configure_denoiser = configure
    script = Path(__file__).parent / "evaluation/world_model_interaction.py"
    sys.path.insert(0, str(script.parent.resolve()))
    sys.argv = [str(script), *inference_args]
    print("DIAGNOSTIC ONLY: profile third eligible public denoiser call after graph "
          "capture/warmup; require_cudnn_tf32=" + args.require_cudnn_tf32 +
          "; no competition result.", flush=True)
    try:
        runpy.run_path(str(script), run_name="__main__")
    except ProfileComplete:
        pass
    finally:
        runtime.configure_denoiser = original_configure
        for audit in audits:audit.close()
        if linear_audit is not None:linear_audit.close()
        if original_attention is not None:
            attention_helper.fused_temporal_attention_fp32 = original_attention


if __name__ == "__main__":
    main()
