"""Check temporal wrapper/cache dispatch and captured weight residual changes."""
import json
import os
from unittest.mock import patch
import torch
from unifolm_wma.utils.triton_compensated_temporal_conv import make_compensated_temporal_forward
from unifolm_wma.utils.world_zero_weight_low_temporal import make_forward, install_world_zero_weight_low_temporal


def identical(a, b):
    assert torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


@torch.no_grad()
def real_guards(launches):
    os.environ['UNIFOLM_IMPLICIT_TEMPORAL_CONV'] = '1'
    torch.manual_seed(829)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for channels, h, w in ((320, 40, 64), (640, 20, 32), (1280, 10, 16), (1280, 5, 8)):
        root = torch.nn.Module()
        root.test = torch.nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0))
        root.other = torch.nn.Conv3d(channels, channels, 1)
        root.eval().cuda()
        module = root.test
        module.weight.copy_(module.weight.half().float())
        module.forward = make_compensated_temporal_forward(module)
        original, other = module.forward, root.other.forward
        keys, pointer = tuple(root.state_dict()), module.weight.data_ptr()
        assert install_world_zero_weight_low_temporal(root) == 1
        assert root.other.forward == other
        try:
            install_world_zero_weight_low_temporal(root)
            raise AssertionError('Duplicate temporal residual wrapper accepted')
        except ValueError:
            pass
        value = torch.randn(1, channels, 16, h, w, device='cuda') * .7
        identical(module(value), original(value))
        if channels==1280:
            initial_slots=tuple(launches[-1][k] for k in ['WH_ptr','WL_ptr','count_ptr'])
            assert launches[-1]['count_nonzero']==0
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                module(value)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = module(value)
        for exact in (False, True):
            value.add_(.01)
            module.weight.add_(.00001)
            if exact:
                module.weight.copy_(module.weight.half().float())
            module.bias.mul_(.95)
            eager = module(value)
            graph.replay()
            identical(captured, eager)
            identical(captured, original(value))
            if channels==1280:
                assert initial_slots==tuple(launches[-1][k] for k in ['WH_ptr','WL_ptr','count_ptr'])
                assert (launches[-1]['count_nonzero']==0)==exact
        for fallback in ('decision', 'training', 'grad', 'autocast', 'noncontiguous', 'other_shape', 'matmul_tf32'):
            module.train(fallback == 'training')
            torch.backends.cudnn.allow_tf32 = fallback == 'decision'
            torch.backends.cuda.matmul.allow_tf32 = fallback == 'matmul_tf32'
            x = (value.transpose(-1, -2).contiguous().transpose(-1, -2) if fallback == 'noncontiguous' else
                 value[:, :, :3].contiguous() if fallback == 'other_shape' else value)
            with torch.set_grad_enabled(fallback == 'grad'):
                with torch.autocast('cuda', enabled=fallback == 'autocast', dtype=torch.float16):
                    with patch('unifolm_wma.utils.world_zero_weight_low_temporal.zero_weight_low_temporal',
                               side_effect=AssertionError('Fallback entered new temporal kernel')):
                        identical(module(x), original(x))
            module.eval()
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        module.forward = original
        module.forward = make_forward(module)
        expected = original(value)
        stream.wait_stream(torch.cuda.current_stream())
        cold = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cold, stream=stream):
            cold_result = module(value)
        cold.replay()
        identical(cold_result, expected)
        assert keys == tuple(root.state_dict()) and module.weight.data_ptr() == pointer
        assert all(p.dtype == torch.float32 for p in root.parameters())
        print('WORLD_ZERO_WEIGHT_LOW_TEMPORAL_RUNTIME ' + json.dumps(dict(channels=channels,
            spatial=[h, w], short3half_bits=True, zero_nonzero_zero_graph_input_master_bias_bits=True,
            cold_capture_previous_bits=True, native_fallbacks=7,
            factory_other_modules_excluded=True, master_state_storage_preserved=True)), flush=True)
        del graph, cold, root, module, value, captured, eager, expected, cold_result
        torch.cuda.empty_cache()
    print('WORLD_ZERO_WEIGHT_LOW_TEMPORAL_RUNTIME_CHECK passed=true configs=4 '
          'full_video_quality=unverified official_score=false', flush=True)


@torch.no_grad()
def main():
 from packed_temporal_schedule_hooks import installed
 from unifolm_wma.utils.triton_zero_weight_low_temporal import zero_weight_low_temporal
 from unifolm_wma.utils.triton_compensated_temporal_conv import temporal_weight_parts
 from unifolm_wma.utils.triton_zero_weight_low_linear import weight_low_count
 from unifolm_wma.utils import conv_buffer_allocation as buffers
 mode='UNIFOLM_WORLD_PACKED_TEMPORAL_SCHEDULE';os.environ[mode]='1'
 os.environ['UNIFOLM_CONV_BUFFER_NOFILL']='all'
 with installed() as launches:
  real_guards(launches)
  x=torch.randn(1,1280,16,10,16,device='cuda')*.7
  small=torch.randn(1,1280,16,5,8,device='cuda')*.7
  parts=temporal_weight_parts(torch.randn(1280,1280,3,1,1,device='cuda')*.02)
  count=weight_low_count(parts);bias=torch.randn(1280,device='cuda')*.02
  other=torch.randn(1,17,3,3,5,device='cuda')*.7
  op=temporal_weight_parts(torch.randn(19,17,3,1,1,device='cuda')*.02)
  oc=weight_low_count(op);ob=torch.randn(19,device='cuda')*.02
  for policy in ['selected_large','selected_small','disabled','decision','matmul_tf32','other_frames','other_channels','implicit']:
   torch.backends.cudnn.allow_tf32=policy=='decision'
   torch.backends.cuda.matmul.allow_tf32=policy=='matmul_tf32'
   value=small if policy in ['selected_small','implicit'] else x[:,:,:3].contiguous() if policy=='other_frames' else other if policy=='other_channels' else x
   pp,cc,bb=(op,oc,ob) if policy=='other_channels' else (parts,count,bias)
   implementation='implicit' if policy=='implicit' else 'packed'
   os.environ[mode]='0';expected=zero_weight_low_temporal(value,pp,cc,bb,implementation=implementation)
   os.environ[mode]='0' if policy=='disabled' else '1'
   before=len(launches);old=buffers._debug_poison
   try:
    buffers._debug_poison=float('nan');actual=zero_weight_low_temporal(value,pp,cc,bb,implementation=implementation)
   finally:buffers._debug_poison=old
   identical(actual,expected);assert torch.isfinite(actual).all()
   observed=launches[before:]
   assert len(observed)==(0 if policy=='implicit' else 1)
   if observed:assert observed[0]['selected']==(policy in ['selected_large','selected_small'])
   fresh=zero_weight_low_temporal(value,pp,cc,bb,implementation=implementation)
   assert fresh.data_ptr()!=actual.data_ptr();identical(fresh,expected)
  os.environ[mode]='invalid'
  try:zero_weight_low_temporal(x,parts,count,bias,implementation='packed')
  except ValueError as e:assert 'packed temporal schedule' in str(e)
  else:raise AssertionError('Invalid packed temporal mode accepted')
  os.environ[mode]='1';torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 print('PACKED_TEMPORAL_SCHEDULE_RUNTIME '+json.dumps(dict(passed=True,real_block_fixtures=4,policy_fixtures=9,
  original_real_wrapper_factory_and_7_fallback_guards_retained=True,first_timed_master_cache_predicate_same_storage=True,
  zero_nonzero_zero_real_graph=True,native_BK64_FP32_signedzero=True,NaN=True,fresh_output=True,
  mode_backend_shape_implicit_fallbacks=True,invalid_mode_rejected=True,actual_kernel_launches=len(launches))),flush=True)
 print('PACKED_TEMPORAL_SCHEDULE_RUNTIME_CHECK passed=true fixtures=4 policies=9',flush=True)


if __name__=='__main__':main()

