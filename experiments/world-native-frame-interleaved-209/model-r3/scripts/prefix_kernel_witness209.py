import ctypes as C
import hashlib

def witness(graph,kernel,pointers,grid,warps,extension,api):
    raw,_=extension.get_handles(graph)
    nodes=api.read(raw,{})['nodes']
    assert len(nodes)==1 and nodes[0]['type']==0
    node=nodes[0]
    assert int(node['function'],16)==kernel.function
    assert node['grid']==[*grid,1] and node['block']==[32*warps,1,1]
    assert node['shared_bytes']==kernel.shared
    assert [arg['size'] for arg in node['arguments']]==[8]*len(pointers)
    assert [int.from_bytes(bytes.fromhex(arg['bytes_hex']),'little') for arg in node['arguments']]==pointers
    for index in range(len(pointers)+1):
        offset,size=C.c_size_t(),C.c_size_t()
        status=api.lib.cuFuncGetParamInfo(C.c_void_p(kernel.function),index,C.byref(offset),C.byref(size))
        if index==len(pointers):assert status==1,'Unexpected extra runtime argument'
        else:
            api.check(status,'fresh prefix kernel parameter metadata')
            assert (offset.value,size.value)==(index*8,8)
    assert kernel.num_ctas==1 and tuple(kernel.cluster_dims)==(1,1,1)
    return dict(argument_sizes=[8]*len(pointers),bounded_pointer_ABI_verified=True,
        all_captured_argument_bytes_match_known_Torch_pointers=True,
        function_grid_block_shared_verified=True,new_kernel_cubin_SHA256=hashlib.sha256(kernel.asm['cubin']).hexdigest(),
        no_argument_bytes_or_addresses_exported=True)
