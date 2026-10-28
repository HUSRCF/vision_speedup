"""NVRTC 12.1.105 SM80 GEGLU scalar math embedded in a Triton epilogue.

Derived from our fp32_geglu kernel's generated PTX, archived with the
diagnostic evidence. Explicit PTX operations retain the native CUDA erff FMA
polynomial and expression rounding. The original large-argument branch is
expressed as a final selection; both paths have no observable side effects.
Installation still requires bitwise checks and complete video validation.
"""
import triton
import triton.language as tl


@triton.jit
def native_geglu(value,gate):
    return tl.inline_asm_elementwise(
        asm=r"""{
            .reg .pred p3,p4;
            .reg .f32 f<33>;
            .reg .b32 r<14>;
            mov.f32 f1,$1;
            mov.f32 f2,$2;
            mul.f32 f3,f2,0f3F3504F3;
            abs.f32 f7,f3;
            setp.ltu.f32 p3,f7,0f3F8060FE;
            setp.ge.f32 p4,f7,0f3F8060FE;
            mul.f32 f8,f3,f3;
            selp.f32 f9,f7,f8,p4;
            selp.f32 f10,0f38EB4C3A,0f38B1E96A,p4;
            selp.f32 f11,0fBAAE005B,0fBA574D20,p4;
            fma.rn.f32 f12,f10,f9,f11;
            selp.f32 f13,0f3C09919F,0f3BAAD5EA,p4;
            fma.rn.f32 f14,f12,f9,f13;
            selp.f32 f15,0fBD24D99A,0fBCDC1BE7,p4;
            fma.rn.f32 f16,f14,f9,f15;
            selp.f32 f17,0f3E235519,0f3DE718AF,p4;
            fma.rn.f32 f18,f16,f9,f17;
            selp.f32 f19,0f3F69B4F9,0fBEC093AC,p4;
            fma.rn.f32 f20,f18,f9,f19;
            selp.f32 f21,0f3F210A14,0f3E0375D3,p4;
            fma.rn.f32 f22,f20,f9,f21;
            neg.f32 f23,f7;
            selp.f32 f24,f23,f3,p4;
            fma.rn.f32 f32,f22,f24,f24;
            ex2.approx.ftz.f32 f25,f32;
            mov.f32 f26,0f3F800000;
            sub.f32 f27,f26,f25;
            mov.b32 r10,f27;
            mov.b32 r11,f3;
            and.b32 r12,r11,-2147483648;
            or.b32 r13,r12,r10;
            mov.b32 f6,r13;
            selp.f32 f32,f32,f6,p3;
            add.f32 f28,f32,0f3F800000;
            mul.f32 f29,f2,0f3F000000;
            mul.f32 f30,f29,f28;
            mul.f32 f31,f1,f30;
            mov.f32 $0,f31;
        }""",
        constraints='=f,f,f',args=[value,gate],dtype=tl.float32,
        is_pure=True,pack=1)
