#!/usr/bin/env python3
"""The Blackhole Tensix compute units, bit for bit.

Every rule here was read off the silicon, one LLK at a time; `jacobi`,
`gaussian_step` and `matmul_block_tile` assemble them into the emitted
programs.
"""

import numpy as np

BF16_BITS = 8     # significand including the hidden bit
SRC_BITS = 11     # SrcA / SrcB are 19-bit: 1-8-10
BF16_FIFTH = np.float32(0.20019531250)   # bf16(0.2), what the emitter tile_fills


def _quantize(x, lsb, mode):
    """Round x to a multiple of `lsb`. Every rule except `rhu` is written on the
    MAGNITUDE, which is how sign-magnitude hardware is built and also what the
    measured drift is about ("the answer comes back larger", not "more
    positive").

    `rhu` is the exception and it is not a hypothetical: `any_tiles_bcast`
    aligns with ties toward +inf on the SIGNED field, which is what
    floor(x/lsb + 1/2) does and what a two's-complement fixed-point aligner
    gives you for free. See BCAST_ALIGN.
    """
    if mode == "rhu":                        # ties toward +inf, on the SIGN
        return np.floor(x / lsb + 0.5) * lsb
    n = np.abs(x) / lsb
    lo = np.floor(n)
    frac = n - lo
    if mode == "raz":                                    # ties away from zero
        lo = lo + (frac >= 0.5)
    elif mode == "rne":                                  # ties to even (IEEE)
        lo = lo + ((frac > 0.5) | ((frac == 0.5) & (np.mod(lo, 2) == 1)))
    elif mode == "rtz":                                  # truncate toward zero
        pass
    elif mode == "rup":                                  # away whenever inexact
        lo = lo + (frac > 0)
    elif mode == "exact":
        return x
    else:
        raise ValueError(mode)
    return np.copysign(lo * lsb, x)


def rnd(x, bits=BF16_BITS, mode="raz"):
    """Round to `bits` significant bits -- what a DEST write does."""
    m, e = np.frexp(np.abs(x))
    lsb = np.ldexp(np.ones_like(m), e - bits)
    return np.where(x == 0, np.zeros_like(x), _quantize(x, lsb, mode))


def _exponent(x):
    _, e = np.frexp(np.abs(x))
    return np.where(x == 0, np.int32(-30000), e)


def fpu_add(a, b, field=SRC_BITS, align="raz", dest="raz", dest_bits=BF16_BITS):
    """ELWADD: align into the 11-bit field at the LARGER operand's exponent,
    add exactly there, round into DEST. `field=None` disables the alignment
    (what an infinitely wide adder would do) so its cost can be priced."""
    if field is None:
        return rnd(a + b, dest_bits, dest)
    e = np.maximum(_exponent(a), _exponent(b))
    lsb = np.ldexp(np.ones(1, dtype=np.asarray(a).dtype),
                   np.maximum(e, -120) - field)
    return rnd(_quantize(a, lsb, align) + _quantize(b, lsb, align),
               dest_bits, dest)


def _truncate_sig(x, bits):
    m, e = np.frexp(np.abs(x))
    return np.copysign(np.ldexp(np.floor(m * float(1 << bits)), e - bits), x)


def fpu_mul(a, b, passes=3, dest="raz", field=SRC_BITS, align="raz",
            dest_bits=BF16_BITS):
    """ELWMUL at HiFi: 5b x 7b partial products accumulated IN DEST.

    `passes` is MATH_FIDELITY. 3 and 4 give bit-identical results on bf16
    operands, which is the measured behaviour and the reason HiFi3 costs nothing
    in accuracy here. `passes=1` is LoFi.

    At a WIDE dest the partials accumulate exactly: they are summed inside the
    accumulator, not fed back through a source register, so nothing narrows
    them. Measured -- routing them through the 11-bit field instead scores
    95.43% where this scores 100%. With both operands bf16 the product is then
    exact from 3 passes on, so at a wide DEST fidelity stops mattering entirely.
    """
    wide = dest_bits != BF16_BITS
    a_hi = _truncate_sig(a, 5); a_lo = a - a_hi       # SrcA: hidden + 4 bits
    b_hi = _truncate_sig(b, 7); b_lo = b - b_hi       # SrcB: hidden + 6 bits
    parts = [a_hi * b_hi, a_lo * b_hi, a_hi * b_lo, a_lo * b_lo][:passes]
    acc = parts[0] if wide else rnd(parts[0], dest_bits, dest)
    for p in parts[1:]:
        acc = acc + p if wide else fpu_add(acc, p, field, align, dest, dest_bits)
    return acc


def dest_to_srca(x):
    """The DEST_TO_SRCA gasket of `binary_dest_reuse_tiles`: bf16, TRUNCATED.

    A no-op when DEST is already bf16 -- which is exactly why the bf16 campaign
    never saw it, and why applying it unconditionally is safe."""
    return rnd(x, BF16_BITS, "rtz")


BCAST_ALIGN = "rhu"
"""How `any_tiles_bcast` aligns -- and it is NOT how `add_tiles` aligns.

Measured against a probe that packs the COL broadcast on its own, which turns
gaussian's first multiply into a one-dimensional function of a single bf16
operand times a fixed scalar: 256 sign-and-mantissa classes, exhaustive, no
sampling. `rhu` reproduces all 256; `raz` reproduces 252. The four it misses all
have a NEGATIVE product and the device is one ulp closer to +inf on each, which
is the signature of a tie rule that is signed rather than magnitude-based.

The two datapaths really do differ, and each rejects the other's rule:

    probe                                 raz (magnitude)   rhu (signed)
    jacobi  add_tiles (10.2M samples)       100.000000%      99.754664%
    gaussian bcast<COL,MUL> (256 classes)    98.4375%       100.0000%

So this is not a free parameter that could be set either way -- it is a measured
difference, and the natural reading is that the broadcast path's aligner is a
two's-complement adder where the elementwise path's is sign-magnitude. Only the
ALIGNER differs; the DEST write is ties-away in both.
"""


def bcast_mul(a, b, align=BCAST_ALIGN):
    """any_tiles_bcast<ELWMUL, COL|ROW>. SrcA is the non-broadcast operand.

    Same 5b x 7b multiplier and same three partial products as `fpu_mul`; only
    the aligner's tie rule is different. Broadcasting itself is exact -- it is
    the unpacker replicating a row or a column, not arithmetic -- so the caller
    passes operands already broadcast to full tiles.
    """
    return fpu_mul(a, b, align=align)


def sfpu_add(a, b, dest="rtz", dest_bits=BF16_BITS):
    """add_binary_tile: exact in the SFPU's 32-bit datapath, TRUNCATED on the
    store back into a 16-bit DEST. At a 32-bit DEST there is nothing to round."""
    return rnd(a + b, dest_bits, dest)


def jacobi_step(a, frame="sender", coef=BF16_FIFTH, field=SRC_BITS,
                align="raz", dest="raz", dest_bits=BF16_BITS, sfpu="rtz",
                passes=3, pack="raz"):
    """One emitted jacobi_2d step, border frozen. Reproduces the device exactly.

    frame="sender"    the emitted tree: two add_tiles, a transpose, a
                      dest-reuse ADD, an SFPU add, a dest-reuse MUL.
                      (unfused-wb-dm-stream*, unfused-wb-compute-stream*)
    frame="receiver"  the serial chain: copy_tile then four dest-reuse ADDs,
                      then the MUL -- no SFPU add at all.
                      (unfused-wb-compute-receive-transpose)

    The transpose is `transpose_wh_dest`, a permutation inside DEST with no CB
    round trip, so it is exact and invisible to an element-wise model.
    """
    kw = dict(field=field, align=align, dest=dest, dest_bits=dest_bits)
    C = a[1:-1, 1:-1]; N = a[:-2, 1:-1]; S = a[2:, 1:-1]
    E = a[1:-1, 2:]; W = a[1:-1, :-2]
    # `dest_to_srca` marks every DEST_TO_SRCA reuse. It is a no-op at a bf16
    # DEST, so this is one code path for both widths and not two to keep in
    # step; under fp32_dest_acc_en it is what stops the extra width propagating.
    if frame == "sender":
        s = sfpu_add(fpu_add(dest_to_srca(fpu_add(C, S, **kw)), N, **kw),
                     fpu_add(E, W, **kw), sfpu, dest_bits)
    elif frame == "receiver":
        s = C
        for k in (W, E, S, N):
            s = fpu_add(dest_to_srca(s), k, **kw)
    else:
        raise ValueError(frame)
    y = fpu_mul(dest_to_srca(s), np.asarray(coef, a.dtype), passes, **kw)
    if dest_bits != BF16_BITS:
        # pack_tile, MEASURED: ties away from zero, not ties-to-even. Only
        # visible here -- at a bf16 DEST the multiply has already rounded before
        # pack runs, so the pack is a no-op. `pack` stays a switch so the
        # counterfactual can still be priced.
        y = rnd(y, BF16_BITS, pack)
    b = a.copy()
    b[1:-1, 1:-1] = y
    return b


def jacobi(a0, steps, frame="sender", **kw):
    a = np.asarray(a0, dtype=np.float32).copy()
    for _ in range(steps):
        a = jacobi_step(a, frame, **kw)
    return a




def trunc_bf16_from_f32(x):
    """Drop the low 16 bits of a float32 -- `shrui 16; trunci` in the emission.

    gaussian's pivot reciprocal is NOT a compute-unit op: it is computed on the
    DM RISC-V core in scalar float32 (`arith.divf`) and written back to bf16 by
    this truncation. One more inward-biased rounding, in the data movement
    thread, where nobody would think to look for arithmetic.
    """
    v = np.asarray(x, dtype=np.float32)
    return (v.view(np.int32) & np.int32(-65536)).view(np.float32)


SFPU_RECIP_MANTISSA = (
    0x00, 0x7e, 0x7c, 0x7a, 0x78, 0x76, 0x75, 0x73, 0x71, 0x6f,
    0x6d, 0x6c, 0x6a, 0x68, 0x67, 0x65, 0x64, 0x62, 0x60, 0x5f,
    0x5d, 0x5c, 0x5a, 0x59, 0x58, 0x56, 0x55, 0x53, 0x52, 0x51,
    0x4f, 0x4e, 0x4d, 0x4c, 0x4a, 0x49, 0x48, 0x47, 0x45, 0x44,
    0x43, 0x42, 0x41, 0x40, 0x3f, 0x3d, 0x3c, 0x3b, 0x3a, 0x39,
    0x38, 0x37, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31, 0x30, 0x2f,
    0x2e, 0x2d, 0x2c, 0x2c, 0x2b, 0x2a, 0x29, 0x28, 0x27, 0x26,
    0x25, 0x25, 0x24, 0x23, 0x22, 0x21, 0x21, 0x20, 0x1f, 0x1e,
    0x1e, 0x1d, 0x1c, 0x1b, 0x1b, 0x1a, 0x19, 0x18, 0x18, 0x17,
    0x16, 0x16, 0x15, 0x14, 0x14, 0x13, 0x12, 0x12, 0x11, 0x10,
    0x10, 0x0f, 0x0e, 0x0e, 0x0d, 0x0d, 0x0c, 0x0b, 0x0b, 0x0a,
    0x0a, 0x09, 0x09, 0x08, 0x07, 0x07, 0x06, 0x06, 0x05, 0x05,
    0x04, 0x04, 0x03, 0x02, 0x02, 0x01, 0x01, 0x00
)
SFPU_RECIP_ESUM = (253, 254)   # exp(x) + exp(1/x); 254 only for x = 2**k exactly


def sfpu_recip(x):
    """`recip_tile` on a bf16 operand at a bf16 DEST. Exact, by table."""
    b = (np.float32(x).view(np.uint32) >> 16).astype(np.uint32)
    sign, exp, mant = b & 0x8000, (b >> 7) & 0xFF, b & 0x7F
    out_m = np.asarray(SFPU_RECIP_MANTISSA, np.uint32)[mant]
    esum = np.where(mant == 0, SFPU_RECIP_ESUM[1], SFPU_RECIP_ESUM[0])
    out = sign | (((esum - exp) & 0xFF) << 7) | out_m
    return (out.astype(np.uint32) << 16).view(np.float32)


def gaussian_step(a, t, mul1="bcast", mul2="bcast", sub="fpu", recip="dm",
                  **kw):
    """One elimination step of the emitted gaussian.

    The default is `mask-dm-bcast-rc` (rung 07a):

        recip = trunc_bf16(1.0f / pivot)                 DM core, float32
        m     = any_tiles_bcast<ELWMUL, COL>(recip, col) FPU, SrcA = recip
        p     = any_tiles_bcast<ELWMUL, ROW>(m,     row) FPU, SrcA = m
        A    -= p    binary_dest_reuse<ELWSUB, DEST_TO_SRCB>

    The masks are affine and exact: columns j >= t (#map4), rows i >= t+1
    (#map5). Outside them an operand is zeroed, so the update is a no-op rather
    than a guarded store -- and `fpu_add(x, 0)` is exact, so it really is one.

    """
    n = a.shape[0]
    piv = np.float32(a[t, t])
    recip = sfpu_recip(piv) if recip == "sfpu" else trunc_bf16_from_f32(
        np.float32(1.0) / piv)
    col = np.where(np.arange(n) >= t + 1, a[:, t], np.float32(0))
    row = np.where(np.arange(n) >= t, a[t, :], np.float32(0))
    R = np.full(n, recip, np.float32)
    if mul1 == "dm":
        # float32 on the mover, one rounding as it lands in the bf16 carrier
        m = trunc_bf16_from_f32(np.float32(recip) * col.astype(np.float32))
    elif mul1 == "elem":
        m = bcast_mul(col, R, **kw)          # SrcA = the column
    else:
        m = bcast_mul(R, col, **kw)          # SrcA = the reciprocal
    p = bcast_mul(np.repeat(m[:, None], n, 1),
                  np.repeat(row[None, :], n, 0), **kw)
    # the ELWSUB is binary_dest_reuse, not a broadcast, so it takes the
    # elementwise aligner -- the two rules meet inside one emitted step.
    if sub == "sfpu":
        return sfpu_add(a, -p, kw.get("sfpu", "rtz"),
                        kw.get("dest_bits", BF16_BITS))
    return fpu_add(a, -p)



SRCA_FID = (4, 5)   # explicit mantissa bits taken by fidelity phase 0 / 1
SRCB_FID = (6, 4)   # ... and by phase 0 / 2, for SrcB

# The matrix and elementwise units both use an 11-bit alignment field.
MVMUL_FIELD = SRC_BITS  # 11

# The product leaf width. The earlier apparent 12-bit peak was measured against
# the exponent of the largest *normalised* product. A normalisation carry makes
# that coordinate one bit finer. The two-term truth table separates the two:
# this is an 11-bit field at the pre-normalisation operand-exponent sum.
MVMUL_PRODUCT_FIELD = 11

def _fid_split(x, hi_bits, lo_bits, take_lo):
    """The operand mantissa bits this fidelity phase sees.

    SRCA_FID / SRCB_FID count EXPLICIT mantissa bits, the way the ISA masks are
    written; `_truncate_sig` counts significant bits, so the hidden one is added
    here. Off by one in either direction and the probe drops to 5-45%.
    """
    x = np.asarray(x, np.float64)
    hi = _truncate_sig(x, hi_bits + 1)
    return _truncate_sig(x - hi, lo_bits + 1) if take_lo else hi


def mvmul(srcb, srca, dst, field=MVMUL_FIELD, align="raz", combine="rhu",
          dst_align="raz", dest="raz", dest_bits=BF16_BITS,
          product_field=MVMUL_PRODUCT_FIELD, srcb_exp=None, srca_exp=None):
    """Dst += SrcB @ SrcA for one k-group.

    Products enter an 11-bit field anchored at the largest pre-normalisation
    operand-exponent sum, then accumulate onto Dst in its own 11-bit field.
    """
    srcb = np.asarray(srcb, np.float64)
    srca = np.asarray(srca, np.float64)
    dst = np.asarray(dst, np.float64)
    if product_field is None:
        x = srcb @ srca
        e = np.where(dst != 0, _exponent(dst), np.int32(-30000))
        lsb = np.ldexp(1.0, np.maximum(e, -120) - field)
        return rnd(_quantize(x, lsb, align) + dst, dest_bits, dest)
    else:
        p = srcb[:, :, None] * srca[None, :, :]          # exact: 5b x 7b
        # The product aligner is anchored before mantissa normalisation: add
        # the two operand exponents and subtract one. If the mantissa product
        # crosses 2, its normalisation carry therefore becomes an effective
        # extra bit. The field itself remains 11 bits.
        be = _exponent(np.abs(srcb)) if srcb_exp is None else srcb_exp
        ae = _exponent(np.abs(srca)) if srca_exp is None else srca_exp
        pe = be[:, :, None] + ae[None, :, :] - 1
        # The 16 lanes are two independent contiguous 8-lane reductions. Each
        # half has its own exponent anchor. Their fixed-point sums are then
        # aligned to the larger half-anchor on the same-width signed field;
        # that second aligner breaks ties toward +inf (rhu), exactly like the
        # broadcast aligner.
        xs, es = [], []
        for g in (slice(0, 8), slice(8, 16)):
            e = pe[:, g, :].max(axis=1, keepdims=True)
            es.append(e[:, 0, :])
            xs.append(_quantize(
                p[:, g, :],
                np.ldexp(1.0, np.maximum(e, -120) - product_field),
                align).sum(axis=1))
        # Dst is the third input of this final fixed-point combine, not a later
        # `Dst + completed_dot` operation. All three values share the largest
        # of the two product-half anchors and Dst's exponent.
        de = np.where(dst != 0, _exponent(dst), np.int32(-30000))
        e = np.maximum(np.maximum(es[0], es[1]), de)
        lsb = np.ldexp(1.0, np.maximum(e, -120) - field)
        x = (_quantize(xs[0], lsb, combine)
             + _quantize(xs[1], lsb, combine)
             + _quantize(dst, lsb, dst_align))
        return rnd(x, dest_bits, dest)


def matmul_block_tile(in0, in1t, dst=None, fidelity=2, **kw):
    """One k-tile of `matmul_block`, MVMUL by MVMUL in the MOP's order.

    `in0` is the SrcB tile (rows are output rows, columns are k); `in1t` is the
    SrcA tile already in its logical orientation (rows are k) -- the `transpose`
    flag is an unpacker/addr-mod matter and does not change the arithmetic.
    `fidelity` is the MathFidelity number: 2 = HiFi2 = phases 0 and 1.
    """
    d = np.zeros((in0.shape[0], in1t.shape[1]), np.float64) if dst is None         else np.asarray(dst, np.float64)
    for phase in range(fidelity if fidelity > 1 else 1):
        b = _fid_split(in0, SRCB_FID[0], SRCB_FID[1], bool(phase & 2))
        a = _fid_split(in1t, SRCA_FID[0], SRCA_FID[1], bool(phase & 1))
        # Fidelity masks do not renormalise their low mantissa chunks. Their
        # exponent anchors stay at the fixed bit position immediately below
        # the high chunk: original exponent - (hidden + explicit-high bits).
        # SrcA phase 1 therefore uses e-5, not exponent(x-high).
        be = _exponent(in0) - ((SRCB_FID[0] + 1) if phase & 2 else 0)
        ae = _exponent(in1t) - ((SRCA_FID[0] + 1) if phase & 1 else 0)
        for g in (slice(0, 16), slice(16, 32)):     # the MOP's two k-groups
            d = mvmul(b[:, g], a[g, :], d,
                      srcb_exp=be[:, g], srca_exp=ae[g, :], **kw)
    return d


def syrk_epilogue(c, acc, alpha=2.0, beta=3.0, **kw):
    """rn(C*beta) + rn(acc*alpha) -- two `mul_tiles` and one `add_binary_tile`.

    100.000000% bit-exact against the shipped run when `acc` is the device's own
    accumulator (read out by a probe that drops the SFPU add and packs DST1).
    Note the add is the SFPU one, so it TRUNCATES where the FPU rounds away.
    """
    return sfpu_add(fpu_mul(c, np.float32(beta), **kw),
                    fpu_mul(acc, np.float32(alpha), **kw))




