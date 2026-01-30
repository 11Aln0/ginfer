#pragma once

// cp.async instructions
#define CP_ASYNC_COMMIT_GROUP() \
    asm volatile(                     \
        "cp.async.commit_group;\n"   \
    );

#define CP_ASYNC_WAIT_ALL() \
    asm volatile(                     \
        "cp.async.wait_all;\n"       \
    );

#define CP_ASYNC_WAIT_GROUP(n) \
    asm volatile(                     \
        "cp.async.wait_group %0;\n"  \
        :                             \
        : "n"(n)                     \
    );                                

#define CP_ASYNC_CA(dst, src, bytes) \
    asm volatile(                                 \
        "cp.async.ca.shared.global [%0], [%1], %2;\n" \
        :                                         \
        : "r"(dst), "l"(src), "n"(bytes)         \
    );

#define CP_ASYNC_CG(dst, src, bytes) \
    asm volatile(                                 \
        "cp.async.cg.shared.global [%0], [%1], %2;\n" \
        :                                         \
        : "r"(dst), "l"(src), "n"(bytes)         \
    );

#define CP_ASYNC_CG_GUARDED(dst, src, cp_bytes, src_bytes) \
asm volatile(                                         \
    "cp.async.cg.shared.global "                      \
    "[%0], [%1], %2, %3;\n"                           \
    :                                                 \
    : "r"(dst),                                       \
      "l"(src),                                       \
      "n"(cp_bytes),                                  \
      "r"(src_bytes)                                  \
)

// matrix load instructions
#define LDMATRIX_X2_B16(R0, R1, addr) \
    asm volatile(                                 \
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" \
        : "=r"(R0), "=r"(R1)                      \
        : "r"(addr)                               \
    )

#define LDMATRIX_X4_B16(R0, R1, R2, R3, addr) \
    asm volatile(                                 \
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)  \
        : "r"(addr)                               \
    )

#define LDMATRIX_X2_TRANS_B16(R0, R1, addr) \
    asm volatile(                                 \
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n" \
        : "=r"(R0), "=r"(R1)                      \
        : "r"(addr)                               \
    )

#define LDMATRIX_X4_TRANS_B16(R0, R1, R2, R3, addr) \
    asm volatile(                                 \
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)  \
        : "r"(addr)                               \
    )

// mma
#define MMA_FP16_ACCFP32(RACC0, RACC1, RACC2, RACC3, RA0, RA1, RA2, RA3, RB0, RB1) \
    asm volatile(                                                                  \
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                       \
        "{%0, %1, %2, %3}, " \
        "{%4, %5, %6, %7}, " \
        "{%8, %9}, " \
        "{%0, %1, %2, %3};\n" \
        : "+f"(RACC0), "+f"(RACC1), "+f"(RACC2), "+f"(RACC3) \
        : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), \
          "r"(RB0), "r"(RB1) \
    )

#define MMA_FP16_ACCFP16(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) \
    asm volatile(                                                    \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "         \
        "{%0, %1}, " \
        "{%2, %3, %4, %5}, " \
        "{%6, %7}, " \
        "{%8, %9};\n" \
        : "=r"(RD0), "=r"(RD1) \
        : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), \
          "r"(RB0), "r"(RB1), \
          "r"(RC0), "r"(RC1) \
    )

// vectorized global load/store for half2
#define ST_GLOBAL_PRED_U32(ptr, val, pred_guard)       \
    asm volatile(                                       \
        "{\n"                                           \
        "  .reg .pred p;\n"                             \
        "  setp.ne.b32 p, %2, 0;\n"                     \
        "  @p st.global.u32 [%0], %1;\n"                \
        "}\n"                                           \
        :                                               \
        : "l"(ptr),                                     \
          "r"(reinterpret_cast<uint32_t const &>(val)), \
          "r"((int)(pred_guard))                         \
    )


#define WARP_SIZE 32