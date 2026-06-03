/*
 * GF(2) matrix rank – M4RI with quick-check + lazy-flip Phase 1.
 *
 * Standard column-by-column Gaussian elimination is memory-bandwidth-limited:
 * each pivot step touches every row of the matrix once.  M4RI processes k
 * columns per "round" by precomputing all 2^k XOR combinations of the k
 * pivot rows into a lookup table; every other row is updated in a single
 * table-lookup + XOR pass.
 *
 * Phase 1 correctness + performance
 * ----------------------------------
 * The k RREF pivot rows within a block must satisfy: row b has 1 at c_b
 * and 0 at all other c_j.  This is required for the lookup-table XOR to
 * zero the correct set of bits.  When searching for a new pivot at column c,
 * reducing candidate row r by the existing pivots may flip r's bit at c;
 * we must check the net flip parity before committing.
 *
 * Optimisation: quick-check + lazy flip
 *   1. First check: is bit_c set in row r (one word read)?  If not, skip.
 *   2. Only for the (rare, sparse) rows where bit_c = 1, compute the flip
 *      parity from the k existing pivot rows.  If the pivot column c_b is
 *      in the same word as c, the word is already in a register – zero extra
 *      memory reads.  Only genuinely scattered pivots cost an extra load.
 * For sparse matrices the quick-check eliminates ~(1 – density) fraction of
 * rows before any flip computation, keeping Phase 1 efficient.
 *
 * Exported symbols:
 *
 *   int gf2_rank_packed(
 *       uint64_t *packed, int nrows, int ncols,
 *       progress_fn cb, void *userdata, int cb_interval)
 *
 *     Compute rank.  `packed[r*nwords + w]` holds bits [64w .. 64w+63] of
 *     row r.  The array is modified in-place; pass a copy if needed.
 *     cb (optional): called as cb(col, ncols, userdata) every cb_interval
 *     pivot columns found.  Pass NULL to disable.
 *
 *   void gf2_transpose_packed(
 *       const uint64_t *src, int src_nrows, int src_ncols,
 *       uint64_t *dst)
 *
 *     Write A^T into dst.
 *     dst must hold ceil(src_nrows/64) * src_ncols uint64_t words.
 *
 * Compile:
 *   gcc -O3 -march=native -shared -fPIC _gf2_rank.c -o _gf2_rank.so
 * With OpenMP (recommended):
 *   gcc -O3 -march=native -fopenmp -shared -fPIC _gf2_rank.c -o _gf2_rank.so
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

typedef int (*progress_fn)(int col, int ncols, void *userdata);

/* Number of pivot columns per M4RI round.  2^M4RI_K entries in the lookup
 * table.  k=8 → 256-entry table, fits comfortably in L1 cache.          */
#define M4RI_K 8
#define M4RI_L (1 << M4RI_K)

/* -----------------------------------------------------------------------
 * Helpers
 * --------------------------------------------------------------------- */

static inline void swap_rows(uint64_t *packed, int nwords, int a, int b)
{
    uint64_t *ra = packed + (size_t)a * nwords;
    uint64_t *rb = packed + (size_t)b * nwords;
    for (int w = 0; w < nwords; w++) {
        uint64_t t = ra[w]; ra[w] = rb[w]; rb[w] = t;
    }
}

static inline void xor_rows(uint64_t *packed, int nwords, int dst, int src)
{
    uint64_t *d = packed + (size_t)dst * nwords;
    const uint64_t *s = packed + (size_t)src * nwords;
    for (int w = 0; w < nwords; w++)
        d[w] ^= s[w];
}

/* -----------------------------------------------------------------------
 * M4RI rank
 * --------------------------------------------------------------------- */

int gf2_rank_packed(
    uint64_t       *packed,
    int             nrows,
    int             ncols,
    progress_fn     cb,
    void           *userdata,
    int             cb_interval)
{
    if (nrows == 0 || ncols == 0) return 0;

    int nwords = (ncols + 63) >> 6;
    int rank   = 0;

    /* Lookup table: T[idx * nwords .. (idx+1)*nwords - 1] for idx in
     * 0..2^k-1.  T[idx] = XOR of pivot rows {b : bit b of idx = 1}.     */
    uint64_t *T = (uint64_t *)malloc((size_t)M4RI_L * nwords * sizeof(uint64_t));
    if (!T) {
        /* Out of memory: plain column-by-column elimination, forward only */
        for (int col = 0; col < ncols && rank < nrows; col++) {
            int word = col >> 6, sh = col & 63;
            uint64_t bitm = (uint64_t)1 << sh;
            int pivot = -1;
            for (int r = rank; r < nrows; r++)
                if (packed[(size_t)r * nwords + word] & bitm) { pivot = r; break; }
            if (pivot < 0) continue;
            if (pivot != rank) swap_rows(packed, nwords, rank, pivot);
            for (int r = rank + 1; r < nrows; r++)
                if (packed[(size_t)r * nwords + word] & bitm)
                    xor_rows(packed, nwords, r, rank);
            rank++;
        }
        return rank;
    }

    /* Pivot columns for the current M4RI round.                             */
    int piv_cols[M4RI_K];
    int piv_col = 0;   /* first unprocessed column                          */

    while (piv_col < ncols && rank < nrows) {

        /* ---------------------------------------------------------------
         * Phase 1: find up to M4RI_K RREF pivot rows.
         *
         * For each candidate column c, building RREF k pivot rows requires
         * that we find a row whose *reduced* form (after XOR-ing in the
         * existing pivots) still has bit_c = 1.
         *
         * Reducing a candidate row r by pivot b flips r's bit at c iff
         * pivot row b has bit_c = 1 AND r has bit at c_b.  So we must
         * check the net flip parity before committing.
         *
         * Optimisation: when no existing pivot row has bit_c set, flip_parity
         * is 0 for every candidate row, so a plain bit_c test suffices (fast
         * path).  For sparse matrices this covers ~99% of columns.  When some
         * pivot does have bit_c set, we fall back to the full computation –
         * crucially including rows with bit_c=0 that become valid after
         * reduction (flip_parity=1 makes final_bit=1).  If the pivot column
         * c_b is in the same word as c, we reuse the already-loaded word.
         * -------------------------------------------------------------- */
        int k = 0;

        for (int c = piv_col; c < ncols && k < M4RI_K && rank + k < nrows; c++) {
            int word = c >> 6, sh = c & 63;
            uint64_t bitm = (uint64_t)1 << sh;

            /* Precompute: does pivot row b have bit_c set? (k reads, pivot
             * rows are hot in cache.)                                       */
            int pv_bit_c[M4RI_K];
            for (int b = 0; b < k; b++)
                pv_bit_c[b] = (packed[(size_t)(rank + b) * nwords + word] >> sh) & 1;

            /* When no pivot has bit_c set, flip_parity = 0 for every row.
             * We can use a cheap bit_c check to find the pivot.            */
            int any_pv_bit_c = 0;
            for (int b = 0; b < k; b++) any_pv_bit_c |= pv_bit_c[b];

            /* Pivot search.                                                 */
            int pivot = -1;
            if (!any_pv_bit_c) {
                /* Fast path (~99% of columns for sparse matrices).         */
                for (int r = rank + k; r < nrows; r++) {
                    if (packed[(size_t)r * nwords + word] & bitm) {
                        pivot = r; break;
                    }
                }
            } else {
                /* Slow path: flip_parity needed; also covers bit_c=0 rows
                 * that become valid after reduction.                        */
                for (int r = rank + k; r < nrows; r++) {
                    uint64_t rw_c = packed[(size_t)r * nwords + word];
                    int flip_parity = 0;
                    for (int b = 0; b < k; b++) {
                        if (!pv_bit_c[b]) continue;
                        int cb = piv_cols[b];
                        if ((cb >> 6) == word) {
                            if (rw_c & ((uint64_t)1 << (cb & 63))) flip_parity ^= 1;
                        } else {
                            if (packed[(size_t)r * nwords + (cb >> 6)] &
                                ((uint64_t)1 << (cb & 63)))
                                flip_parity ^= 1;
                        }
                    }
                    if (((rw_c >> sh) & 1) ^ flip_parity) { pivot = r; break; }
                }
            }
            if (pivot < 0) continue;

            /* Swap to rank+k, reduce by existing pivots, eliminate c.      */
            if (pivot != rank + k)
                swap_rows(packed, nwords, rank + k, pivot);

            for (int b = 0; b < k; b++) {
                int cb = piv_cols[b];
                if (packed[(size_t)(rank + k) * nwords + (cb >> 6)] &
                    ((uint64_t)1 << (cb & 63)))
                    xor_rows(packed, nwords, rank + k, rank + b);
            }

            for (int r = rank; r < rank + k; r++)
                if (packed[(size_t)r * nwords + word] & bitm)
                    xor_rows(packed, nwords, r, rank + k);

            piv_cols[k++] = c;
        }

        if (k == 0) break;   /* no more independent columns */

        /* ---------------------------------------------------------------
         * Phase 2: build the lookup table T.
         * T[0]    = zero row
         * T[1<<b] = pivot row (rank+b)
         * T[i]    = T[i ^ lowbit(i)] XOR T[lowbit(i)]
         * -------------------------------------------------------------- */
        memset(T, 0, (size_t)M4RI_L * nwords * sizeof(uint64_t));
        for (int i = 1; i < (1 << k); i++) {
#ifdef __GNUC__
            int bit = __builtin_ctz(i);
#else
            int bit = 0;
            { int tmp = i; while (!(tmp & 1)) { tmp >>= 1; bit++; } }
#endif
            uint64_t       *dst  = T + (size_t) i            * nwords;
            const uint64_t *src1 = T + (size_t)(i ^ (1 << bit)) * nwords;
            const uint64_t *src2 = packed + (size_t)(rank + bit) * nwords;
            for (int w = 0; w < nwords; w++)
                dst[w] = src1[w] ^ src2[w];
        }

        /* ---------------------------------------------------------------
         * Phase 3: sweep all non-pivot rows.
         *
         * Build a compact active-set (rows with at least one pivot bit set)
         * to skip zero rows without loading full row data.  OMP parallelises
         * the XOR pass over active rows.
         * -------------------------------------------------------------- */
        int piv_words[M4RI_K];
        uint64_t piv_bits[M4RI_K];
        for (int b = 0; b < k; b++) {
            piv_words[b] = piv_cols[b] >> 6;
            piv_bits[b]  = (uint64_t)1 << (piv_cols[b] & 63);
        }

        int *active_rows = (int *)malloc((size_t)(nrows - k) * sizeof(int));
        int  n_active    = 0;

        if (active_rows) {
            for (int r = 0; r < nrows; r++) {
                if (r >= rank && r < rank + k) continue;
                uint32_t idx = 0;
                for (int b = 0; b < k; b++)
                    if (packed[(size_t)r * nwords + piv_words[b]] & piv_bits[b])
                        idx |= (uint32_t)1 << b;
                if (idx)
                    active_rows[n_active++] = (r << 8) | (int)idx;
            }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) \
            shared(packed, T, active_rows, n_active, nwords)
#endif
            for (int i = 0; i < n_active; i++) {
                int entry   = active_rows[i];
                int r       = entry >> 8;
                int idx     = entry & 0xFF;
                uint64_t       *row = packed + (size_t)r * nwords;
                const uint64_t *t   = T + (size_t)idx * nwords;
                for (int w = 0; w < nwords; w++)
                    row[w] ^= t[w];
            }
            free(active_rows);
        } else {
            for (int r = 0; r < nrows; r++) {
                if (r >= rank && r < rank + k) continue;
                uint32_t idx = 0;
                for (int b = 0; b < k; b++)
                    if (packed[(size_t)r * nwords + piv_words[b]] & piv_bits[b])
                        idx |= (uint32_t)1 << b;
                if (!idx) continue;
                uint64_t       *row = packed + (size_t)r * nwords;
                const uint64_t *t   = T + (size_t)idx * nwords;
                for (int w = 0; w < nwords; w++)
                    row[w] ^= t[w];
            }
        }

        /* Progress callback.                                               */
        if (cb && cb_interval > 0) {
            rank += k;
            if ((rank / cb_interval) != ((rank - k) / cb_interval))
                cb(rank, nrows < ncols ? nrows : ncols, userdata);
            rank -= k;
        }

        piv_col = piv_cols[k - 1] + 1;
        rank   += k;
    }

    free(T);
    if (cb) cb(rank, rank, userdata);
    return rank;
}


/* -----------------------------------------------------------------------
 * Bit-level transpose using 64×64 tile kernel
 * --------------------------------------------------------------------- */

/*
 * Transpose a 64×64 bit matrix stored as buf[0..63] where buf[i] is row i.
 * Column j of the result becomes row j.
 *
 * Uses the standard "bit-level merge sort" approach:
 * for step in [32, 16, 8, 4, 2, 1]:
 *   process pairs of rows (i, i+step) in groups of size 2*step:
 *     swap the upper `step` bits of buf[i] with the lower `step` bits of buf[i+step]
 */
static void transpose_64x64(uint64_t *buf)
{
    static const uint64_t masks[6] = {
        UINT64_C(0x00000000FFFFFFFF),  /* step=32 */
        UINT64_C(0x0000FFFF0000FFFF),  /* step=16 */
        UINT64_C(0x00FF00FF00FF00FF),  /* step=8  */
        UINT64_C(0x0F0F0F0F0F0F0F0F),  /* step=4  */
        UINT64_C(0x3333333333333333),  /* step=2  */
        UINT64_C(0x5555555555555555),  /* step=1  */
    };
    int steps[6] = { 32, 16, 8, 4, 2, 1 };

    for (int s = 0; s < 6; s++) {
        int step = steps[s];
        uint64_t mask = masks[s];
        for (int grp = 0; grp < 64; grp += 2 * step) {
            for (int i = 0; i < step; i++) {
                uint64_t x = buf[grp + i];
                uint64_t y = buf[grp + i + step];
                uint64_t t = ((x >> step) ^ y) & mask;
                buf[grp + i]        = x ^ (t << step);
                buf[grp + i + step] = y ^ t;
            }
        }
    }
}

void gf2_transpose_packed(
    const uint64_t *src,
    int             src_nrows,
    int             src_ncols,
    uint64_t       *dst)
{
    if (src_nrows == 0 || src_ncols == 0) return;

    int src_nwords = (src_ncols + 63) >> 6;
    int dst_nwords = (src_nrows + 63) >> 6;

    memset(dst, 0, (size_t)src_ncols * dst_nwords * sizeof(uint64_t));

    uint64_t tile[64];

    for (int tile_r = 0; tile_r < src_nrows; tile_r += 64) {
        int r_end      = tile_r + 64 < src_nrows ? tile_r + 64 : src_nrows;
        int tile_r_cnt = r_end - tile_r;

        for (int tile_c = 0; tile_c < src_ncols; tile_c += 64) {
            int c_end      = tile_c + 64 < src_ncols ? tile_c + 64 : src_ncols;
            int tile_c_cnt = c_end - tile_c;

            /* extract tile: tile[i] holds columns [tile_c .. tile_c+63] of
             * src row (tile_r + i), zero-padded for out-of-bounds rows.    */
            memset(tile, 0, sizeof(tile));

            int tc_word = tile_c >> 6;
            int tc_sh   = tile_c & 63;

            if (tc_sh == 0) {
                for (int i = 0; i < tile_r_cnt; i++)
                    tile[i] = src[(size_t)(tile_r + i) * src_nwords + tc_word];
            } else {
                for (int i = 0; i < tile_r_cnt; i++) {
                    size_t base = (size_t)(tile_r + i) * src_nwords;
                    uint64_t lo = src[base + tc_word] >> tc_sh;
                    uint64_t hi = (tc_word + 1 < src_nwords)
                                  ? src[base + tc_word + 1] << (64 - tc_sh)
                                  : UINT64_C(0);
                    tile[i] = lo | hi;
                }
            }

            /* mask off column bits beyond src_ncols */
            if (tile_c_cnt < 64) {
                uint64_t col_mask = ((uint64_t)1 << tile_c_cnt) - 1;
                for (int i = 0; i < 64; i++) tile[i] &= col_mask;
            }

            /* transpose the 64×64 tile in-place */
            transpose_64x64(tile);

            /* scatter: after transpose, tile[j] holds src rows [tile_r..tile_r+63]
             * for src column (tile_c + j).  In dst: row = tile_c+j, word = tile_r>>6 */
            int dr_word = tile_r >> 6;
            int dr_sh   = tile_r & 63;

            if (dr_sh == 0) {
                for (int j = 0; j < tile_c_cnt; j++)
                    dst[(size_t)(tile_c + j) * dst_nwords + dr_word] |= tile[j];
            } else {
                for (int j = 0; j < tile_c_cnt; j++) {
                    size_t base = (size_t)(tile_c + j) * dst_nwords;
                    dst[base + dr_word]     |= tile[j] << dr_sh;
                    if (dr_word + 1 < dst_nwords)
                        dst[base + dr_word + 1] |= tile[j] >> (64 - dr_sh);
                }
            }
        }
    }
}
