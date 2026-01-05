// -----------------------------------------------------------------------------
// BLAZE KERNEL FRAME ABI OFFSETS
// -----------------------------------------------------------------------------
// Struct: BlazeKernelFrame
// Layout:
//   Dim     [3]uint64        -> Size: 24 | Offsets: 0, 8, 16
//   Flags   uint64           -> Size: 8  | Offset:  24
//   Params  [6]uint64        -> Size: 48 | Offsets: 32, 40, 48, 56, 64, 72
//   Buffers [6]BufferSlot    -> Size: 96 | Start:   80
//   Ret     [2]uintptr       -> Size: 16 | Start:   176
//   Opaque  uintptr          -> Size: 8  | Offset:  192
//   Pad     [2]uint64        -> Size: 16 | End:     216 (Total Size aligned)
// -----------------------------------------------------------------------------

// ---- Execution Bounds ----
#define FRAME_DIM0       0
#define FRAME_DIM1       8
#define FRAME_DIM2       16

// ---- Configuration ----
#define FRAME_FLAGS      24

// ---- Parameters ----
#define FRAME_PARAM0     32
#define FRAME_PARAM1     40
#define FRAME_PARAM2     48
#define FRAME_PARAM3     56
#define FRAME_PARAM4     64
#define FRAME_PARAM5     72

// ---- Buffers ----
// Buffer 0
#define FRAME_BUF0_PTR   80
#define FRAME_BUF0_STR   88

// Buffer 1
#define FRAME_BUF1_PTR   96
#define FRAME_BUF1_STR   104

// Buffer 2
#define FRAME_BUF2_PTR   112
#define FRAME_BUF2_STR   120

// Buffer 3
#define FRAME_BUF3_PTR   128
#define FRAME_BUF3_STR   136

// Buffer 4
#define FRAME_BUF4_PTR   144
#define FRAME_BUF4_STR   152

// Buffer 5
#define FRAME_BUF5_PTR   160
#define FRAME_BUF5_STR   168

// ---- Returns ----
#define FRAME_RET0       176
#define FRAME_RET1       184

// ---- Opaque Context ----
#define FRAME_OPAQUE     192