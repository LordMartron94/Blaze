package blaze

import "blaze/simd"

/* BlazeInitialize initializes the dispatch table. */
func BlazeInitialize() {
	simd.BlazeSIMDDispatchInit()
}
