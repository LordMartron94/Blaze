package internal

var registry []KernelManifest

/* Register adds a kernel candidate to the global registry. */
func Register(m KernelManifest) {
	registry = append(registry, m)
}

/*
GetRegistry returns the full list of registered kernels.

	Used by the SIMD package to build the dispatch table.
*/
func GetRegistry() []KernelManifest {
	return registry
}
