package internal

/* BlazeOperationID is a unique identifier for a given operation. */
type BlazeOperationID uint16

/* BlazeDataType represents the data type associated with data. */
type BlazeDataType uint8

/*
	PackSignature creates a unique ID for a type combination.

We support up to 3 inputs + 1 output (Standard FMA/Ternary limit).
Layout: [ Out(8) | In2(8) | In1(8) | In0(8) ]
*/
func PackSignature(out BlazeDataType, inputs ...BlazeDataType) uint32 {
	var sig = uint32(out) << 24
	if len(inputs) > 0 {
		sig |= uint32(inputs[0])
	}
	if len(inputs) > 1 {
		sig |= uint32(inputs[1]) << 8
	}
	if len(inputs) > 2 {
		sig |= uint32(inputs[2]) << 16
	}
	return sig
}
