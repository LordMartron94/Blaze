package core

import "blaze/internal"

/* BlazeOperationID is a unique identifier for a given operation. */
type BlazeOperationID = internal.BlazeOperationID

const (
	Blaze_Operation_Vector_Sum BlazeOperationID = iota

	/* BLAZE_OPERATION_COUNT is a constant representing how manu operations are supported.

	This is NOT a valid operation ID.
	*/
	BLAZE_OPERATION_COUNT
)

/* BlazeDType represents the data type associated with data. */
type BlazeDType = internal.BlazeDataType

const (
	DTypeUnknown BlazeDType = iota

	// Float
	DTypeF64
	DTypeF32

	// Int
	DTypeI64
	DTypeI32
	DTypeI16
	DTypeI8

	// Uint
	DTypeU64
	DTypeU32
	DTypeU16
	DTypeU8
)

/* BlazeDTypeGet returns the BlazeDataDType corresponding to the generic type T. */
func BlazeDTypeGet[T any]() BlazeDType {
	var zero T
	switch any(zero).(type) {
	case float64:
		return DTypeF64
	case float32:
		return DTypeF32
	case int64:
		return DTypeI64
	case int32:
		return DTypeI32
	case int16:
		return DTypeI16
	case int8:
		return DTypeI8
	case uint64:
		return DTypeU64
	case uint32:
		return DTypeU32
	case uint16:
		return DTypeU16
	case uint8:
		return DTypeU8
	default:
		return DTypeUnknown
	}
}

/* BlazeExecutionFn performs a Blaze operation. */
type BlazeExecutionFn = internal.BlazeExecutionFn
