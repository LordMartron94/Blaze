package core

import (
	"echo"
	"essence"
)

const strRepr = "bdc9ab6e-551c-41ea-adb4-ed4e43d105af"

var BlazeUUID essence.UUID

func init() {
	genID, _ := essence.UUIDFromString(strRepr)
	BlazeUUID = genID

	echo.EchoSystemRegister(genID, echo.EchoSystemConfiguration{
		MinLogLevel:    echo.DEBUG,
		SystemPrefixes: []string{"Blaze"},
	})
}
