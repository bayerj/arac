#!/bin/bash

OLDNAME="$1"
NEWNAME="${OLDNAME/.cpp/.c}"

mv -f "$OLDNAME" "$NEWNAME"

git mv "$NEWNAME" "$OLDNAME"
