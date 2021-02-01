#!/usr/bin/env bash
export data_folder="criteo"

sh bash/LR.sh
sh bash/FM.sh
sh bash/FFM.sh
sh bash/FwFM.sh
sh bash/FvFM.sh
sh bash/FmFM.sh
sh bash/DCN.sh
