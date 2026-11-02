# Turns a binary file into a C header with a byte array.
# Usage: cmake -DINPUT=<file> -DOUTPUT=<header> -DSYMBOL=<name> -P embed_file.cmake
file(READ "${INPUT}" hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
# 16 bytes per line so the generated source stays readable for compilers.
string(REPEAT "0x[0-9a-f][0-9a-f]," 16 line_pattern)
string(REGEX REPLACE "(${line_pattern})" "\\1\n" bytes "${bytes}")
file(WRITE "${OUTPUT}"
  "// Generated from ${INPUT} by embed_file.cmake - do not edit.\n"
  "#pragma once\n"
  "static const unsigned char ${SYMBOL}[] = {\n${bytes}\n};\n")
