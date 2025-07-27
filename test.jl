
using JSON3
import CSV
using DataFrames

cd(@__DIR__)

println(read(`python3 test.py`, String))

