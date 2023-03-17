using CSV
using DataFrames

df = CSV.read("../Python Scripts/pcd_analysis/velodyne_10frame_merge_without_empty_2.csv", DataFrame)
print(names(df))