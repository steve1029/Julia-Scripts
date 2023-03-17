using DataFrames
using XLSX
using Printf

xf = XLSX.readxlsx("../Python Scripts/pcd_analysis/velodyne_10frame_merge_without_empty_2.xlsx")

shs = XLSX.sheetnames(xf)
print(shs)