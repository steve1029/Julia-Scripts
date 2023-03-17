import Formatting as fmt

for i=1:2:10
    println(i)
end

fmt.printfmtln("{1:d}, {3:>010.5f}, {2:c}", 3, 'f', 3.34)

ff = 3.14
fmt.printfmtln("{1:f}", ff)