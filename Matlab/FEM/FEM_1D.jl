import Pkg; Pkg.add("Plots")
import Plots as plt

Ne = 20
Nn = Ne + 1
K = fill(0, (Nn, Nn))
b = zeros(Nn)

function elmconn(e,i)

end

for e=1:Ne
    for i=1:2
        for j=1:2
            K(elmconn(e,i), elmconn(e,j)) += Ke(i,j)
        end
    end
end