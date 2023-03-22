# println(readdir())
# println(dirname(pwd()))
# println(pwd())
onlynames = readdir(join=false)
names = readdir(join=true)

# (root, dirs, files) = walkdir(".")

# println(root)
# println(typeof(root))

# println(dirs)
# println(files)

root = pwd()
println(root)

for (name, onlyname) in zip(names, onlynames)

    # println(name)
    # println(onlyname)

    if isdir(name) == true
        inners = readdir(name, join=true, sort=true)
        onlyinners = readdir(name, join=false, sort=true)
        for (inner, onlyinner) in zip(inners, onlyinners)

            if endswith(onlyinner, "html") == true
                newpath = joinpath(root, onlyinner)
                mv(inner, newpath)
                # println(inner)
            else
                newname = string(onlyname, '-', onlyinner)
                newpath = joinpath(root, newname)
                println(inner)
                println(newpath)
                mv(inner, newpath)
            end

        end
    end
end

#=
for (root, dirs, files) in walkdir(".")
    for dir in dirs
        # println(file)
        path = joinpath(root, dir)
        inner = readdir(path, join=true, sort=true)
        println(inner)
        if endswith(inner) == "pdf"
            println(pdf)
        end
    end
end

=#