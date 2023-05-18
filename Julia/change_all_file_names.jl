function recursive_search(dir::AbstractString)
    for file in readdir(dir)
        if isdir(joinpath(dir, file))
            recursive_search(joinpath(dir, file))
        else
            if endswith(file, ".mp4")
                # 파일 이름 변경
                new_name = replace(file, ".mp4" => "_new.mp4")
                # 파일 이동
                mv(joinpath(dir, file), joinpath(pwd(), new_name))
            end
        end
    end
end

# 현재 폴더에서 검색 시작
recursive_search(pwd())
