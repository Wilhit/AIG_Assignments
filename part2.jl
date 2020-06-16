function getBest(max_branch,alpha, beta)
    best = alpha
    for i in 1:2
        val = find_solution(max_branch - 1,i,alpha,beta,false,components)
        best = best >= val ? best : val
        alpha = alpha >= val ? alpha : val
        if beta <= alpha
            break
        end
    end
    return best
end

function alternate(max_branch,alpha, beta)
    best = beta
    for i in 1:2
        val = find_solution(max_branch - 1,i,alpha,beta,true,components)
        best = best <= val ? best : val
        beta = beta <= val ? beta : val
        if beta <= alpha
            break
        end
    end
    return best
end

function find_solution(max_branch, index,alpha,beta,condition,components)
    if max_branch == 1
        first_component = components[index]
		return first_component
    end
    if condition
        best = getBest(max_branch,alpha,beta)
		return best
		
    else
        best = alternate(max_branch,alpha,beta)
		return best
    end
end

println("-------------NO OF ITEMS-------------------")


print("Enter the number of items: ")
n = parse(Int64, readline())

println("-------------POPULATE-------------------")

components = []

for x in 1:n
    print("ENTER ", x, " :")
	i = parse(Int64, readline())
	push!(components, i)
end

print("ENTER ALPHA VALUE", " :")
maximum = parse(Int64, readline())

print("ENTER BETA VALUE", " :")
minimim = parse(Int64, readline())

print("ENTER MAX NO BRANCHES", " :")
max_branch = parse(Int64, readline())

print("ENTER MAX NO  of PLAYERS", " :")
players = parse(Int64, readline())

print("START GAME: (Y/N)")
start = readline()
if start == "Y"
    start = true
    xx = find_solution(max_branch,1,minimim,maximum,start,components)
    println("------SOLUTION_-----")
    println(xx)
else
	print("GAME CANCELLED")
end
