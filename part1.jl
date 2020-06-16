#=-----------------------------------------------=#

struct Item
    weight::Float64
    value::Float64
end

#=-----------------------------------------------=#


function Knapsack(capacity, items)
    current_weight = current_value = 0
    chosen_weight = 0
	chosen_value = 0
    path_taken = []
    chosen_solution = []
    size_ = length(items)
	for first in 1:size_
        path_taken = [items[first]]
        current_weight = items[first].weight
        current_value = items[first].value
        for second in 1:size_
            if(first != second)
                if((current_weight + items[second].weight) <= capacity)
                    #=Add to path=#
					push!(path_taken,items[second])
					
					#=Change currents=#
					current_value += items[second].value
                    current_weight += items[second].weight
                end
            end
        end
        if (chosen_value < current_value)
            chosen_value = current_value
            chosen_weight = current_weight
            chosen_solution = path_taken
        end
    end
	return chosen_weight, chosen_value, path_taken
end
#=-----------------------------------------------=#

weights = Float64[]
values = Float64[]

println("-------------NO OF ITEMS-------------------")


print("Enter the number of items: ")
n = parse(Int64, readline())

println("-------------POPULATE-------------------")

for x in 1:n
    print("WEIGHT ", x, " :")
	w = parse(Int64, readline())
	print("VALUE ", x, " :")
	v = parse(Int64, readline())
	append!(weights, w)
	append!(values, v)
end

println("----------------- PRINT -------------------")

println(weights)
println(values)

println("ENTER KNACP CAPACITY ")
capacity = parse(Int64, readline())

println("-------------------------------------")

items = Array{Item, 1}(undef, n)
for i in 1:n
	items[i] = Item(weights[i], values[i])
end

println(items)

println("-------------------------------------")

chosen_weight,chosen_value, solution = Knapsack(capacity, items)

println("------------SOLUTION------------")
println("WEIGHT   : ", chosen_weight)
println("VALUE    : ", chosen_value)
println("SOLUTION -> ")

for step in solution
    println("STEP: ", step)
end
println("--------------END---------------")