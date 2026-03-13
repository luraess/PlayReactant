using Reactant

input1 = Reactant.ConcreteRArray(ones(10))
input2 = Reactant.ConcreteRArray(ones(10))

function sinsum_add(x, y)
   return sum(sin.(x) .+ y)
end

f = @compile sinsum_add(input1,input2)

f(input1, input2)

function add(a, b)
   a + b
end

x = ConcreteRNumber(3)
y = ConcreteRNumber(4)

addxy = @compile add(x, y)

res = addxy(x, y)

res = addxy(ConcreteRNumber(7), ConcreteRNumber(8))

addx4 = @compile add(x, 4)

res = addx4(x, 4)

res = addx4(ConcreteRNumber(7), 8)

t = Reactant.to_rarray(0.5; track_numbers=true)

addx4 = @compile add(x, t)

t = Reactant.to_rarray(10.5; track_numbers=true)

res = addx4(ConcreteRNumber(7), t)
